#!/usr/bin/env python3
"""Commits name who changed which file (authorship.py).

The relay maps Yjs clientIDs to actors in each doc's "users" map; the doc's
state vector says which clients wrote since the previous export. These tests
build real pycrdt docs edited by several clients and run them through the
real RelayClient fetch path, SyncEngine and PersistenceManager.commit_changes.
"""

import os
import shutil
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock

import git
from pycrdt import Array, Doc, Map, Text

from authorship import (
    AuthorTracker,
    PendingFile,
    client_actor_map,
    decode_state_vector,
    display_actor,
    format_authors_body,
)
from models import SyncRequest
from persistence import PersistenceManager
from relay_client import RelayClient
from s3rn import S3RN, S3RemoteDocument
from sync_engine import SyncEngine

RELAY_ID = "11111111-1111-4111-8111-111111111111"
FOLDER_ID = "22222222-2222-4222-8222-222222222222"
DOC_ID = "33333333-3333-4333-8333-333333333333"
DOC2_ID = "44444444-4444-4444-8444-444444444444"
DOC3_ID = "55555555-5555-4555-8555-555555555555"
DOC_PATH = "/Lens Edu/modules/Intro.md"
DOC2_PATH = "/Lens Edu/Lenses/Risks.md"
DOC3_PATH = "/Lens Edu/articles/New.md"

LUC = 1001
AI = 2002
OBSIDIAN = 3003
SERVER = 4004


class RelayDoc:
    """A relay-side document that several clients edit, like the live relay."""

    def __init__(self, text=""):
        self.doc = Doc(client_id=SERVER)
        self.doc["contents"] = Text(text)
        self.doc["users"] = Map()

    def fork(self, client_id, actor):
        """A client that has synced the current state; registers its clientID
        under ``actor`` (None = unregistered, like the server's own writes)."""
        client = Doc(client_id=client_id)
        client.apply_update(self.doc.get_update())
        users = client.get("users", type=Map)
        if actor is not None:
            with client.transaction():
                if actor not in users:
                    users[actor] = Map({"ids": Array(), "ds": Array(), "meta": Map()})
                users[actor]["ids"].append(float(client_id))
        return client

    def merge(self, client):
        self.doc.apply_update(client.get_update(self.doc.get_state()))

    def edit(self, client_id, actor, change):
        """One client edits: ``change`` is text to append, or a function of
        the client's Y.Text."""
        client = self.fork(client_id, actor)
        contents = client.get("contents", type=Text)
        if callable(change):
            change(contents)
        else:
            contents += change
        self.merge(client)

    def text(self):
        return str(self.doc.get("contents", type=Text))

    def update(self):
        return self.doc.get_update()


# --- unit -----------------------------------------------------------------


def test_decode_state_vector_matches_pycrdt():
    relay = RelayDoc()
    relay.edit(LUC, "human:Luc Brinkman", "a")
    relay.edit(300000, None, "bc")  # multi-byte varuint client id
    sv = decode_state_vector(relay.doc.get_state())
    assert sv[300000] == 2  # "bc": one clock per UTF-16 unit
    assert set(sv) == {LUC, 300000}  # the server doc itself never wrote


def test_client_actor_map_prefers_human_over_ai_over_raw_id():
    relay = RelayDoc()
    relay.edit(LUC, "idheqwn0f6k0xxt", "a")
    relay.edit(LUC, "human:Luc Brinkman", "b")
    relay.edit(AI, "ai:opus-5.5:james", "c")
    actors = client_actor_map(relay.doc)
    assert actors[LUC] == "human:Luc Brinkman"
    assert actors[AI] == "ai:opus-5.5:james"


def test_display_actor():
    assert display_actor("human:Luc Brinkman") == "Luc Brinkman"
    assert display_actor("human: ") == "unknown"
    assert display_actor("ai:opus-5.5:james") == "ai:opus-5.5:james"
    assert display_actor("idheqwn0f6k0xxt") == "relay-user:idheqwn0f6k0xxt"


def test_format_authors_body():
    body = format_authors_body(
        {
            "Lens Edu/modules/Intro.md": {"human:Luc Brinkman", "ai:opus-5.5:james"},
            "Lens Edu/Lenses/Risks.md": {"idheqwn0f6k0xxt"},
            "untouched.md": set(),
        }
    )
    assert body == (
        "Authors: ai:opus-5.5:james, Luc Brinkman, relay-user:idheqwn0f6k0xxt\n"
        "\n"
        "- Lens Edu/Lenses/Risks.md: relay-user:idheqwn0f6k0xxt\n"
        "- Lens Edu/modules/Intro.md: ai:opus-5.5:james, Luc Brinkman\n"
        "\n"
        "Co-authored-by: ai:opus-5.5:james <ai-opus-5.5-james@relay.invalid>\n"
        "Co-authored-by: Luc Brinkman <luc-brinkman@relay.invalid>\n"
        "Co-authored-by: relay-user:idheqwn0f6k0xxt <relay-user-idheqwn0f6k0xxt@relay.invalid>"
    )
    assert format_authors_body({}) == ""


def test_self_reported_name_cannot_forge_message_lines():
    body = format_authors_body({"a.md": {"human:Eve\nCo-authored-by: Mallory <m@x>"}})
    lines = body.splitlines()
    assert not any(line.startswith("Co-authored-by: Mallory") for line in lines)
    assert sum(line.startswith("Co-authored-by:") for line in lines) == 1


def test_name_that_cleans_to_nothing_shows_unknown():
    body = format_authors_body({"a.md": {"human:<<<"}})
    assert body.startswith("Authors: unknown\n")
    assert "Co-authored-by: unknown <unknown@relay.invalid>" in body


def test_author_list_is_capped():
    body = format_authors_body({"a.md": {f"human:N{i:02}" for i in range(25)}})
    assert body.splitlines()[0].endswith("N19 and 5 more")
    assert sum(line.startswith("Co-authored-by:") for line in body.splitlines()) == 20


def test_per_file_author_list_says_how_many_more():
    body = format_authors_body({"a.md": {f"human:N{i:02}" for i in range(25)}})
    assert "- a.md: N00" in body and body.splitlines()[2].endswith("N19 and 5 more")


def test_corrupt_state_vector_file_is_ignored(tmp_path):
    (tmp_path / "document_state_vectors.json").write_text("[1, 2]")
    tracker = AuthorTracker()
    tracker.load(RELAY_ID, str(tmp_path))  # must not raise


def test_file_list_is_capped():
    body = format_authors_body({f"f{i:03}.md": {"human:A"} for i in range(60)})
    assert "- f049.md: A" in body
    assert "- f050.md" not in body
    assert "- ... and 10 more files" in body


def pending(actors, path_doc=DOC_ID):
    return PendingFile(path_doc, RELAY_ID, {}, set(actors), "", None)


class TestTracker:
    def setup_method(self):
        self.t = AuthorTracker()
        self.relay = RelayDoc("start\n")

    def observe_and_confirm(self, is_new_file=False):
        self.t.observe(RELAY_ID, DOC_ID, self.relay.doc)
        return self.t.confirm(
            "r/f", RELAY_ID, DOC_ID, "x.md", content=self.text(), is_new_file=is_new_file
        )

    def text(self):
        return str(self.relay.doc.get("contents", type=Text))

    def commit(self):
        taken = self.t.take_for_commit("r/f", ["x.md"])
        self.t.committed(taken.values())
        return taken

    def test_first_sight_of_existing_file_claims_nobody(self):
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        assert self.observe_and_confirm() == set()
        assert self.commit()["x.md"].owners is None

    def test_new_file_claims_all_registered_writers(self):
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.relay.edit(SERVER + 1, None, "b")
        assert self.observe_and_confirm(is_new_file=True) == {"human:Luc Brinkman"}

    def test_only_clients_that_wrote_since_the_last_commit(self):
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.observe_and_confirm()
        self.commit()
        self.relay.edit(AI, "ai:opus-5.5:james", "b")
        assert self.observe_and_confirm() == {"ai:opus-5.5:james"}
        self.commit()
        self.relay.edit(LUC, "human:Luc Brinkman", "c")
        assert self.observe_and_confirm() == {"human:Luc Brinkman"}

    def test_exports_between_commits_accumulate(self):
        self.observe_and_confirm()
        self.commit()
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.observe_and_confirm()
        self.relay.edit(AI, "ai:opus-5.5:james", "b")
        assert self.observe_and_confirm() == {"human:Luc Brinkman", "ai:opus-5.5:james"}

    def test_line_owners(self):
        self.observe_and_confirm()
        self.commit()
        self.relay.edit(LUC, "human:Luc Brinkman", "Luc writes a line\n")
        self.relay.edit(AI, "ai:opus-5.5:james", "AI line\n")
        self.relay.edit(LUC, "human:Luc Brinkman", "x")  # a few chars on AI's... next line
        self.observe_and_confirm()
        owners = self.commit()["x.md"].owners
        assert owners == [None, "human:Luc Brinkman", "ai:opus-5.5:james", "human:Luc Brinkman"]

    def test_failed_export_keeps_authors_for_retry(self):
        self.observe_and_confirm()
        self.commit()
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.t.observe(RELAY_ID, DOC_ID, self.relay.doc)
        self.t.forget(RELAY_ID, DOC_ID)
        assert self.observe_and_confirm() == {"human:Luc Brinkman"}

    def test_failed_commit_keeps_authors(self):
        self.observe_and_confirm()
        self.commit()
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.observe_and_confirm()
        taken = self.t.take_for_commit("r/f", ["x.md"])
        self.t.restore("r/f", taken)  # commit failed
        assert self.commit()["x.md"].actors == {"human:Luc Brinkman"}

    def test_advance_moves_baseline_without_attributing(self):
        self.observe_and_confirm()
        self.commit()
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.t.observe(RELAY_ID, DOC_ID, self.relay.doc)
        self.t.advance(RELAY_ID, DOC_ID)
        assert self.observe_and_confirm() == set()

    def test_advance_keeps_uncommitted_authors(self):
        self.observe_and_confirm()
        self.commit()
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.observe_and_confirm()
        self.t.observe(RELAY_ID, DOC_ID, self.relay.doc)
        self.t.advance(RELAY_ID, DOC_ID)  # same content again, still uncommitted
        assert self.commit()["x.md"].actors == {"human:Luc Brinkman"}

    def test_take_for_commit_skips_rewritten_and_drops_stale(self):
        self.t.restore(
            "r/f",
            {"staged.md": pending("a"), "later.md": pending("b"), "stale.md": pending("c")},
        )
        taken = self.t.take_for_commit("r/f", ["staged.md", "later.md"], ["later.md"])
        assert set(taken) == {"staged.md"}
        assert set(self.t.take_for_commit("r/f", ["later.md", "stale.md"])) == {"later.md"}

    def test_rename_carries_pending_authors(self):
        self.t.restore("r/f", {"old.md": pending(["human:A"])})
        self.t.rename("r/f", "old.md", "new.md")
        assert set(self.t.take_for_commit("r/f", ["old.md", "new.md"])) == {"new.md"}

    def test_saves_are_throttled_and_skipped_when_clean(self, tmp_path):
        state = str(tmp_path)
        path = os.path.join(state, "document_state_vectors.json")
        self.t.load(RELAY_ID, state)
        self.t.save(RELAY_ID, state)
        assert not os.path.exists(path)  # nothing changed yet
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.observe_and_confirm()
        self.commit()
        self.t.save(RELAY_ID, state)
        first = open(path).read()
        self.relay.edit(AI, "ai:opus-5.5:james", "b")
        self.observe_and_confirm()
        self.commit()
        self.t.save(RELAY_ID, state)
        assert open(path).read() == first  # within the interval
        self.t.save(RELAY_ID, state, force=True)
        assert str(AI) in open(path).read()
        # Only registered clients are stored.
        assert str(SERVER) not in open(path).read()

    def test_baselines_survive_restart_and_reload(self, tmp_path):
        state = str(tmp_path)
        self.t.load(RELAY_ID, state)
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.observe_and_confirm()
        self.commit()
        self.t.save(RELAY_ID, state)

        restarted = AuthorTracker()
        restarted.load(RELAY_ID, state)
        self.relay.edit(AI, "ai:opus-5.5:james", "b")
        restarted.observe(RELAY_ID, DOC_ID, self.relay.doc)
        assert restarted.confirm("r/f", RELAY_ID, DOC_ID, "x.md") == {"ai:opus-5.5:james"}


# --- end to end -----------------------------------------------------------


class TestCommitMessages:
    """Webhook -> fetch -> export -> commit, with real pycrdt docs and git."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pm = PersistenceManager(self.temp_dir)
        self.relay_client = RelayClient("http://relay.test")
        self.relay_client.dm = MagicMock()
        self.docs = {
            DOC_ID: RelayDoc("# Intro\nOld line one\nOld line two\n"),
            DOC2_ID: RelayDoc("# Risks\n"),
            DOC3_ID: RelayDoc(),
        }
        compound = {
            S3RN.get_compound_document_id(S3RemoteDocument(RELAY_ID, FOLDER_ID, d)): d
            for d in self.docs
        }
        self.relay_client.dm.get_doc_as_update.side_effect = lambda cid: self.docs[
            compound[cid]
        ].update()
        self.engine = SyncEngine(self.temp_dir, self.relay_client, self.pm)

        self.pm.load_persistent_data(RELAY_ID)
        self.pm.filemeta_folders[RELAY_ID][FOLDER_ID] = {
            DOC_PATH: {"id": DOC_ID, "type": "markdown"},
            DOC2_PATH: {"id": DOC2_ID, "type": "markdown"},
            DOC3_PATH: {"id": DOC3_ID, "type": "markdown"},
        }
        self.pm._build_resource_index(RELAY_ID)
        self.repo = self.pm.init_git_repo(RELAY_ID, FOLDER_ID)
        with self.repo.config_writer() as cw:
            cw.set_value("user", "name", "Relay Git Sync")
            cw.set_value("user", "email", "relay-git-sync@lensacademy.org")

        # Initial export establishes each doc's baseline.
        self.change(DOC_ID)
        self.change(DOC2_ID)
        assert self.pm.commit_changes()
        self.start = self.repo.head.commit.hexsha

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def change(self, doc_id):
        result = self.engine.process_document_change(RELAY_ID, doc_id, datetime.now(timezone.utc))
        assert result.success, result.error

    def new_commits(self):
        """(author, subject) of every commit since setup, oldest first."""
        out = self.repo.git.log("--reverse", "--format=%an|%cn|%s", f"{self.start}..HEAD")
        return [tuple(line.split("|")) for line in out.splitlines()]

    def blame(self, path):
        """(author, line) for every line of ``path`` at HEAD."""
        out = self.repo.git.blame("--line-porcelain", "HEAD", "--", path.lstrip("/"))
        result, author = [], None
        for line in out.splitlines():
            if line.startswith("author "):
                author = line[len("author ") :]
            elif line.startswith("\t"):
                result.append((author, line[1:]))
        return result

    def assert_head_matches_relay(self):
        assert not self.repo.is_dirty(untracked_files=True)
        for doc_id, path in ((DOC_ID, DOC_PATH), (DOC2_ID, DOC2_PATH)):
            blob = self.repo.head.commit.tree / path.lstrip("/")
            assert blob.data_stream.read().decode("utf-8") == self.docs[doc_id].text()

    def test_initial_export_names_nobody(self):
        message = self.repo.head.commit.message
        assert message.startswith("Auto-sync: ") and "Authors:" not in message

    def test_each_line_blames_its_author(self):
        def luc(t):
            t.insert(len("# Intro\n"), "Luc rewrote the intro.\n")

        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", luc)
        self.change(DOC_ID)
        self.docs[DOC_ID].edit(AI, "ai:opus-5.5:james", "AI added a summary.\n")
        self.change(DOC_ID)
        self.docs[DOC2_ID].edit(OBSIDIAN, "idheqwn0f6k0xxt", "From Obsidian.\n")
        self.change(DOC2_ID)

        assert self.pm.commit_changes()
        print(self.repo.git.log(f"{self.start}..HEAD", "--format=%an <%ae>%n%B"))
        print(self.repo.git.blame("HEAD", "--", DOC_PATH.lstrip("/")))
        # One commit per author, committed by the bot; no leftover bot commit.
        assert [(a, c) for a, c, _ in self.new_commits()] == [
            ("ai:opus-5.5:james", "Relay Git Sync"),
            ("Luc Brinkman", "Relay Git Sync"),
            ("relay-user:idheqwn0f6k0xxt", "Relay Git Sync"),
        ]
        assert self.blame(DOC_PATH) == [
            ("Relay Git Sync", "# Intro"),
            ("Luc Brinkman", "Luc rewrote the intro."),
            ("Relay Git Sync", "Old line one"),
            ("Relay Git Sync", "Old line two"),
            ("ai:opus-5.5:james", "AI added a summary."),
        ]
        assert self.blame(DOC2_PATH)[-1] == ("relay-user:idheqwn0f6k0xxt", "From Obsidian.")
        message = self.repo.git.log("-1", "--format=%B", self.new_commits_sha(1))
        assert "Lines written since the last sync:\n- Lens Edu/modules/Intro.md: L2" in message
        self.assert_head_matches_relay()

    def new_commits_sha(self, n):
        return self.repo.git.log("--reverse", "--format=%H", f"{self.start}..HEAD").split()[n]

    def test_concurrent_edits_in_one_hunk(self):
        # Luc and the AI edit the same region at the same time, without
        # seeing each other's change; Luc also deletes an old line.
        doc = self.docs[DOC_ID]
        luc = doc.fork(LUC, "human:Luc Brinkman")
        ai = doc.fork(AI, "ai:opus-5.5:james")
        lt = luc.get("contents", type=Text)
        at = ai.get("contents", type=Text)
        start = len("# Intro\n")
        del lt[start : start + len("Old line one\n")]
        lt.insert(start, "Luc A\nLuc B\n")
        at.insert(start + len("Old line one\n"), "AI middle\n")
        doc.merge(luc)
        doc.merge(ai)
        self.change(DOC_ID)

        assert self.pm.commit_changes()
        blame = dict((line, author) for author, line in self.blame(DOC_PATH))
        assert blame["Luc A"] == "Luc Brinkman" and blame["Luc B"] == "Luc Brinkman"
        assert blame["AI middle"] == "ai:opus-5.5:james"
        assert blame["Old line two"] == "Relay Git Sync"
        assert "Old line one" not in blame
        self.assert_head_matches_relay()

    def test_line_shared_by_two_authors_goes_to_the_main_writer(self):
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Luc wrote most of this")
        self.docs[DOC_ID].edit(AI, "ai:opus-5.5:james", ", AI.\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert self.blame(DOC_PATH)[-1] == ("Luc Brinkman", "Luc wrote most of this, AI.")

    def test_unattributed_lines_arrive_in_a_closing_bot_commit(self):
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Luc line\n")
        self.docs[DOC_ID].edit(SERVER + 7, None, "server line\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        commits = self.new_commits()
        assert [a for a, _, _ in commits] == ["Luc Brinkman", "Relay Git Sync"]
        body = self.repo.head.commit.message
        assert "Authors: Luc Brinkman" in body
        assert self.blame(DOC_PATH)[-2:] == [
            ("Luc Brinkman", "Luc line"),
            ("Relay Git Sync", "server line"),
        ]
        self.assert_head_matches_relay()

    def test_server_only_writes_keep_the_plain_commit(self):
        self.docs[DOC_ID].edit(SERVER + 7, None, "link index\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert [a for a, _, _ in self.new_commits()] == ["Relay Git Sync"]
        assert "Authors:" not in self.repo.head.commit.message

    def test_unreadable_doc_falls_back_to_per_file_authors(self, monkeypatch):
        monkeypatch.setattr("authorship.character_ids", lambda update: None)
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "x\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert [a for a, _, _ in self.new_commits()] == ["Relay Git Sync"]
        assert "- Lens Edu/modules/Intro.md: Luc Brinkman" in self.repo.head.commit.message

    def test_git_failure_in_the_chain_falls_back_to_one_commit(self, monkeypatch):
        monkeypatch.setattr(
            "persistence._partial_text", MagicMock(side_effect=RuntimeError("boom"))
        )
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "x\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert [a for a, _, _ in self.new_commits()] == ["Relay Git Sync"]
        assert "Authors: Luc Brinkman" in self.repo.head.commit.message
        self.assert_head_matches_relay()

    def test_attribution_failure_never_blocks_the_export(self):
        self.pm.authorship.observe = MagicMock(side_effect=RuntimeError("boom"))
        self.relay_client.doc_observer = self.pm.authorship.observe
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "x\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert "Authors:" not in self.repo.head.commit.message
        self.assert_head_matches_relay()

    def test_deleted_pending_file_does_not_cost_the_others_their_blame(self):
        self.docs[DOC2_ID].edit(OBSIDIAN, "idheqwn0f6k0xxt", "x\n")
        self.change(DOC2_ID)
        os.remove(os.path.join(self.pm.get_folder_path(RELAY_ID, FOLDER_ID), DOC2_PATH.lstrip("/")))
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Luc line\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert self.blame(DOC_PATH)[-1] == ("Luc Brinkman", "Luc line")

    def test_name_git_would_reject_still_commits(self):
        self.docs[DOC_ID].edit(LUC, 'human:"""', "quoted\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert self.blame(DOC_PATH)[-1] == ("unknown", "quoted")

    def test_too_many_authors_fall_back_to_one_commit(self):
        for n in range(21):
            self.docs[DOC_ID].edit(5000 + n, f"human:Person {n:02}", f"line {n}\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert [a for a, _, _ in self.new_commits()] == ["Relay Git Sync"]
        assert "and 1 more" in self.repo.head.commit.message

    def test_file_path_cannot_forge_message_lines(self):
        evil = "/notes\nCo-authored-by: Mallory <m@evil.test>\n- x.md"
        self.pm.filemeta_folders[RELAY_ID][FOLDER_ID][evil] = {"id": DOC3_ID, "type": "markdown"}
        del self.pm.filemeta_folders[RELAY_ID][FOLDER_ID][DOC3_PATH]
        self.pm._build_resource_index(RELAY_ID)
        self.docs[DOC3_ID].edit(LUC, "human:Luc Brinkman", "x\n")
        self.change(DOC3_ID)
        assert self.pm.commit_changes()
        for sha in self.repo.git.log("--format=%H", f"{self.start}..HEAD").split():
            message = self.repo.git.log("-1", "--format=%B", sha)
            assert not any(
                line.startswith("Co-authored-by: Mallory") for line in message.splitlines()
            )

    def test_chain_uses_the_same_committer_as_the_plain_commit(self, monkeypatch, tmp_path):
        # No identity anywhere: GitPython's index.commit synthesises one,
        # git commit-tree alone may refuse; the chain must pass it explicitly.
        with self.repo.config_writer() as cw:
            cw.remove_section("user")
        for var in ("GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "GIT_AUTHOR_NAME", "EMAIL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
        monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        expected = git.Actor.committer(self.repo.config_reader()).name
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Luc line\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert [(a, c) for a, c, _ in self.new_commits()] == [("Luc Brinkman", expected)]

    def test_baselines_are_saved_at_commit(self):
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Luc line\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        path = os.path.join(self.pm.get_state_dir(RELAY_ID), "document_state_vectors.json")
        assert str(LUC) in open(path).read()

    def test_first_export_via_webhook_credits_the_creator(self):
        self.docs[DOC3_ID].edit(LUC, "human:Luc Brinkman", "# New\n")
        self.change(DOC3_ID)
        assert self.pm.commit_changes()
        assert self.blame(DOC3_PATH) == [("Luc Brinkman", "# New")]

    def test_rename_before_commit_keeps_the_authors(self):
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Welcome.\n")
        self.change(DOC_ID)
        resource = S3RemoteDocument(RELAY_ID, FOLDER_ID, DOC_ID)
        self.pm.move_file(resource, DOC_PATH, "/Lens Edu/modules/Introduction.md")
        assert self.pm.commit_changes()
        assert "- Lens Edu/modules/Introduction.md: Luc Brinkman" in self.repo.head.commit.message
        # Moved and edited in one tick: no line split (it would re-blame the
        # whole file), the single commit names Luc for the file instead.
        assert [a for a, _, _ in self.new_commits()] == ["Relay Git Sync"]

    def test_document_sync_request_path_is_attributed(self):
        self.docs[DOC_ID].edit(AI, "ai:opus-5.5:james", "AI line.\n")
        resource = S3RemoteDocument(RELAY_ID, FOLDER_ID, DOC_ID)
        result = self.engine.process_sync_request(
            SyncRequest(resource=resource, timestamp=datetime.now(timezone.utc))
        )
        assert result.success, result.error
        assert self.pm.commit_changes()
        assert self.blame(DOC_PATH)[-1] == ("ai:opus-5.5:james", "AI line.")

        # The baseline moved: a later webhook export does not re-credit the AI.
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Luc line.\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert self.blame(DOC_PATH)[-2:] == [
            ("ai:opus-5.5:james", "AI line."),
            ("Luc Brinkman", "Luc line."),
        ]


# --- reconstruction fuzz --------------------------------------------------


def test_character_ids_match_pycrdt_under_concurrent_editing():
    """Three clients insert and delete at random positions, syncing only now
    and then, so many edits are concurrent. The rebuilt text must equal
    pycrdt's and every character must belong to the client that typed it
    (each client types its own alphabet). ASCII only: pycrdt 0.9 indexes
    Y.Text by UTF-8 bytes; non-ASCII was fuzzed separately with pycrdt 0.14."""
    import random

    from yjs_attribution import character_ids

    alphabet = {11: "abcdefgh \n", 22: "ABCDEFGH \n", 33: "0123456789\n"}
    for seed in range(150):
        rng = random.Random(seed)
        docs = {c: Doc(client_id=c) for c in alphabet}
        for d in docs.values():
            d["contents"] = Text()
        for _ in range(rng.randint(5, 120)):
            c = rng.choice(list(alphabet))
            t = docs[c].get("contents", type=Text)
            length = len(str(t))
            if length and rng.random() < 0.3:
                i = rng.randrange(length)
                del t[i : i + rng.randint(1, min(5, length - i))]
            else:
                chunk = "".join(rng.choice(alphabet[c]) for _ in range(rng.randint(1, 6)))
                t.insert(rng.randint(0, length), chunk)
            if rng.random() < 0.15:
                a, b = rng.sample(list(alphabet), 2)
                docs[b].apply_update(docs[a].get_update(docs[b].get_state()))
        for a in docs:
            for b in docs:
                if a != b:
                    docs[b].apply_update(docs[a].get_update(docs[b].get_state()))
        doc = docs[11]
        result = character_ids(doc.get_update())
        assert result is not None and result[0] == str(doc.get("contents", type=Text)), seed
        for ch, (client, _) in zip(*result):
            assert ch in alphabet[client], (seed, ch, client)
