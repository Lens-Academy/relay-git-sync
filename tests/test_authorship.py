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

    def edit(self, client_id, actor, append):
        """One client appends text; registers its clientID under ``actor``
        (None = unregistered, like the server's own writes)."""
        client = Doc(client_id=client_id)
        client.apply_update(self.doc.get_update())
        contents = client.get("contents", type=Text)
        users = client.get("users", type=Map)
        with client.transaction():
            contents += append
            if actor is not None:
                if actor not in users:
                    users[actor] = Map({"ids": Array(), "ds": Array(), "meta": Map()})
                users[actor]["ids"].append(float(client_id))
        self.doc.apply_update(client.get_update(self.doc.get_state()))

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


def test_file_list_is_capped():
    body = format_authors_body({f"f{i:03}.md": {"human:A"} for i in range(60)})
    assert "- f049.md: A" in body
    assert "- f050.md" not in body
    assert "- ... and 10 more files" in body


class TestTracker:
    def setup_method(self):
        self.t = AuthorTracker()
        self.relay = RelayDoc("start")

    def observe_and_confirm(self, is_new_file=False):
        self.t.observe(RELAY_ID, DOC_ID, self.relay.doc)
        return self.t.confirm("r/f", RELAY_ID, DOC_ID, "x.md", is_new_file=is_new_file)

    def test_first_sight_of_existing_file_claims_nobody(self):
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        assert self.observe_and_confirm() == set()

    def test_new_file_claims_all_registered_writers(self):
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.relay.edit(SERVER + 1, None, "b")
        assert self.observe_and_confirm(is_new_file=True) == {"human:Luc Brinkman"}

    def test_only_clients_that_wrote_since_baseline(self):
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.observe_and_confirm()
        self.relay.edit(AI, "ai:opus-5.5:james", "b")
        assert self.observe_and_confirm() == {"ai:opus-5.5:james"}
        self.relay.edit(LUC, "human:Luc Brinkman", "c")
        assert self.observe_and_confirm() == {"human:Luc Brinkman"}

    def test_failed_export_keeps_authors_for_retry(self):
        self.observe_and_confirm()
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.t.observe(RELAY_ID, DOC_ID, self.relay.doc)
        self.t.forget(RELAY_ID, DOC_ID)
        assert self.observe_and_confirm() == {"human:Luc Brinkman"}

    def test_advance_moves_baseline_without_attributing(self):
        self.observe_and_confirm()
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.t.observe(RELAY_ID, DOC_ID, self.relay.doc)
        self.t.advance(RELAY_ID, DOC_ID)
        assert self.observe_and_confirm() == set()

    def test_take_for_commit_keeps_later_writes_drops_stale(self):
        self.t.restore("r/f", {"staged.md": {"a"}, "later.md": {"b"}, "stale.md": {"c"}})
        taken = self.t.take_for_commit("r/f", ["staged.md"], ["later.md"])
        assert taken == {"staged.md": {"a"}}
        assert self.t.take_for_commit("r/f", ["later.md", "stale.md"]) == {"later.md": {"b"}}

    def test_rename_carries_pending_authors(self):
        self.t.restore("r/f", {"old.md": {"human:A"}})
        self.t.rename("r/f", "old.md", "new.md")
        assert self.t.take_for_commit("r/f", ["old.md", "new.md"]) == {"new.md": {"human:A"}}

    def test_saves_are_throttled_and_skipped_when_clean(self, tmp_path):
        state = str(tmp_path)
        path = os.path.join(state, "document_state_vectors.json")
        self.t.load(RELAY_ID, state)
        self.t.save(RELAY_ID, state)
        assert not os.path.exists(path)  # nothing changed yet
        self.relay.edit(LUC, "human:Luc Brinkman", "a")
        self.observe_and_confirm()
        self.t.save(RELAY_ID, state)
        first = open(path).read()
        self.relay.edit(AI, "ai:opus-5.5:james", "b")
        self.observe_and_confirm()
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
        self.t.save(RELAY_ID, state)

        restarted = AuthorTracker()
        restarted.load(RELAY_ID, state)
        self.relay.edit(AI, "ai:opus-5.5:james", "b")
        restarted.observe(RELAY_ID, DOC_ID, self.relay.doc)
        assert restarted.confirm("r/f", RELAY_ID, DOC_ID, "x.md") == {"ai:opus-5.5:james"}


# --- end to end -----------------------------------------------------------


class TestCommitMessages:
    """Webhook -> fetch -> export -> commit, with real pycrdt docs."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pm = PersistenceManager(self.temp_dir)
        self.relay_client = RelayClient("http://relay.test")
        self.relay_client.dm = MagicMock()
        self.docs = {
            DOC_ID: RelayDoc("# Intro\n"),
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

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def change(self, doc_id):
        result = self.engine.process_document_change(RELAY_ID, doc_id, datetime.now(timezone.utc))
        assert result.success, result.error

    def last_message(self):
        return self.repo.head.commit.message

    def test_initial_export_names_nobody(self):
        assert self.last_message().startswith("Auto-sync: ")
        assert "Authors:" not in self.last_message()

    def test_batched_commit_names_each_file_and_author(self):
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Welcome.\n")
        self.change(DOC_ID)
        self.docs[DOC_ID].edit(AI, "ai:opus-5.5:james", "AI summary.\n")
        self.change(DOC_ID)
        self.docs[DOC2_ID].edit(OBSIDIAN, "idheqwn0f6k0xxt", "From Obsidian.\n")
        self.change(DOC2_ID)

        assert self.pm.commit_changes()
        message = self.last_message()
        print(message)
        lines = message.splitlines()
        assert lines[0].startswith("Auto-sync: ")
        assert lines[2:] == [
            "Authors: ai:opus-5.5:james, Luc Brinkman, relay-user:idheqwn0f6k0xxt",
            "",
            "- Lens Edu/Lenses/Risks.md: relay-user:idheqwn0f6k0xxt",
            "- Lens Edu/modules/Intro.md: ai:opus-5.5:james, Luc Brinkman",
            "",
            "Co-authored-by: ai:opus-5.5:james <ai-opus-5.5-james@relay.invalid>",
            "Co-authored-by: Luc Brinkman <luc-brinkman@relay.invalid>",
            "Co-authored-by: relay-user:idheqwn0f6k0xxt <relay-user-idheqwn0f6k0xxt@relay.invalid>",
        ]
        # The git identity stays the bot's: the names are self-reported.
        assert self.repo.head.commit.author.name == "Relay Git Sync"

        # Authors are consumed: the next commit only names its own writers.
        self.docs[DOC2_ID].edit(AI, "ai:opus-5.5:james", "More.\n")
        self.change(DOC2_ID)
        assert self.pm.commit_changes()
        assert "Authors: ai:opus-5.5:james\n" in self.last_message()
        assert "Luc Brinkman" not in self.last_message()

    def test_server_only_writes_keep_the_plain_message(self):
        self.docs[DOC_ID].edit(SERVER + 7, None, "link index\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert "Authors:" not in self.last_message()

    def test_attribution_failure_never_blocks_the_export(self):
        self.pm.authorship.observe = MagicMock(side_effect=RuntimeError("boom"))
        self.relay_client.doc_observer = self.pm.authorship.observe
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "x\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert "Authors:" not in self.last_message()
        path = os.path.join(self.pm.get_folder_path(RELAY_ID, FOLDER_ID), DOC_PATH.lstrip("/"))
        with open(path, encoding="utf-8") as f:
            assert f.read().endswith("x\n")

    def test_first_export_via_webhook_credits_the_creator(self):
        # The file was never exported; the doc webhook exports it through the
        # update path, which must still treat it as a new file.
        self.docs[DOC3_ID].edit(LUC, "human:Luc Brinkman", "# New\n")
        self.change(DOC3_ID)
        assert self.pm.commit_changes()
        assert "- Lens Edu/articles/New.md: Luc Brinkman" in self.last_message()

    def test_rename_before_commit_keeps_the_authors(self):
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Welcome.\n")
        self.change(DOC_ID)
        resource = S3RemoteDocument(RELAY_ID, FOLDER_ID, DOC_ID)
        self.pm.move_file(resource, DOC_PATH, "/Lens Edu/modules/Introduction.md")
        assert self.pm.commit_changes()
        assert "- Lens Edu/modules/Introduction.md: Luc Brinkman" in self.last_message()

    def test_document_sync_request_path_is_attributed(self):
        self.docs[DOC_ID].edit(AI, "ai:opus-5.5:james", "AI line.\n")
        resource = S3RemoteDocument(RELAY_ID, FOLDER_ID, DOC_ID)
        result = self.engine.process_sync_request(
            SyncRequest(resource=resource, timestamp=datetime.now(timezone.utc))
        )
        assert result.success, result.error
        assert self.pm.commit_changes()
        assert "Authors: ai:opus-5.5:james\n" in self.last_message()

        # The baseline moved: a later webhook export does not re-credit the AI.
        self.docs[DOC_ID].edit(LUC, "human:Luc Brinkman", "Luc line.\n")
        self.change(DOC_ID)
        assert self.pm.commit_changes()
        assert "Authors: Luc Brinkman\n" in self.last_message()
