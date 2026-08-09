#!/usr/bin/env python3
"""Regression tests for edits lost around the git commit timer.

Two failure modes had the same symptom (a change reaches the working tree but
never reaches GitHub until an unrelated later edit):

1. The commit timer was gated on sync_state.has_changes and reset the flag
   after a slow commit+push returned, wiping the signal a worker set while
   the commit was in flight.
2. Push failures are swallowed, and a clean-but-ahead repo was never pushed
   again because push only ran for dirty repos.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import git

from operations_queue import OperationsQueue
from persistence import PersistenceManager


def make_queue(commit_result=True):
    """Build an OperationsQueue with a mocked sync engine.

    A large commit_interval keeps the background timer thread from firing
    during the test; _maybe_commit_changes is invoked directly.
    """
    sync_engine = MagicMock()
    sync_engine.persistence_manager.commit_changes.return_value = commit_result
    queue = OperationsQueue(sync_engine, commit_interval=3600)
    return queue, sync_engine


class TestCommitTimerNotGatedOnFlag:
    def test_commits_even_when_flag_is_false(self):
        """An edit written during a previous commit+push leaves has_changes
        False (the old gate) but the repo dirty; the timer must still ask
        git instead of trusting the flag."""
        queue, sync_engine = make_queue()
        queue.sync_state.has_changes = False

        queue._maybe_commit_changes()

        sync_engine.persistence_manager.commit_changes.assert_called_once()

    def test_flag_reset_only_after_successful_commit(self):
        queue, sync_engine = make_queue(commit_result=True)
        queue.sync_state.has_changes = True

        queue._maybe_commit_changes()

        assert queue.sync_state.has_changes is False
        assert queue.sync_state.last_git_commit is not None

    def test_flag_survives_when_nothing_committed(self):
        queue, sync_engine = make_queue(commit_result=False)
        queue.sync_state.has_changes = True

        queue._maybe_commit_changes()

        assert queue.sync_state.has_changes is True

    def test_commit_exception_does_not_propagate(self):
        queue, sync_engine = make_queue()
        sync_engine.persistence_manager.commit_changes.side_effect = RuntimeError("boom")

        queue._maybe_commit_changes()  # must not raise


class TestPushRetry:
    """Tests against real git repos in a temp dir."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = PersistenceManager(self.temp_dir)

        # A bare repo standing in for GitHub, and a work repo tracking it.
        self.remote_path = os.path.join(self.temp_dir, "remote.git")
        git.Repo.init(self.remote_path, bare=True, initial_branch="main")

        self.work_path = os.path.join(self.temp_dir, "work")
        self.repo = git.Repo.init(self.work_path, initial_branch="main")
        self.repo.create_remote("origin", self.remote_path)
        self._set_git_identity(self.repo)

        self._commit_file("seed.md", "seed")
        self.repo.remotes.origin.push("main", set_upstream=True)

        self.persistence.git_repos["relay/folder"] = self.repo

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _set_git_identity(repo):
        # `git pull --rebase` (subprocess) needs a committer identity, which
        # the temp-dir test repos lack; the production container has one.
        with repo.config_writer() as cw:
            cw.set_value("user", "name", "Test Sync")
            cw.set_value("user", "email", "test@example.com")

    def _commit_file(self, name, content):
        path = os.path.join(self.work_path, name)
        with open(path, "w") as f:
            f.write(content)
        self.repo.git.add(A=True)
        self.repo.index.commit(f"add {name}")

    def _remote_head(self):
        return git.Repo(self.remote_path).commit("main").hexsha

    def test_has_unpushed_commits_detection(self):
        assert not self.persistence._has_unpushed_commits("relay/folder", self.repo)

        self._commit_file("new.md", "content")
        assert self.persistence._has_unpushed_commits("relay/folder", self.repo)

        self.repo.remotes.origin.push()
        assert not self.persistence._has_unpushed_commits("relay/folder", self.repo)

    def test_has_unpushed_commits_no_remote(self):
        solo = git.Repo.init(os.path.join(self.temp_dir, "solo"), initial_branch="main")
        assert not self.persistence._has_unpushed_commits("relay/solo", solo)

    def test_clean_but_ahead_repo_is_pushed(self):
        """A commit stranded by an earlier failed push must be pushed on the
        next timer tick even though the working tree is clean."""
        self._commit_file("stranded.md", "content")
        local_head = self.repo.head.commit.hexsha
        assert self._remote_head() != local_head
        assert not self.repo.is_dirty()

        self.persistence.commit_changes()

        assert self._remote_head() == local_head

    def test_dirty_repo_still_committed_and_pushed(self):
        with open(os.path.join(self.work_path, "dirty.md"), "w") as f:
            f.write("uncommitted")

        committed = self.persistence.commit_changes()

        assert committed
        assert not self.repo.is_dirty()
        assert self._remote_head() == self.repo.head.commit.hexsha

    def test_diverged_remote_recovers_via_pull_then_push(self):
        """GitPython's push() reports a non-fast-forward rejection via flags
        instead of raising; the rejection must still trigger the
        pull-then-retry path so a diverged remote converges."""
        # Someone else pushes to the remote...
        other_path = os.path.join(self.temp_dir, "other")
        other = git.Repo.clone_from(self.remote_path, other_path)
        self._set_git_identity(other)
        with open(os.path.join(other_path, "other.md"), "w") as f:
            f.write("remote change")
        other.git.add(A=True)
        other.index.commit("remote change")
        other.remotes.origin.push()

        # ...while we have a local commit the remote lacks.
        self._commit_file("local.md", "local change")
        assert not self.repo.is_dirty()

        self.persistence.commit_changes()

        remote_repo = git.Repo(self.remote_path)
        assert self._remote_head() == self.repo.head.commit.hexsha
        remote_files = remote_repo.git.ls_tree("--name-only", "main").splitlines()
        assert "other.md" in remote_files
        assert "local.md" in remote_files

    def test_failed_push_retry_backs_off(self):
        """A push retry that does not converge must set a backoff instead of
        re-pushing on every 5s timer tick."""
        self.repo.remotes.origin.set_url(os.path.join(self.temp_dir, "gone.git"))
        self._commit_file("stranded.md", "content")

        push_calls = []
        original = self.persistence._push_to_remote
        self.persistence._push_to_remote = lambda *a, **kw: (
            push_calls.append(a),
            original(*a, **kw),
        )

        self.persistence.commit_changes()
        assert len(push_calls) == 1
        assert "relay/folder" in self.persistence._push_retry_backoff

        # Immediately after, the backoff window is open: no second attempt.
        self.persistence.commit_changes()
        assert len(push_calls) == 1

    def test_one_repo_failure_does_not_block_others(self):
        """A broken repo earlier in the iteration must not stop later repos
        from being committed."""
        broken = MagicMock()
        broken.is_dirty.side_effect = RuntimeError("corrupt repo")
        # dicts preserve insertion order: broken first, healthy repo second
        self.persistence.git_repos.clear()
        self.persistence.git_repos["relay/broken"] = broken
        self.persistence.git_repos["relay/folder"] = self.repo

        with open(os.path.join(self.work_path, "after-broken.md"), "w") as f:
            f.write("content")

        committed = self.persistence.commit_changes()

        assert committed
        assert not self.repo.is_dirty()
        assert self._remote_head() == self.repo.head.commit.hexsha
