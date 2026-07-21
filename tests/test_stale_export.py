#!/usr/bin/env python3
"""Regression test: a transient write failure must not permanently poison the
document hash cache.

Scenario (matches the 2026-07-19 Lens Academy incident): a document changes,
the webhook arrives, the content fetch succeeds, but writing the file to the
git repo fails once (disk hiccup, git lock, etc.). If the new content hash is
recorded anyway, every subsequent webhook for the same content compares equal
hashes and skips the write - the export stays stale until the document is
edited *again*.
"""

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock

from persistence import PersistenceManager
from sync_engine import SyncEngine

RELAY_ID = "11111111-1111-4111-8111-111111111111"
FOLDER_ID = "22222222-2222-4222-8222-222222222222"
DOC_ID = "33333333-3333-4333-8333-333333333333"
DOC_PATH = "/Lenses/test-lens.md"
CONTENT = "---\ntitle: Test\n---\nclean accepted content\n"


class TestTransientWriteFailure:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pm = PersistenceManager(self.temp_dir)
        self.relay_client = MagicMock()
        self.relay_client.fetch_document_content.return_value = CONTENT
        self.engine = SyncEngine(self.temp_dir, self.relay_client, self.pm)

        # Seed state: one known folder containing one document.
        self.pm.load_persistent_data(RELAY_ID)
        self.pm.filemeta_folders[RELAY_ID][FOLDER_ID] = {
            DOC_PATH: {"id": DOC_ID, "type": "document"}
        }
        self.pm._build_resource_index(RELAY_ID)
        self.pm.init_git_repo(RELAY_ID, FOLDER_ID)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def _exported_file(self):
        folder_path = self.pm.get_folder_path_with_prefix(RELAY_ID, FOLDER_ID)
        return os.path.join(folder_path, DOC_PATH.lstrip("/"))

    def _process(self):
        return self.engine.process_document_change(
            RELAY_ID, DOC_ID, datetime.now(timezone.utc)
        )

    def test_retry_after_transient_write_failure(self):
        """A webhook retry after a failed write must write the file."""
        real_write = self.pm.write_file_content
        calls = {"n": 0}

        def flaky_write(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise IOError("simulated transient disk/git failure")
            return real_write(*args, **kwargs)

        self.pm.write_file_content = flaky_write

        # First webhook: fetch succeeds, write fails.
        result = self._process()
        assert result.success  # engine reports overall success; op carries error
        assert not os.path.exists(self._exported_file())

        # Hash cache must NOT claim this content was exported.
        assert self.pm.document_hashes[RELAY_ID].get(DOC_ID) != hashlib.sha256(
            CONTENT.encode("utf-8")
        ).hexdigest(), "content hash recorded although the write failed"

        # Second webhook for the SAME content (e.g. debounced duplicate or the
        # relay re-poking): must retry the write and succeed.
        self._process()
        assert os.path.exists(self._exported_file()), (
            "stale export: retry webhook skipped the write because the hash "
            "cache was poisoned by the failed attempt"
        )
        with open(self._exported_file()) as f:
            assert f.read() == CONTENT

    def test_successful_write_records_hash(self):
        """Happy path: hash is recorded and duplicate webhooks skip rewrites."""
        self._process()
        assert os.path.exists(self._exported_file())
        expected_hash = hashlib.sha256(CONTENT.encode("utf-8")).hexdigest()
        assert self.pm.document_hashes[RELAY_ID].get(DOC_ID) == expected_hash

        # Duplicate webhook: no second write needed.
        self.pm.write_file_content = MagicMock(
            side_effect=AssertionError("must not rewrite unchanged content")
        )
        result = self._process()
        assert result.success
        assert result.operations == []
