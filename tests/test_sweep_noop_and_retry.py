#!/usr/bin/env python3
"""Regression tests for two related failure modes (2026-08 Lens Academy):

1. Folder sweeps re-fetched every markdown/canvas doc. Those filemeta entries
   never carry a hash (only blob uploads write one), and should_update_file
   treated "no hash" as "update needed" - so any folder-doc webhook triggered
   thousands of sequential HTTP fetches (~4 min on prod) that blocked the
   single worker thread and delayed every queued per-doc change by minutes.
   Fix: fall back to the hash of our own last successful export and skip the
   fetch when the local file still matches it; record that hash after sweep
   writes too so docs first seen via a sweep converge to the cheap path.

2. Failed per-doc syncs were silently dropped. fetch_document_content returned
   None for transient errors (502/timeout) and legitimately-empty docs alike,
   and the per-doc path reported success - a relay hiccup stranded that doc's
   change until the doc was edited again. Fix: the per-doc path raises
   RelayFetchError on fetch failure, and the queue retries with capped backoff.

3. Fix (1) removed the sweep's role as catch-all repair: a change lost to a
   dropped webhook, an exhausted retry budget, or a restart that wiped the
   in-memory retry table left the local file matching our own last export, so
   every later sweep NOOPed and the doc stayed stale forever. Fix: a periodic
   reconcile timer enqueues forced sweeps (SyncRequest.force) that bypass the
   export-hash fallback and re-fetch every hashless doc.
"""

import hashlib
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from models import SyncOperation, OperationType, SyncResult
from operations_queue import OperationsQueue
from persistence import PersistenceManager
from relay_client import RelayClient, RelayFetchError
from s3rn import S3RemoteCanvas, S3RemoteDocument, S3RemoteFolder
from sync_engine import SyncEngine

RELAY_ID = "11111111-1111-4111-8111-111111111111"
FOLDER_ID = "22222222-2222-4222-8222-222222222222"
DOC_ID = "33333333-3333-4333-8333-333333333333"
DOC_PATH = "/Lenses/test-lens.md"
CONTENT = "---\ntitle: Test\n---\nsynced content\n"
CONTENT_HASH = hashlib.sha256(CONTENT.encode("utf-8")).hexdigest()


class EngineHarness:
    """Real PersistenceManager + SyncEngine over a tempdir, mocked relay client."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pm = PersistenceManager(self.temp_dir)
        self.relay_client = MagicMock()
        self.relay_client.fetch_document_content.return_value = CONTENT
        self.engine = SyncEngine(self.temp_dir, self.relay_client, self.pm)

        self.pm.load_persistent_data(RELAY_ID)
        self.filemeta = {DOC_PATH: {"id": DOC_ID, "type": "markdown"}}
        self.pm.filemeta_folders[RELAY_ID][FOLDER_ID] = self.filemeta
        self.pm._build_resource_index(RELAY_ID)
        self.pm.init_git_repo(RELAY_ID, FOLDER_ID)
        self.folder = S3RemoteFolder(RELAY_ID, FOLDER_ID)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def exported_file(self):
        folder_path = self.pm.get_folder_path_with_prefix(RELAY_ID, FOLDER_ID)
        return os.path.join(folder_path, DOC_PATH.lstrip("/"))

    def write_local(self, content=CONTENT):
        path = self.exported_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def sweep(self):
        return self.engine.apply_remote_folder_changes(
            RELAY_ID, self.folder, {}, self.filemeta
        )


class TestSweepSkipsUnchangedDocs(EngineHarness):
    def test_matching_export_hash_skips_fetch(self):
        """Local file matches our last export -> NOOP, no network fetch."""
        self.write_local()
        self.pm.document_hashes[RELAY_ID][DOC_ID] = CONTENT_HASH

        operations = self.sweep()

        doc_ops = [op for op in operations if op.path == DOC_PATH]
        assert doc_ops and doc_ops[0].type == OperationType.NOOP
        self.relay_client.fetch_document_content.assert_not_called()

    def test_missing_export_hash_still_fetches(self):
        """No recorded export hash -> fetch once (legacy behavior)."""
        self.write_local("stale local content\n")

        operations = self.sweep()

        doc_ops = [op for op in operations if op.path == DOC_PATH]
        assert doc_ops and doc_ops[0].type == OperationType.UPDATE
        self.relay_client.fetch_document_content.assert_called()
        with open(self.exported_file(), encoding="utf-8") as f:
            assert f.read() == CONTENT

    def test_differing_export_hash_fetches(self):
        """Local file diverged from the last export -> fetch and rewrite."""
        self.write_local("out-of-band local edit\n")
        self.pm.document_hashes[RELAY_ID][DOC_ID] = CONTENT_HASH

        operations = self.sweep()

        doc_ops = [op for op in operations if op.path == DOC_PATH]
        assert doc_ops and doc_ops[0].type == OperationType.UPDATE
        with open(self.exported_file(), encoding="utf-8") as f:
            assert f.read() == CONTENT

    def test_filemeta_hash_still_wins_for_blobs(self):
        """A filemeta hash (blob entries) is authoritative; the export-hash
        fallback must not be consulted."""
        self.write_local()
        local_hash = CONTENT_HASH
        # Export hash would say "changed"; filemeta hash says "unchanged".
        self.pm.document_hashes[RELAY_ID][DOC_ID] = "0" * 64
        meta = {"id": DOC_ID, "type": "file", "hash": local_hash}
        assert (
            self.engine.should_update_file(RELAY_ID, DOC_ID, meta, self.exported_file())
            is False
        )
        # And mismatching filemeta hash says "changed" even if export hash matches.
        self.pm.document_hashes[RELAY_ID][DOC_ID] = local_hash
        meta = {"id": DOC_ID, "type": "file", "hash": "f" * 64}
        assert (
            self.engine.should_update_file(RELAY_ID, DOC_ID, meta, self.exported_file())
            is True
        )


class TestSweepRecordsExportHash(EngineHarness):
    def test_sweep_create_records_hash(self):
        """A doc first exported via sweep gets its hash recorded, so the next
        sweep NOOPs instead of fetching again."""
        operations = self.sweep()

        creates = [op for op in operations if op.path == DOC_PATH]
        assert creates and creates[0].type == OperationType.CREATE
        assert self.pm.document_hashes[RELAY_ID][DOC_ID] == CONTENT_HASH

        self.relay_client.fetch_document_content.reset_mock()
        operations = self.sweep()
        doc_ops = [op for op in operations if op.path == DOC_PATH]
        assert doc_ops and doc_ops[0].type == OperationType.NOOP
        self.relay_client.fetch_document_content.assert_not_called()

    def test_sweep_update_records_hash(self):
        self.write_local("stale local content\n")

        self.sweep()

        assert self.pm.document_hashes[RELAY_ID][DOC_ID] == CONTENT_HASH

    def test_failed_write_records_no_hash_and_next_sweep_refetches(self):
        self.write_local("stale local content\n")

        def failing_write(*args, **kwargs):
            raise IOError("simulated disk failure")

        real_write = self.pm.write_file_content
        self.pm.write_file_content = failing_write
        operations = self.sweep()
        doc_ops = [op for op in operations if op.path == DOC_PATH]
        assert doc_ops and doc_ops[0].error
        assert DOC_ID not in self.pm.document_hashes[RELAY_ID]

        self.pm.write_file_content = real_write
        self.relay_client.fetch_document_content.reset_mock()
        self.sweep()
        self.relay_client.fetch_document_content.assert_called()
        with open(self.exported_file(), encoding="utf-8") as f:
            assert f.read() == CONTENT

    def test_canvas_update_records_hash(self):
        canvas = S3RemoteCanvas(RELAY_ID, FOLDER_ID, DOC_ID)
        canvas_json = '{"edges": [], "nodes": []}'
        operation = SyncOperation(
            type=OperationType.UPDATE,
            path="/board.canvas",
            folder_resource=self.folder,
            document_resource=canvas,
            content=canvas_json,
            metadata={"id": DOC_ID, "type": "canvas"},
        )

        self.engine.handle_server_update(RELAY_ID, operation)

        assert (
            self.pm.document_hashes[RELAY_ID][DOC_ID]
            == hashlib.sha256(canvas_json.encode("utf-8")).hexdigest()
        )


class TestForcedReconcileSweep(EngineHarness):
    def sweep_forced(self):
        return self.engine.apply_remote_folder_changes(
            RELAY_ID, self.folder, {}, self.filemeta, force=True
        )

    def test_force_bypasses_export_hash_fallback(self):
        """Local file matches our own last export (ordinary sweep NOOPs), but
        the remote changed under a lost webhook - a forced sweep re-fetches
        and repairs the stale export."""
        stale = "content from before the lost webhook\n"
        self.write_local(stale)
        self.pm.document_hashes[RELAY_ID][DOC_ID] = hashlib.sha256(
            stale.encode("utf-8")
        ).hexdigest()

        operations = self.sweep()
        doc_ops = [op for op in operations if op.path == DOC_PATH]
        assert doc_ops and doc_ops[0].type == OperationType.NOOP
        self.relay_client.fetch_document_content.assert_not_called()

        operations = self.sweep_forced()
        doc_ops = [op for op in operations if op.path == DOC_PATH]
        assert doc_ops and doc_ops[0].type == OperationType.UPDATE
        with open(self.exported_file(), encoding="utf-8") as f:
            assert f.read() == CONTENT
        assert self.pm.document_hashes[RELAY_ID][DOC_ID] == CONTENT_HASH

    def test_force_respects_authoritative_filemeta_hash(self):
        """Blob entries carry a server-side hash the sweep fetched fresh; force
        must not override that comparison."""
        self.write_local()
        meta = {"id": DOC_ID, "type": "file", "hash": CONTENT_HASH}
        assert (
            self.engine.should_update_file(
                RELAY_ID, DOC_ID, meta, self.exported_file(), force=True
            )
            is False
        )

    def test_reconcile_requests_built_from_git_connectors(self):
        from git_config import GitConnector

        self.pm.git_config.connectors = [
            GitConnector(
                shared_folder_id=FOLDER_ID,
                relay_id=RELAY_ID,
                url="git@example.com:lens/repo.git",
            )
        ]
        queue = OperationsQueue(self.engine, commit_interval=3600, reconcile_interval=0)

        requests = queue._build_reconcile_requests()

        assert len(requests) == 1
        assert requests[0].force is True
        resource = requests[0].resource
        assert isinstance(resource, S3RemoteFolder)
        assert resource.relay_id == RELAY_ID
        assert resource.folder_id == FOLDER_ID

    def test_reconcile_interval_zero_starts_no_timer(self):
        queue = OperationsQueue(self.engine, commit_interval=3600, reconcile_interval=0)
        assert not hasattr(queue, "reconcile_timer_thread")


def make_change(attempt=None):
    change = {
        "relay_id": RELAY_ID,
        "resource_id": DOC_ID,
        "timestamp": datetime.now(timezone.utc),
    }
    if attempt is not None:
        change["_retry_attempt"] = attempt
    return change


def make_queue(results):
    """OperationsQueue over a mocked engine returning the given results in order
    (last one repeats). Large commit_interval keeps the commit timer quiet."""
    sync_engine = MagicMock()
    sync_engine.process_document_change.side_effect = lambda *a, **k: (
        results.pop(0) if len(results) > 1 else results[0]
    )
    return OperationsQueue(sync_engine, commit_interval=3600), sync_engine


def failure(error="relay 502"):
    return SyncResult(resource=None, operations=[], success=False, error=error)


def success():
    return SyncResult(resource=None, operations=[], success=True)


def wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestRetryScheduling:
    def test_failure_schedules_retry_then_success_clears(self):
        queue, engine = make_queue([failure(), success()])

        queue.enqueue_document_change(make_change())
        assert wait_until(lambda: queue.get_pending_retry_count() == 1)

        key = (RELAY_ID, DOC_ID)
        entry = queue._pending_retries[key]
        assert entry["change_data"]["_retry_attempt"] == 1
        assert entry["due_at"] == pytest.approx(
            time.time() + OperationsQueue.RETRY_DELAYS[0], abs=2
        )

        # Force the retry due and flush; the engine now succeeds.
        with queue._retry_lock:
            queue._pending_retries[key]["due_at"] = time.time() - 1
        queue._flush_due_retries()
        assert wait_until(lambda: engine.process_document_change.call_count == 2)
        assert wait_until(lambda: queue.get_pending_retry_count() == 0)

    def test_repeated_failures_back_off(self):
        queue, engine = make_queue([failure()])

        queue.enqueue_document_change(make_change())
        key = (RELAY_ID, DOC_ID)
        for attempt, delay in enumerate(OperationsQueue.RETRY_DELAYS, start=1):
            assert wait_until(
                lambda: queue._pending_retries.get(key, {})
                .get("change_data", {})
                .get("_retry_attempt")
                == attempt
            )
            assert queue._pending_retries[key]["due_at"] == pytest.approx(
                time.time() + delay, abs=2
            )
            with queue._retry_lock:
                queue._pending_retries[key]["due_at"] = time.time() - 1
            queue._flush_due_retries()

        # Schedule exhausted: dropped loudly, nothing pending.
        assert wait_until(
            lambda: engine.process_document_change.call_count
            == 1 + len(OperationsQueue.RETRY_DELAYS)
        )
        assert wait_until(lambda: queue.get_pending_retry_count() == 0)

    def test_giving_up_logs_error(self, caplog):
        queue, _ = make_queue([success()])
        with caplog.at_level(logging.ERROR, logger="operations_queue"):
            queue._schedule_retry(
                make_change(attempt=len(OperationsQueue.RETRY_DELAYS)), "relay 502"
            )
        assert queue.get_pending_retry_count() == 0
        assert any("Giving up" in r.message for r in caplog.records)

    def test_fresh_webhook_supersedes_pending_retry(self):
        queue, _ = make_queue([success()])
        with queue._retry_lock:
            queue._pending_retries[(RELAY_ID, DOC_ID)] = {
                "due_at": time.time() + 300,
                "change_data": make_change(attempt=3),
            }

        queue.enqueue_document_change(make_change())

        assert queue.get_pending_retry_count() == 0


class TestPerDocFetchFailureEndToEnd(EngineHarness):
    def test_transient_fetch_failure_is_retried_and_file_written(self):
        self.relay_client.fetch_document_content.side_effect = [
            RelayFetchError("relay 502"),
            CONTENT,
        ]
        queue = OperationsQueue(self.engine, commit_interval=3600)

        queue.enqueue_document_change(make_change())
        assert wait_until(lambda: queue.get_pending_retry_count() == 1)
        assert not os.path.exists(self.exported_file())

        with queue._retry_lock:
            queue._pending_retries[(RELAY_ID, DOC_ID)]["due_at"] = time.time() - 1
        queue._flush_due_retries()

        assert wait_until(lambda: os.path.exists(self.exported_file()))
        with open(self.exported_file(), encoding="utf-8") as f:
            assert f.read() == CONTENT
        assert wait_until(lambda: queue.get_pending_retry_count() == 0)

    def test_empty_doc_is_not_an_error_and_not_retried(self):
        """None without an exception means 'doc exists but has no contents' -
        a no-op, not a failure to retry."""
        self.relay_client.fetch_document_content.side_effect = None
        self.relay_client.fetch_document_content.return_value = None

        result = self.engine.process_document_change(
            RELAY_ID, DOC_ID, datetime.now(timezone.utc)
        )

        assert result.success
        assert not os.path.exists(self.exported_file())


class TestRelayClientErrorSeam:
    def _client(self):
        client = RelayClient.__new__(RelayClient)
        client.dm = MagicMock()
        return client

    def test_fetch_error_raises_only_with_flag(self):
        client = self._client()
        client.dm.get_doc_as_update.side_effect = RuntimeError("502 Bad Gateway")
        doc = S3RemoteDocument(RELAY_ID, FOLDER_ID, DOC_ID)

        assert client.fetch_document_content(doc) is None
        with pytest.raises(RelayFetchError):
            client.fetch_document_content(doc, raise_on_error=True)

        canvas = S3RemoteCanvas(RELAY_ID, FOLDER_ID, DOC_ID)
        assert client.fetch_canvas_content(canvas) is None
        with pytest.raises(RelayFetchError):
            client.fetch_canvas_content(canvas, raise_on_error=True)

    def test_empty_doc_returns_none_in_both_modes(self):
        from pycrdt import Doc

        client = self._client()
        client.dm.get_doc_as_update.return_value = Doc().get_update()
        doc = S3RemoteDocument(RELAY_ID, FOLDER_ID, DOC_ID)

        assert client.fetch_document_content(doc) is None
        assert client.fetch_document_content(doc, raise_on_error=True) is None
