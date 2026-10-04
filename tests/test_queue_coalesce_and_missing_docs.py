#!/usr/bin/env python3
"""Regression tests for the webhook backlog of 2026-10-04 (Lens Academy).

An agent's burst sent ~1058 webhooks in 5 h. The single worker processed them
one by one in arrival order, ~16 s per shared-folder event, so at peak 154
events were waiting (52 of them repeats for the same folder) and edits reached
GitHub ~32 min late.

1. Repeat events for a resource already waiting in the queue are coalesced:
   the worker fetches current state at processing time, so one pass covers
   every change before it. An event arriving while the resource is being
   processed is still queued once more.
2. Docs listed in filemeta but deleted on the relay were re-fetched on every
   folder event ("Skipping create operation ...: document not found"). The
   miss is now cached for MISSING_DOC_TTL, cleared early by a webhook for the
   doc or by a forced reconcile sweep.
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from models import OperationType, SyncRequest, SyncResult
from operations_queue import OperationsQueue
from s3rn import S3RemoteCanvas, S3RemoteDocument, S3RemoteFolder
from tests.test_sweep_noop_and_retry import (
    CONTENT,
    DOC_ID,
    DOC_PATH,
    FOLDER_ID,
    RELAY_ID,
    EngineHarness,
    wait_until,
)

OTHER_ID = "44444444-4444-4444-8444-444444444444"
BLOCKER_ID = "55555555-5555-4555-8555-555555555555"
T0 = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def change(resource_id, seconds=0, attempt=None):
    data = {
        "relay_id": RELAY_ID,
        "resource_id": resource_id,
        "timestamp": T0 + timedelta(seconds=seconds),
    }
    if attempt is not None:
        data["_retry_attempt"] = attempt
    return data


class GatedEngine:
    """Mock engine whose process_document_change blocks on chosen resources
    until released, recording every call."""

    def __init__(self, block_ids):
        self.block_ids = set(block_ids)
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.lock = threading.Lock()
        self.engine = MagicMock()
        self.engine.process_document_change.side_effect = self._process

    def _process(self, relay_id, resource_id, timestamp):
        with self.lock:
            self.calls.append((resource_id, timestamp))
            block = resource_id in self.block_ids
            self.block_ids.discard(resource_id)  # block the first call only
        if block:
            self.started.set()
            assert self.release.wait(10)
        return SyncResult(resource=None, operations=[], success=True)

    def ids(self):
        with self.lock:
            return [c[0] for c in self.calls]


def make_queue(gate):
    return OperationsQueue(gate.engine, commit_interval=3600, reconcile_interval=0)


class TestQueueCoalescing:
    def test_repeats_waiting_in_queue_are_processed_once(self):
        gate = GatedEngine([BLOCKER_ID])
        queue = make_queue(gate)

        queue.enqueue_document_change(change(BLOCKER_ID))
        assert gate.started.wait(5)  # worker busy, everything else waits

        for i in range(5):
            queue.enqueue_document_change(change(DOC_ID, seconds=i))
        queue.enqueue_document_change(change(OTHER_ID))
        queue.enqueue_document_change(change(DOC_ID, seconds=9))

        assert queue.get_queue_size() == 2
        assert queue.get_pending_change_count() == 2

        gate.release.set()
        assert wait_until(lambda: len(gate.ids()) == 3)
        time.sleep(0.2)
        assert gate.ids() == [BLOCKER_ID, DOC_ID, OTHER_ID]
        # The coalesced event carries the newest timestamp.
        assert gate.calls[1][1] == T0 + timedelta(seconds=9)
        assert queue.get_pending_change_count() == 0

    def test_event_during_processing_is_queued_again(self):
        """The worker releases the key when it takes the event, so a change
        that lands mid-fetch still gets its own pass."""
        gate = GatedEngine([DOC_ID])
        queue = make_queue(gate)

        queue.enqueue_document_change(change(DOC_ID))
        assert gate.started.wait(5)
        assert queue.get_pending_change_count() == 0

        queue.enqueue_document_change(change(DOC_ID, seconds=1))
        queue.enqueue_document_change(change(DOC_ID, seconds=2))
        assert queue.get_queue_size() == 1

        gate.release.set()
        assert wait_until(lambda: len(gate.ids()) == 2)
        time.sleep(0.2)
        assert gate.ids() == [DOC_ID, DOC_ID]
        assert gate.calls[1][1] == T0 + timedelta(seconds=2)

    def test_fresh_webhook_resets_retry_count_of_waiting_retry(self):
        gate = GatedEngine([BLOCKER_ID])
        queue = make_queue(gate)
        queue.enqueue_document_change(change(BLOCKER_ID))
        assert gate.started.wait(5)

        # A due retry is re-queued, then a fresh webhook for the same doc.
        with queue._retry_lock:
            queue._pending_retries[(RELAY_ID, DOC_ID)] = {
                "due_at": time.time() - 1,
                "change_data": change(DOC_ID, attempt=3),
            }
        queue._flush_due_retries()
        queue.enqueue_document_change(change(DOC_ID, seconds=5))

        assert queue.get_queue_size() == 1
        waiting = queue._pending_changes[(RELAY_ID, DOC_ID)]
        assert "_retry_attempt" not in waiting
        assert waiting["timestamp"] == T0 + timedelta(seconds=5)
        gate.release.set()

    def test_due_retry_folds_into_waiting_fresh_event(self):
        gate = GatedEngine([BLOCKER_ID])
        queue = make_queue(gate)
        queue.enqueue_document_change(change(BLOCKER_ID))
        assert gate.started.wait(5)

        queue.enqueue_document_change(change(DOC_ID, seconds=5))
        # Inject the retry directly: enqueue_document_change would drop it.
        with queue._retry_lock:
            queue._pending_retries[(RELAY_ID, DOC_ID)] = {
                "due_at": time.time() - 1,
                "change_data": change(DOC_ID, attempt=2),
            }
        queue._flush_due_retries()

        assert queue.get_queue_size() == 1
        waiting = queue._pending_changes[(RELAY_ID, DOC_ID)]
        assert "_retry_attempt" not in waiting
        assert waiting["timestamp"] == T0 + timedelta(seconds=5)

        gate.release.set()
        assert wait_until(lambda: len(gate.ids()) == 2)
        time.sleep(0.2)
        assert gate.ids() == [BLOCKER_ID, DOC_ID]

    def test_sync_requests_are_not_coalesced(self):
        gate = GatedEngine([BLOCKER_ID])
        queue = make_queue(gate)
        queue.enqueue_document_change(change(BLOCKER_ID))
        assert gate.started.wait(5)

        folder = S3RemoteFolder(RELAY_ID, FOLDER_ID)
        queue.enqueue_sync_request(SyncRequest(resource=folder, timestamp=T0))
        queue.enqueue_sync_request(SyncRequest(resource=folder, timestamp=T0))
        assert queue.get_queue_size() == 2
        gate.release.set()


class TestMissingDocCache(EngineHarness):
    """Doc listed in filemeta, no local file, deleted on the relay."""

    def setup_method(self):
        super().setup_method()
        self.relay_client.fetch_document_content.return_value = None
        self.doc = S3RemoteDocument(RELAY_ID, FOLDER_ID, DOC_ID)

    def doc_op(self, operations):
        ops = [op for op in operations if op.path == DOC_PATH]
        assert len(ops) == 1
        return ops[0]

    def test_second_sweep_skips_the_fetch(self):
        op = self.doc_op(self.sweep())
        assert op.type == OperationType.CREATE and op.error
        assert self.relay_client.fetch_document_content.call_count == 1

        op = self.doc_op(self.sweep())
        assert op.type == OperationType.CREATE and op.error
        assert self.relay_client.fetch_document_content.call_count == 1

    def test_entry_expires_after_ttl(self):
        self.sweep()
        with self.engine._missing_docs_lock:
            key = (RELAY_ID, DOC_ID)
            self.engine._missing_docs[key] = time.monotonic() - 1

        self.relay_client.fetch_document_content.return_value = CONTENT
        self.sweep()
        assert self.relay_client.fetch_document_content.call_count == 2
        with open(self.exported_file(), encoding="utf-8") as f:
            assert f.read() == CONTENT

    def test_webhook_for_the_doc_clears_the_entry(self):
        self.sweep()
        assert self.engine._is_known_missing(self.doc)

        # The doc's own webhook: per-doc path fetches with raise_on_error and
        # exports it, regardless of the cache.
        self.relay_client.fetch_document_content.return_value = CONTENT
        result = self.engine.process_document_change(RELAY_ID, DOC_ID, T0)
        assert result.success
        assert not self.engine._is_known_missing(self.doc)
        with open(self.exported_file(), encoding="utf-8") as f:
            assert f.read() == CONTENT

    def test_forced_reconcile_sweep_refetches(self):
        self.sweep()
        assert self.relay_client.fetch_document_content.call_count == 1

        self.relay_client.fetch_document_content.return_value = CONTENT
        self.relay_client.get_document_structure.return_value = (
            MagicMock(),
            {"type": "folder", "filemeta": dict(self.filemeta)},
        )
        result = self.engine.process_sync_request(
            SyncRequest(resource=self.folder, timestamp=T0, force=True)
        )
        assert result.success
        assert self.relay_client.fetch_document_content.call_count == 2
        with open(self.exported_file(), encoding="utf-8") as f:
            assert f.read() == CONTENT

    def test_found_doc_is_not_cached(self):
        self.relay_client.fetch_document_content.return_value = CONTENT
        self.sweep()
        assert not self.engine._is_known_missing(self.doc)

    def test_update_path_and_canvas_are_cached_too(self):
        canvas_path = "/Boards/board.canvas"
        canvas_id = OTHER_ID
        self.filemeta[canvas_path] = {"id": canvas_id, "type": "canvas"}
        self.relay_client.fetch_canvas_content.return_value = None
        # Doc exists locally but our export hash is stale -> UPDATE path.
        self.write_local("old\n")
        self.pm.document_hashes[RELAY_ID][DOC_ID] = "0" * 64

        self.sweep()
        self.sweep()
        assert self.relay_client.fetch_document_content.call_count == 1
        assert self.relay_client.fetch_canvas_content.call_count == 1
        assert self.engine._is_known_missing(
            S3RemoteCanvas(RELAY_ID, FOLDER_ID, canvas_id)
        )
