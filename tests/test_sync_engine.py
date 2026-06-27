#!/usr/bin/env python3

from datetime import datetime, timezone
from unittest.mock import Mock

from pycrdt import Doc, Map

from models import OperationType
from persistence import PersistenceManager
from s3rn import S3RemoteDocument, S3RemoteFolder
from sync_engine import SyncEngine


def test_process_document_change_refreshes_filemeta_for_unknown_document(tmp_path):
    """Unknown content webhooks should refresh filemeta before skipping."""
    relay_id = "relay-123"
    stale_folder_id = "stale-folder"
    folder_id = "folder-123"
    document_id = "doc-123"
    document_path = "/new.md"
    content = "# New document\n"

    persistence = PersistenceManager(str(tmp_path))
    persistence.filemeta_folders[relay_id] = {
        stale_folder_id: {},
        folder_id: {},
    }
    persistence.document_hashes[relay_id] = {}
    persistence.local_file_state[relay_id] = {}
    persistence._build_resource_index(relay_id)

    stale_doc = Doc()

    folder_doc = Doc()
    filemeta = folder_doc.get("filemeta_v0", type=Map)
    filemeta[document_path] = {"id": document_id, "type": "document"}

    relay_client = Mock()

    def get_doc_object(resource):
        if isinstance(resource, S3RemoteFolder) and resource.folder_id == stale_folder_id:
            return stale_doc
        if isinstance(resource, S3RemoteFolder) and resource.folder_id == folder_id:
            return folder_doc
        raise AssertionError(f"Unexpected folder fetch: {resource!r}")

    relay_client.get_doc_object.side_effect = get_doc_object
    relay_client._map_to_dict.side_effect = lambda value: dict(value)
    relay_client.fetch_document_content.return_value = content

    engine = SyncEngine(
        data_dir=str(tmp_path),
        relay_client=relay_client,
        persistence_manager=persistence,
    )

    result = engine.process_document_change(
        relay_id=relay_id,
        resource_id=document_id,
        timestamp=datetime.now(timezone.utc),
    )

    assert result.success is True
    assert len(result.operations) == 1
    operation = result.operations[0]
    assert operation.type == OperationType.UPDATE
    assert operation.path == document_path
    assert isinstance(operation.document_resource, S3RemoteDocument)
    assert operation.document_resource.folder_id == folder_id
    assert operation.document_resource.document_id == document_id

    relay_client.fetch_document_content.assert_called_once()
    fetched_resource = relay_client.fetch_document_content.call_args.args[0]
    assert isinstance(fetched_resource, S3RemoteDocument)
    assert fetched_resource.folder_id == folder_id
    assert fetched_resource.document_id == document_id

    assert persistence.get_resource_path(relay_id, document_id) == document_path
    assert persistence.document_hashes[relay_id][document_id]
    assert (tmp_path / "repos" / relay_id / folder_id / "new.md").read_text() == content
