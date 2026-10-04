#!/usr/bin/env python3
"""Hashless "file" entries (the editor's .html pages) are text documents.

Relay gives every text document whose path does not end in .md type "file"
without a hash. Before this fix git-sync treated those as blobs, so .html pages
never reached the synced repo. Real blobs always carry a hash and stay files.
"""

import shutil
import tempfile

from models import create_document_resource_from_metadata, effective_resource_type
from persistence import PersistenceManager
from s3rn import S3RemoteDocument, S3RemoteFile

RELAY_ID = "11111111-1111-4111-8111-111111111111"
FOLDER_ID = "22222222-2222-4222-8222-222222222222"
HTML_ID = "33333333-3333-4333-8333-333333333333"
IMG_ID = "44444444-4444-4444-8444-444444444444"

HTML_META = {"id": HTML_ID, "type": "file"}
IMG_META = {"id": IMG_ID, "type": "file", "hash": "ab" * 32, "mimetype": "image/png"}


class TestEffectiveResourceType:
    def test_hashless_file_is_markdown(self):
        assert effective_resource_type(HTML_META) == "markdown"

    def test_file_with_hash_stays_file(self):
        assert effective_resource_type(IMG_META) == "file"

    def test_other_types_unchanged(self):
        for t in ("markdown", "canvas", "image", "folder"):
            assert effective_resource_type({"id": "x", "type": t}) == t
        assert effective_resource_type({"id": "x", "type": "image"}) == "image"

    def test_missing_type_is_none(self):
        assert effective_resource_type({"id": "x"}) is None


class TestDocumentResource:
    def test_html_page_becomes_document(self):
        res = create_document_resource_from_metadata(RELAY_ID, FOLDER_ID, HTML_META)
        assert isinstance(res, S3RemoteDocument)

    def test_blob_stays_file(self):
        res = create_document_resource_from_metadata(RELAY_ID, FOLDER_ID, IMG_META)
        assert isinstance(res, S3RemoteFile)


class TestResourceIndex:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pm = PersistenceManager(self.temp_dir)
        self.pm.load_persistent_data(RELAY_ID)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_index_records_html_as_document(self):
        self.pm.update_resource_index_for_document(
            RELAY_ID, HTML_ID, FOLDER_ID, "/Pages/intro.html", HTML_META
        )
        self.pm.update_resource_index_for_document(
            RELAY_ID, IMG_ID, FOLDER_ID, "/attachments/a.png", IMG_META
        )
        assert isinstance(self.pm.lookup_resource(RELAY_ID, HTML_ID), S3RemoteDocument)
        assert isinstance(self.pm.lookup_resource(RELAY_ID, IMG_ID), S3RemoteFile)

    def test_build_index_from_filemeta_records_html_as_document(self):
        self.pm.filemeta_folders.setdefault(RELAY_ID, {})[FOLDER_ID] = {
            "/Pages/intro.html": HTML_META,
            "/attachments/a.png": IMG_META,
        }
        self.pm._build_resource_index(RELAY_ID)
        assert isinstance(self.pm.lookup_resource(RELAY_ID, HTML_ID), S3RemoteDocument)
        assert isinstance(self.pm.lookup_resource(RELAY_ID, IMG_ID), S3RemoteFile)
