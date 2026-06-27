#!/usr/bin/env python3

from unittest.mock import Mock

from webhook_handler import WebhookProcessor


def test_process_webhook_uses_payload_timestamp_from_live_relay_envelope():
    relay_id = "12345678-1234-1234-1234-123456789abc"
    document_id = "87654321-4321-4321-4321-cba987654321"
    timestamp = "2026-06-27T22:49:28.014730444+00:00"

    relay_client = Mock()
    relay_client.extract_relay_id.return_value = relay_id
    relay_client.extract_document_id.return_value = document_id

    processor = WebhookProcessor(relay_client)

    result = processor.process_webhook(
        {
            "eventType": "document.updated",
            "eventId": "evt_test",
            "payload": {
                "doc_id": f"{relay_id}-{document_id}",
                "metadata": {},
                "timestamp": timestamp,
                "user": None,
            },
        }
    )

    assert result is not None
    assert result["relay_id"] == relay_id
    assert result["resource_id"] == document_id
    assert result["timestamp"].isoformat().startswith("2026-06-27T22:49:28.014730")
    assert result["timestamp"].utcoffset().total_seconds() == 0
