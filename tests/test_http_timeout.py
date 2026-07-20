import pytest
import requests

from http_timeout import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
    install_default_timeout,
)


class StopRequest(Exception):
    pass


@pytest.fixture
def captured_send_kwargs(monkeypatch):
    captured = {}

    def fake_send(self, request, **kwargs):
        captured.update(kwargs)
        raise StopRequest()

    monkeypatch.setattr(requests.Session, "send", fake_send)
    return captured


def test_timeout_injected_for_module_helpers(captured_send_kwargs):
    install_default_timeout()
    with pytest.raises(StopRequest):
        requests.get("http://example.invalid/doc")
    assert captured_send_kwargs["timeout"] == (
        DEFAULT_CONNECT_TIMEOUT,
        DEFAULT_READ_TIMEOUT,
    )


def test_timeout_injected_for_requests_request(captured_send_kwargs):
    install_default_timeout()
    with pytest.raises(StopRequest):
        requests.request("GET", "http://example.invalid/doc")
    assert captured_send_kwargs["timeout"] == (
        DEFAULT_CONNECT_TIMEOUT,
        DEFAULT_READ_TIMEOUT,
    )


def test_explicit_timeout_wins(captured_send_kwargs):
    install_default_timeout()
    with pytest.raises(StopRequest):
        requests.get("http://example.invalid/doc", timeout=5)
    assert captured_send_kwargs["timeout"] == 5


def test_install_is_idempotent():
    install_default_timeout()
    patched = requests.Session.request
    install_default_timeout()
    assert requests.Session.request is patched
