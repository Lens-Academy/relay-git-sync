"""Default HTTP timeout guard (incident 2026-07-17).

y-sweet-sdk calls requests without a timeout, so a silently dropped
connection (e.g. Cloudflare idling out an idle keep-alive socket) blocks
the caller forever in recv(). In app.py that meant the single
operations-queue worker wedged and all syncing stopped while webhooks
kept returning 200. Until the SDK passes timeouts itself, inject a
default timeout at requests' single choke point, Session.request, which
covers requests.request, requests.get/post/..., and any Session usage.

Note: the read timeout is a between-bytes timeout, not a total-request
deadline; it closes the dropped-connection hang but not a byte-trickling
degraded proxy.
"""

import functools
import os

import requests

DEFAULT_CONNECT_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_CONNECT", "10"))
DEFAULT_READ_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_READ", "120"))


def install_default_timeout():
    """Idempotently patch Session.request with a default timeout."""
    if getattr(requests.Session.request, "_has_default_timeout", False):
        return

    original_request = requests.Session.request

    @functools.wraps(original_request)
    def request_with_default_timeout(self, method, url, **kwargs):
        kwargs.setdefault("timeout", (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT))
        return original_request(self, method, url, **kwargs)

    request_with_default_timeout._has_default_timeout = True
    requests.Session.request = request_with_default_timeout
