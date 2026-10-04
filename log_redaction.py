#!/usr/bin/env python3
"""Central secret redaction for logging.

Added after an incident where y-sweet 404s logged the relay API key
embedded as URL userinfo, e.g. ``http://<token>@relay-server:8080/...``.

The relay/y-sweet API key is handed to the y-sweet SDK as
``scheme://<key>@host/...`` (see ``relay_client.py::_init_document_manager``)
because that is the only auth the SDK accepts; when a request against that
URL fails, the SDK's own error text and the underlying HTTP client's
exception both embed the full URL, and this app's
``logger.error(f"...: {e}")`` calls then print it verbatim. The same
embedding happens for a GitHub (or other host) token placed in a
``git_connector`` URL for private-repo pushes (``git_config.py`` /
``persistence.py``), including inside GitPython's own exception text, which
quotes the full git command line it ran.

Rather than chase every call site, ``install_log_redaction()`` wraps the
root logger's handler(s) once at process startup so every record that
reaches them is masked, whatever module logged it and whether the secret
came through as the message, an interpolated arg, or an exception/
traceback. ``redact()`` is also exposed directly for the few places that
write straight to stdout via ``print()`` instead of logging (CLI commands
that echo a configured connector URL back to the operator's terminal) so a
pasted terminal never carries a live token either.

Masks:
  - URL userinfo:        ``scheme://user:pass@host`` / ``scheme://token@host``
                          -> ``scheme://***@host``
  - Bearer auth headers:  ``Bearer <token>`` -> ``Bearer ***``
  - token query params:   ``?token=...`` / ``?access_token=...`` (and
                          ``&``-joined) -> ``...=***``
  - any exact value registered via ``register_secret()``/
    ``register_secret_from_url()``, or found in a known secret env var
    (``RELAY_SERVER_API_KEY``, ``WEBHOOK_SECRET``, ``JWT_SECRET``,
    ``SSH_PRIVATE_KEY``) -> ``***``
"""

import logging
import os
import re
from typing import List, Optional
from urllib.parse import urlparse

_MASK = "***"

# scheme://user:pass@ or scheme://token@ -> scheme://***@
# [^/\s]* is greedy, so on "user:p@ss@host" it backtracks to the *last* '@'
# before the next '/' or whitespace - i.e. the end of the authority's
# userinfo - rather than stopping at a '@' that is itself part of the
# password.
_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^/\s]*@")
# Bearer <token> -> Bearer *** (covers "Authorization: Bearer x" and dict
# reprs like {'Authorization': 'Bearer x'} alike, since "Bearer" is already
# an unambiguous signal)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)([^\s'\"]+)")
# Query params whose key signals a credential: ?token=, &access_token=,
# ?client_secret=, and AWS SigV4 presigned-URL params (X-Amz-Signature,
# X-Amz-Credential, X-Amz-Security-Token all contain "token"/"signature"/
# "credential") -> ...=***
_SENSITIVE_QUERY_RE = re.compile(
    r"(?i)([?&][\w.-]*(?:token|signature|credential|secret)[\w.-]*=)[^&\s'\"]+"
)

_KNOWN_SECRET_ENV_VARS = (
    "RELAY_SERVER_API_KEY",
    "WEBHOOK_SECRET",
    "JWT_SECRET",
    "SSH_PRIVATE_KEY",
)

# Extra exact-match secrets registered at runtime (e.g. a token parsed out of
# a git_connector URL at config-load time). Env vars are read fresh on every
# call in _known_secrets() rather than cached, so tests can monkeypatch
# os.environ without needing a reset hook.
_registered_secrets: List[str] = []


def register_secret(value: Optional[str]) -> None:
    """Register one more exact string to mask wherever it appears in a log
    line. Safe to call with None/empty; safe to call repeatedly with the
    same value."""
    if value and value not in _registered_secrets:
        _registered_secrets.append(value)


def register_secret_from_url(url: Optional[str]) -> None:
    """Pull a userinfo-embedded token out of `url` (if any) and register it
    by exact value too - a backstop for the rare case where it later shows
    up outside a scheme://...@ context (e.g. a bare value in a GitPython
    exception's extra-output section).

    Only http(s)/y-sweet URLs are considered: an ssh:// URL's "username" is
    a fixed, non-secret convention (git@github.com uses the literal user
    "git"), not a credential, and registering it would mask that word
    everywhere. When the URL has both a username and a password
    (https://oauth2:TOKEN@host, https://x-access-token:TOKEN@host), the
    username is conventionally a fixed non-secret role name and the
    password is the actual secret, so only the password is registered;
    with no password (https://TOKEN@host) the username is the token.
    """
    if not url:
        return
    try:
        parsed = urlparse(url)
    except ValueError:
        return
    if parsed.scheme not in ("http", "https", "ys", "yss"):
        return
    if parsed.password:
        register_secret(parsed.password)
    elif parsed.username:
        register_secret(parsed.username)


def _known_secrets() -> List[str]:
    env_secrets = [os.environ.get(var) for var in _KNOWN_SECRET_ENV_VARS]
    return [s for s in env_secrets if s] + _registered_secrets


def redact(text: Optional[str]) -> Optional[str]:
    """Return `text` with any URL userinfo, bearer token, token query
    param, or known secret value masked. None/empty input is returned
    unchanged."""
    if not text:
        return text
    redacted = _USERINFO_RE.sub(lambda m: f"{m.group(1)}{_MASK}@", text)
    redacted = _BEARER_RE.sub(lambda m: f"{m.group(1)}{_MASK}", redacted)
    redacted = _SENSITIVE_QUERY_RE.sub(lambda m: f"{m.group(1)}{_MASK}", redacted)
    for secret in _known_secrets():
        if secret and secret in redacted:
            redacted = redacted.replace(secret, _MASK)
    return redacted


class RedactingFilter(logging.Filter):
    """Masks the rendered message (and args) on every record that reaches
    the handler this is attached to, before formatting runs."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class _RedactingFormatter(logging.Formatter):
    """Wraps another Formatter and masks its fully rendered output.

    `inner.format(record)` does the normal work, including calling its own
    `formatException`/`formatStack` to render and cache any traceback onto
    `record.exc_text` - we don't need to override those ourselves, since the
    single `redact()` call below runs on the complete string those produce,
    message and traceback alike. This is the backstop for secrets that only
    appear once exc_info is formatted, independent of the message-level
    RedactingFilter above (which runs before formatting, so it can't see
    traceback text).

    Only covers handlers this has actually been installed on - a handler
    added to the logger later, without going through install_log_redaction()
    again, is not wrapped and will format unredacted.
    """

    def __init__(self, inner: logging.Formatter):
        super().__init__()
        self._inner = inner

    def format(self, record: logging.LogRecord) -> str:
        return redact(self._inner.format(record))


def install_log_redaction(logger: Optional[logging.Logger] = None) -> None:
    """Idempotently attach the redaction filter + formatter wrapper to every
    handler on `logger` (the root logger by default).

    Handlers, not loggers, are the single choke point every module's
    records flow through regardless of which `logging.getLogger(__name__)`
    produced them (a Filter added to a Logger only runs for records
    originating at that exact logger, not ones that merely propagate
    through it) - so this must attach to the handler(s) installed by
    `logging.basicConfig()`. Call once at process startup, right after it.
    """
    target = logger or logging.getLogger()
    for handler in target.handlers:
        if getattr(handler, "_redaction_installed", False):
            continue
        handler.addFilter(RedactingFilter())
        handler.setFormatter(_RedactingFormatter(handler.formatter or logging.Formatter()))
        handler._redaction_installed = True
