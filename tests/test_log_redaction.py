#!/usr/bin/env python3
"""Tests for log_redaction.py.

Covers the incident this guards against: a y-sweet 404 (or any error from a
URL built with embedded credentials) must never reach a log line with the
token intact, whether it arrives as a plain message, an exception string,
or a formatted traceback - and a normal log line must come through
byte-for-byte unchanged.
"""

import io
import logging

import pytest

import log_redaction
from log_redaction import (
    install_log_redaction,
    redact,
    register_secret,
    register_secret_from_url,
)


@pytest.fixture(autouse=True)
def _clean_registered_secrets():
    """Each test gets its own registered-secrets list so one test's
    register_secret() call can't leak into another."""
    original = list(log_redaction._registered_secrets)
    log_redaction._registered_secrets.clear()
    yield
    log_redaction._registered_secrets.clear()
    log_redaction._registered_secrets.extend(original)


@pytest.fixture
def clean_secret_env(monkeypatch):
    """Ensure the known secret env vars are unset unless a test sets them."""
    for var in log_redaction._KNOWN_SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestRedactUrlUserinfo:
    def test_masks_token_only_userinfo(self):
        text = "GET http://sk_live_abcdef123@relay-server:8080/f/doc/download-url"
        assert "sk_live_abcdef123" not in redact(text)
        assert redact(text) == "GET http://***@relay-server:8080/f/doc/download-url"

    def test_masks_user_colon_pass_userinfo(self):
        text = "https://user:hunter2@github.com/example/repo.git"
        assert redact(text) == "https://***@github.com/example/repo.git"

    def test_masks_ys_scheme_connection_string(self):
        # The y-sweet connection string built in relay_client.py's
        # _init_document_manager, if it ever ends up in a log line directly.
        text = "Connecting with ys://sk_live_realKeyValue@relay.example.com:8080"
        assert "sk_live_realKeyValue" not in redact(text)

    def test_leaves_plain_url_without_userinfo_unchanged(self):
        text = "Fetching https://relay-server:8080/f/doc/download-url"
        assert redact(text) == text

    def test_masks_through_last_at_when_password_contains_at(self):
        # A naive non-greedy/negated-class match stops at the *first* '@',
        # leaking the rest of a password that itself contains '@'.
        text = "clone failed https://user:p@ssw0rd@github.com/org/repo.git"
        redacted = redact(text)
        assert "p@ssw0rd" not in redacted
        assert redacted == "clone failed https://***@github.com/org/repo.git"


class TestRedactBearerToken:
    def test_masks_authorization_header_line(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
        assert redact(text) == "Authorization: Bearer ***"

    def test_masks_bearer_inside_dict_repr(self):
        text = "sending headers {'Authorization': 'Bearer abc.def.ghi'}"
        redacted = redact(text)
        assert "abc.def.ghi" not in redacted
        assert "Bearer ***" in redacted


class TestRedactTokenQueryParam:
    def test_masks_token_param(self):
        text = "GET /download?token=supersecret123&hash=abc"
        assert redact(text) == "GET /download?token=***&hash=abc"

    def test_masks_access_token_param(self):
        text = "GET /oauth/callback?access_token=supersecret456"
        assert redact(text) == "GET /oauth/callback?access_token=***"

    def test_masks_aws_presigned_url_params(self):
        # relay_client.py's S3 file download hits a presigned URL; on
        # failure the exception's "for url: ..." text includes these.
        text = (
            "404 Client Error: Not Found for url: "
            "https://bucket.s3.amazonaws.com/key"
            "?X-Amz-Credential=AKIAREALCREDENTIAL%2F20261004"
            "&X-Amz-Signature=realsignaturevalue123"
            "&X-Amz-Security-Token=realsecuritytoken456"
        )
        redacted = redact(text)
        assert "AKIAREALCREDENTIAL" not in redacted
        assert "realsignaturevalue123" not in redacted
        assert "realsecuritytoken456" not in redacted
        assert "X-Amz-Credential=***" in redacted
        assert "X-Amz-Signature=***" in redacted
        assert "X-Amz-Security-Token=***" in redacted

    def test_masks_client_secret_param(self):
        text = "POST /token?client_secret=abcSecretXyz&grant_type=client_credentials"
        assert redact(text) == "POST /token?client_secret=***&grant_type=client_credentials"


class TestRedactKnownSecrets:
    def test_masks_exact_env_secret_without_wrapper(self, clean_secret_env, monkeypatch):
        monkeypatch.setenv("RELAY_SERVER_API_KEY", "bareRelayKeyNoWrapper")
        text = "debug: using key bareRelayKeyNoWrapper for this request"
        assert redact(text) == "debug: using key *** for this request"

    def test_masks_registered_secret(self):
        register_secret("myRegisteredGithubToken")
        text = "clone failed: myRegisteredGithubToken rejected by remote"
        assert redact(text) == "clone failed: *** rejected by remote"

    def test_register_secret_from_url_extracts_bare_token_username(self):
        register_secret_from_url("https://ghp_extracted@github.com/example/repo.git")
        # The pattern-based userinfo mask already catches this inside a URL;
        # exercise the exact-match path by looking for the bare token.
        assert redact("token was ghp_extracted") == "token was ***"

    def test_register_secret_from_url_prefers_password_over_username(self):
        # https://oauth2:TOKEN@host and https://x-access-token:TOKEN@host are
        # the common "fixed role name as username, real secret as password"
        # forms (GitLab deploy tokens, GitHub App installation tokens). Only
        # the password should be registered - see next test for why.
        register_secret_from_url("https://oauth2:realSecretValue@gitlab.com/org/repo.git")
        assert redact("leaked realSecretValue here") == "leaked *** here"
        assert "oauth2" not in log_redaction._registered_secrets

    def test_register_secret_from_url_skips_ssh_fixed_username(self):
        # git@github.com:org/repo.git-style URLs (and ssh:// URLs) use the
        # fixed, non-secret username "git" by SSH convention - registering
        # it as an exact-match secret would mask that word everywhere.
        register_secret_from_url("ssh://git@github.com/example/repo.git")
        assert log_redaction._registered_secrets == []
        assert redact("Initialized git repository with .gitignore") == (
            "Initialized git repository with .gitignore"
        )

    def test_register_secret_ignores_empty_values(self):
        register_secret(None)
        register_secret("")
        assert log_redaction._registered_secrets == []


class TestRedactNormalLines:
    def test_unrelated_text_unchanged(self):
        text = "2026-10-04 12:00:00 - INFO - Synced folder abc123 (3 operations)"
        assert redact(text) == text

    def test_empty_and_none_safe(self):
        assert redact("") == ""
        assert redact(None) is None


def _make_isolated_logger(name):
    """A logger + StreamHandler pair that is not the root logger, so tests
    don't depend on (or pollute) logging.basicConfig global state."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, stream


class TestInstallLogRedactionIntegration:
    def test_log_record_with_embedded_token_url_is_masked(self):
        logger, stream = _make_isolated_logger("test_log_redaction.url")
        install_log_redaction(logger)

        logger.error(
            "Error fetching document S3RemoteFile(x): 404 Client Error: "
            "Not Found for url: http://sk_live_realSecretValue@relay-server:8080/f/doc/download-url"
        )

        output = stream.getvalue()
        assert "sk_live_realSecretValue" not in output
        assert "***@relay-server:8080" in output

    def test_exception_traceback_is_masked(self):
        logger, stream = _make_isolated_logger("test_log_redaction.exc")
        install_log_redaction(logger)

        try:
            raise RuntimeError(
                "fetch failed for url https://ghp_realGithubTokenValue@github.com/org/repo.git"
            )
        except RuntimeError:
            logger.exception("Document fetch traceback")

        output = stream.getvalue()
        assert "ghp_realGithubTokenValue" not in output
        assert "Traceback" in output  # the traceback itself still comes through
        assert "***@github.com/org/repo.git" in output

    def test_bearer_header_in_log_record_is_masked(self):
        logger, stream = _make_isolated_logger("test_log_redaction.bearer")
        install_log_redaction(logger)

        logger.warning("Request failed with Authorization: Bearer realBearerTokenValue")

        output = stream.getvalue()
        assert "realBearerTokenValue" not in output
        assert "Bearer ***" in output

    def test_normal_log_line_is_unaffected(self):
        logger, stream = _make_isolated_logger("test_log_redaction.normal")
        install_log_redaction(logger)

        logger.info("Synced folder 12345678-1234-1234-1234-123456789abc (2 operations)")

        output = stream.getvalue()
        assert "Synced folder 12345678-1234-1234-1234-123456789abc (2 operations)" in output

    def test_install_is_idempotent(self):
        logger, stream = _make_isolated_logger("test_log_redaction.idempotent")
        install_log_redaction(logger)
        install_log_redaction(logger)  # second call should not double-wrap

        logger.error("http://tok@host:8080/x")

        output = stream.getvalue()
        # A single call line, masked exactly once (not "***@***@host" from a
        # double-wrapped formatter, and not printed twice from a duplicate filter).
        assert output.count("tok") == 0
        assert output.count("***@host:8080/x") == 1
