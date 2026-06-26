"""Tests for automatic Claude Code OAuth token refresh.

When the cached access token in ~/.claude/.credentials.json is expired but a
refresh token is present, the loader should exchange it for a fresh access
token, persist the result back to the file, and return the live credential
instead of failing the request with an auth error (regression: expired-token
runs surfaced "Could not resolve authentication method").
"""

import json
import time

import deerflow.models.credential_loader as cl
from deerflow.models.credential_loader import load_claude_code_credential


def _clear_claude_code_env(monkeypatch) -> None:
    for env_var in (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_CREDENTIALS_PATH",
    ):
        monkeypatch.delenv(env_var, raising=False)


def _write_creds(path, *, access, refresh, expires_at, extra=None) -> None:
    oauth = {"accessToken": access, "refreshToken": refresh, "expiresAt": expires_at}
    if extra:
        oauth.update(extra)
    path.write_text(json.dumps({"claudeAiOauth": oauth}))


def test_expired_token_is_refreshed_and_persisted(tmp_path, monkeypatch):
    _clear_claude_code_env(monkeypatch)
    cred_path = tmp_path / "creds.json"
    expired_at = int(time.time() * 1000) - 1000  # already expired
    _write_creds(
        cred_path,
        access="sk-ant-oat01-old",
        refresh="sk-ant-ort01-old",
        expires_at=expired_at,
        extra={"scopes": ["user:inference"]},
    )
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))

    captured = {}

    def fake_request(refresh_token):
        captured["refresh_token"] = refresh_token
        return {
            "access_token": "sk-ant-oat01-new",
            "refresh_token": "sk-ant-ort01-new",
            "expires_in": 3600,
        }

    monkeypatch.setattr(cl, "_request_oauth_token", fake_request)

    cred = load_claude_code_credential()

    # Returned the refreshed credential
    assert cred is not None
    assert cred.access_token == "sk-ant-oat01-new"
    assert cred.refresh_token == "sk-ant-ort01-new"
    assert not cred.is_expired
    assert captured["refresh_token"] == "sk-ant-ort01-old"

    # Persisted back to disk, preserving unrelated fields
    on_disk = json.loads(cred_path.read_text())["claudeAiOauth"]
    assert on_disk["accessToken"] == "sk-ant-oat01-new"
    assert on_disk["refreshToken"] == "sk-ant-ort01-new"
    assert on_disk["expiresAt"] > int(time.time() * 1000)
    assert on_disk["scopes"] == ["user:inference"]


def test_expired_token_without_refresh_token_returns_none(tmp_path, monkeypatch):
    _clear_claude_code_env(monkeypatch)
    # Isolate HOME so the loader cannot fall back to the real default file.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cred_path = tmp_path / "creds.json"
    _write_creds(
        cred_path,
        access="sk-ant-oat01-old",
        refresh="",
        expires_at=int(time.time() * 1000) - 1000,
    )
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))

    def fail_request(refresh_token):  # pragma: no cover - must not be called
        raise AssertionError("refresh should not be attempted without a refresh token")

    monkeypatch.setattr(cl, "_request_oauth_token", fail_request)

    assert load_claude_code_credential() is None


def test_expired_token_refresh_failure_returns_none_and_keeps_file(tmp_path, monkeypatch):
    _clear_claude_code_env(monkeypatch)
    # Isolate HOME so the loader cannot fall back to the real default file.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cred_path = tmp_path / "creds.json"
    expired_at = int(time.time() * 1000) - 1000
    _write_creds(cred_path, access="sk-ant-oat01-old", refresh="sk-ant-ort01-old", expires_at=expired_at)
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))

    monkeypatch.setattr(cl, "_request_oauth_token", lambda refresh_token: None)

    assert load_claude_code_credential() is None
    # File is left untouched on failure
    on_disk = json.loads(cred_path.read_text())["claudeAiOauth"]
    assert on_disk["accessToken"] == "sk-ant-oat01-old"
    assert on_disk["expiresAt"] == expired_at


def test_valid_token_does_not_trigger_refresh(tmp_path, monkeypatch):
    _clear_claude_code_env(monkeypatch)
    cred_path = tmp_path / "creds.json"
    _write_creds(
        cred_path,
        access="sk-ant-oat01-live",
        refresh="sk-ant-ort01-live",
        expires_at=int(time.time() * 1000) + 3_600_000,
    )
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", str(cred_path))

    def fail_request(refresh_token):  # pragma: no cover - must not be called
        raise AssertionError("refresh should not run for a live token")

    monkeypatch.setattr(cl, "_request_oauth_token", fail_request)

    cred = load_claude_code_credential()
    assert cred is not None
    assert cred.access_token == "sk-ant-oat01-live"


def test_request_oauth_token_posts_refresh_grant(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"access_token": "a", "refresh_token": "r", "expires_in": 3600}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    result = cl._request_oauth_token("sk-ant-ort01-old")

    assert result == {"access_token": "a", "refresh_token": "r", "expires_in": 3600}
    assert captured["url"] == cl.OAUTH_TOKEN_URL
    assert captured["json"]["grant_type"] == "refresh_token"
    assert captured["json"]["refresh_token"] == "sk-ant-ort01-old"
    assert captured["json"]["client_id"] == cl.OAUTH_CLIENT_ID
