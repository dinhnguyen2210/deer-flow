"""Auto-load credentials from Claude Code CLI and Codex CLI.

Implements two credential strategies:
  1. Claude Code OAuth token from explicit env vars or an exported credentials file
     - Uses Authorization: Bearer header (NOT x-api-key)
     - Requires anthropic-beta: oauth-2025-04-20,claude-code-20250219
     - Supports $CLAUDE_CODE_OAUTH_TOKEN, $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR, and $ANTHROPIC_AUTH_TOKEN
     - Override path with $CLAUDE_CODE_CREDENTIALS_PATH
     - When a file-backed access token is expired, the stored refresh token is
       exchanged for a fresh one and written back, so runs self-heal instead of
       failing with "Could not resolve authentication method"
  2. Codex CLI token from ~/.codex/auth.json
     - Uses chatgpt.com/backend-api/codex/responses endpoint
     - Supports both legacy top-level tokens and current nested tokens shape
     - Override path with $CODEX_AUTH_PATH
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Required beta headers for Claude Code OAuth tokens
OAUTH_ANTHROPIC_BETAS = "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14"

# Public Claude Code OAuth client used for the refresh-token grant. These are the
# same well-known values the Claude Code CLI itself uses to mint new access
# tokens, so DeerFlow can self-heal an expired token from the stored refresh
# token instead of failing the request with an auth error.
OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_REFRESH_TIMEOUT_S = 30.0


def is_oauth_token(token: str) -> bool:
    """Check if a token is a Claude Code OAuth token (not a standard API key)."""
    return isinstance(token, str) and "sk-ant-oat" in token


@dataclass
class ClaudeCodeCredential:
    """Claude Code CLI OAuth credential."""

    access_token: str
    refresh_token: str = ""
    expires_at: int = 0
    source: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() * 1000 > self.expires_at - 60_000  # 1 min buffer


@dataclass
class CodexCliCredential:
    """Codex CLI credential."""

    access_token: str
    account_id: str = ""
    source: str = ""


def _resolve_credential_path(env_var: str, default_relative_path: str) -> Path:
    configured_path = os.getenv(env_var)
    if configured_path:
        return Path(configured_path).expanduser()
    return _home_dir() / default_relative_path


def _home_dir() -> Path:
    home = os.getenv("HOME")
    if home:
        return Path(home).expanduser()
    return Path.home()


def _load_json_file(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        logger.debug(f"{label} not found: {path}")
        return None
    if path.is_dir():
        logger.warning(f"{label} path is a directory, expected a file: {path}")
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {label}: {e}")
        return None


def _read_secret_from_file_descriptor(env_var: str) -> str | None:
    fd_value = os.getenv(env_var)
    if not fd_value:
        return None

    try:
        fd = int(fd_value)
    except ValueError:
        logger.warning(f"{env_var} must be an integer file descriptor, got: {fd_value}")
        return None

    try:
        secret = os.read(fd, 1024 * 1024).decode().strip()
    except OSError as e:
        logger.warning(f"Failed to read {env_var}: {e}")
        return None

    return secret or None


def _credential_from_direct_token(access_token: str, source: str) -> ClaudeCodeCredential | None:
    token = access_token.strip()
    if not token:
        return None
    return ClaudeCodeCredential(access_token=token, source=source)


def _iter_claude_code_credential_paths() -> list[Path]:
    paths: list[Path] = []
    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    if override_path:
        paths.append(Path(override_path).expanduser())

    default_path = _home_dir() / ".claude/.credentials.json"
    if not paths or paths[-1] != default_path:
        paths.append(default_path)

    return paths


def _request_oauth_token(refresh_token: str) -> dict[str, Any] | None:
    """Exchange a Claude Code refresh token for a fresh access token.

    Returns the parsed token JSON (``access_token``/``refresh_token``/
    ``expires_in``) on success, or ``None`` if the request fails or is rejected.
    Kept as a thin seam so the network call is easy to stub in tests.
    """
    try:
        import httpx

        response = httpx.post(
            OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OAUTH_CLIENT_ID,
            },
            headers={"Content-Type": "application/json"},
            timeout=OAUTH_REFRESH_TIMEOUT_S,
        )
    except Exception as e:
        logger.warning(f"Claude Code OAuth refresh request failed: {e}")
        return None

    if response.status_code != 200:
        logger.warning(f"Claude Code OAuth refresh rejected (HTTP {response.status_code}). Run 'claude' to refresh.")
        return None

    try:
        return response.json()
    except (ValueError, json.JSONDecodeError):
        logger.warning("Claude Code OAuth refresh returned an unparseable response")
        return None


def _persist_credentials(path: Path, data: dict[str, Any]) -> None:
    """Atomically write refreshed credentials back to disk, preserving file mode."""
    tmp_path = path.parent / f"{path.name}.tmp"
    try:
        tmp_path.write_text(json.dumps(data, indent=2))
        mode: int | None = None
        try:
            mode = path.stat().st_mode
        except OSError:
            mode = None
        os.replace(tmp_path, path)
        if mode is not None:
            try:
                os.chmod(path, mode & 0o777)
            except OSError:
                pass
    except OSError as e:
        logger.warning(f"Failed to persist refreshed Claude Code credentials: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _refresh_claude_code_credential(data: dict[str, Any], oauth: dict[str, Any], path: Path, source: str) -> ClaudeCodeCredential | None:
    """Refresh an expired token via the stored refresh token and persist the result."""
    refresh_token = oauth.get("refreshToken", "")
    if not refresh_token:
        logger.warning("Claude Code OAuth token is expired and no refresh token is available. Run 'claude' to refresh.")
        return None

    token_json = _request_oauth_token(refresh_token)
    if not token_json:
        logger.warning("Claude Code OAuth token is expired and automatic refresh failed. Run 'claude' to refresh.")
        return None

    new_access = token_json.get("access_token", "")
    if not new_access:
        logger.warning("Claude Code OAuth refresh response did not contain an access token. Run 'claude' to refresh.")
        return None

    new_refresh = token_json.get("refresh_token") or refresh_token
    expires_in = token_json.get("expires_in")
    new_expires_at = int(time.time() * 1000 + float(expires_in) * 1000) if expires_in else 0

    new_oauth = dict(oauth)
    new_oauth["accessToken"] = new_access
    new_oauth["refreshToken"] = new_refresh
    if new_expires_at:
        new_oauth["expiresAt"] = new_expires_at
    new_data = dict(data)
    new_data["claudeAiOauth"] = new_oauth
    _persist_credentials(path, new_data)

    logger.info("Refreshed expired Claude Code OAuth token automatically")
    return ClaudeCodeCredential(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_at=new_expires_at,
        source=source,
    )


def _extract_claude_code_credential(data: dict[str, Any], source: str, path: Path | None = None) -> ClaudeCodeCredential | None:
    oauth = data.get("claudeAiOauth", {})
    access_token = oauth.get("accessToken", "")
    if not access_token:
        logger.debug("Claude Code credentials container exists but no accessToken found")
        return None

    cred = ClaudeCodeCredential(
        access_token=access_token,
        refresh_token=oauth.get("refreshToken", ""),
        expires_at=oauth.get("expiresAt", 0),
        source=source,
    )

    if cred.is_expired:
        # Self-heal from the stored refresh token (only possible for file-backed
        # credentials, where we can write the rotated token back).
        if path is not None:
            refreshed = _refresh_claude_code_credential(data, oauth, path, source)
            if refreshed:
                return refreshed
        else:
            logger.warning("Claude Code OAuth token is expired. Run 'claude' to refresh.")
        return None

    return cred


def load_claude_code_credential() -> ClaudeCodeCredential | None:
    """Load OAuth credential from explicit Claude Code handoff sources.

    Lookup order:
      1. $CLAUDE_CODE_OAUTH_TOKEN or $ANTHROPIC_AUTH_TOKEN
      2. $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR
      3. $CLAUDE_CODE_CREDENTIALS_PATH
      4. ~/.claude/.credentials.json

    Exported credentials files contain:
    {
      "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-...",
        "refreshToken": "sk-ant-ort01-...",
        "expiresAt": 1773430695128,
        "scopes": ["user:inference", ...],
        ...
      }
    }
    """
    direct_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    if direct_token:
        cred = _credential_from_direct_token(direct_token, "claude-cli-env")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from environment")
        return cred

    fd_token = _read_secret_from_file_descriptor("CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR")
    if fd_token:
        cred = _credential_from_direct_token(fd_token, "claude-cli-fd")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from file descriptor")
        return cred

    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    override_path_obj = Path(override_path).expanduser() if override_path else None
    for cred_path in _iter_claude_code_credential_paths():
        data = _load_json_file(cred_path, "Claude Code credentials")
        if data is None:
            continue
        cred = _extract_claude_code_credential(data, "claude-cli-file", path=cred_path)
        if cred:
            source_label = "override path" if override_path_obj is not None and cred_path == override_path_obj else "plaintext file"
            logger.info(f"Loaded Claude Code OAuth credential from {source_label} (expires_at={cred.expires_at})")
            return cred

    return None


def load_codex_cli_credential() -> CodexCliCredential | None:
    """Load credential from Codex CLI (~/.codex/auth.json)."""
    cred_path = _resolve_credential_path("CODEX_AUTH_PATH", ".codex/auth.json")
    data = _load_json_file(cred_path, "Codex CLI credentials")
    if data is None:
        return None
    tokens = data.get("tokens", {})
    if not isinstance(tokens, dict):
        tokens = {}

    access_token = data.get("access_token") or data.get("token") or tokens.get("access_token", "")
    account_id = data.get("account_id") or tokens.get("account_id", "")
    if not access_token:
        logger.debug("Codex CLI credentials file exists but no token found")
        return None

    logger.info("Loaded Codex CLI credential")
    return CodexCliCredential(
        access_token=access_token,
        account_id=account_id,
        source="codex-cli",
    )
