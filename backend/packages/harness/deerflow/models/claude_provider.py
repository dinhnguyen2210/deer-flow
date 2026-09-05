"""Custom Claude provider with OAuth Bearer auth, prompt caching, and smart thinking.

Supports two authentication modes:
  1. Standard API key (x-api-key header) — default ChatAnthropic behavior
  2. Claude Code OAuth token (Authorization: Bearer header)
     - Detected by sk-ant-oat prefix
     - Requires anthropic-beta: oauth-2025-04-20,claude-code-20250219
     - Requires billing header in system prompt for all OAuth requests

Auto-loads credentials from explicit runtime handoff:
  - $ANTHROPIC_API_KEY environment variable
  - $CLAUDE_CODE_OAUTH_TOKEN or $ANTHROPIC_AUTH_TOKEN
  - $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR
  - $CLAUDE_CODE_CREDENTIALS_PATH
  - ~/.claude/.credentials.json
"""

import hashlib
import json
import logging
import os
import re
import socket
import time
import uuid
from typing import Any

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
THINKING_BUDGET_RATIO = 0.8


def _ensure_consecutive_system_messages(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    """Demote any ``SystemMessage`` outside the leading block to a ``HumanMessage``.

    The Anthropic API only accepts system content as a single top-level block, so
    ``langchain_anthropic`` raises "Received multiple non-consecutive system
    messages" if a ``SystemMessage`` appears after any non-system message. DeerFlow
    can produce that layout from several independent sources — a date reminder
    pushed down by a summary message, a checkpoint poisoned before the midnight
    fix, or any future middleware — so this provider-level guard makes every
    request Anthropic-safe regardless of upstream ordering.

    The leading consecutive run of system messages (the real system prompt, the
    OAuth billing block, the date reminder when it stays at the front) is kept as
    system. Any later system message is converted in place to a hidden
    ``HumanMessage`` (content/id/name/kwargs preserved). Returns the rewritten list
    and the number of demotions.
    """
    result: list[BaseMessage] = []
    in_leading_block = True
    demoted = 0
    for message in messages:
        if isinstance(message, SystemMessage):
            if in_leading_block:
                result.append(message)
                continue
            kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
            kwargs.setdefault("hide_from_ui", True)
            result.append(
                HumanMessage(
                    content=message.content,
                    id=getattr(message, "id", None),
                    name=getattr(message, "name", None),
                    additional_kwargs=kwargs,
                )
            )
            demoted += 1
        else:
            in_leading_block = False
            result.append(message)
    return result, demoted


# Matches a UTF-16 surrogate pair, a BMP ``\uXXXX``, or an astral ``\U00XXXXXX``
# escape written as *literal* backslash characters (not a real control char).
_STRAY_ESCAPE_RE = re.compile(
    r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})"  # UTF-16 surrogate pair
    r"|\\U([0-9a-fA-F]{8})"  # astral \U00XXXXXX (uppercase U, must precede \u)
    r"|\\u([0-9a-fA-F]{4})",  # BMP \uXXXX
)

# Only decode escapes that resolve to a real non-ASCII character (>= U+00A0).
# This repairs Vietnamese/CJK text while leaving ASCII escapes (``A``), C0/C1
# control escapes, and code/regex snippets that legitimately contain ``\uXXXX``
# untouched.
_MIN_DECODABLE_CODEPOINT = 0xA0


def _decode_stray_escape(match: "re.Match[str]") -> str:
    """Replace one matched escape with its character, or leave it verbatim.

    Verbatim (unchanged) when: the code point is ASCII / C1-control (< U+00A0), a
    lone surrogate, or otherwise not a valid character. This keeps the transform
    conservative — it only ever *repairs* leaked human-language text.
    """
    hi, lo, astral, bmp = match.groups()
    try:
        if hi and lo:
            code = 0x10000 + ((int(hi, 16) - 0xD800) << 10) + (int(lo, 16) - 0xDC00)
        elif bmp is not None:
            code = int(bmp, 16)
        else:
            code = int(astral, 16)
    except (TypeError, ValueError):
        return match.group(0)
    if code < _MIN_DECODABLE_CODEPOINT or code > 0x10FFFF or 0xD800 <= code <= 0xDFFF:
        return match.group(0)
    try:
        return chr(code)
    except (ValueError, OverflowError):
        return match.group(0)


def _repair_text(value: str) -> str:
    """Decode stray ``\\uXXXX`` sequences in ``value`` (fast no-op when absent)."""
    if "\\u" not in value and "\\U" not in value:
        return value
    return _STRAY_ESCAPE_RE.sub(_decode_stray_escape, value)


def _repair_obj(obj: Any) -> Any:
    """Recursively repair every string inside ``obj`` (dict/list/str)."""
    if isinstance(obj, str):
        return _repair_text(obj)
    if isinstance(obj, dict):
        return {key: _repair_obj(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_repair_obj(item) for item in obj]
    return obj


def _decode_stray_unicode_escapes(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    """Repair human-language text that leaked into history as literal ``\\uXXXX``.

    Once any string value (typically a tool-call argument like ``"description":
    "T\\u00ecm"``) enters the conversation as *literal* backslash-u characters
    instead of the real character (``Tìm``), it is a stable fixed point: every
    ``json.dumps``/``json.loads`` round-trip preserves it verbatim, and the model —
    seeing its own history rendered as ``\\uXXXX`` — reproduces the pattern in new
    tool calls *and* in prose (writing ``C\\u00e1c`` while still emitting emoji
    raw). The result is a self-reinforcing poison loop, in the same family as the
    assistant-prefill loop guarded by :func:`_strip_error_fallback_messages`.

    We break the loop at the request boundary: decode stray escapes in every
    outgoing message's content and tool-call args so the model stops seeing (and
    reproducing) them. The decode is conservative — only escapes resolving to a
    real non-ASCII character (>= U+00A0) are touched, leaving ASCII escapes and
    genuine code/regex snippets alone (see :func:`_decode_stray_escape`).

    Returns the rewritten list and the number of messages that were repaired.
    """
    repaired: list[BaseMessage] = []
    repaired_count = 0
    for message in messages:
        updates: dict[str, Any] = {}

        content = getattr(message, "content", None)
        if isinstance(content, (str, list)):
            new_content = _repair_obj(content)
            if new_content != content:
                updates["content"] = new_content

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            new_tool_calls = _repair_obj(tool_calls)
            if new_tool_calls != tool_calls:
                updates["tool_calls"] = new_tool_calls

        if updates:
            repaired.append(message.model_copy(update=updates))
            repaired_count += 1
        else:
            repaired.append(message)
    return repaired, repaired_count


def _strip_error_fallback_messages(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    """Drop DeerFlow error-fallback messages before they reach the model.

    ``LLMErrorHandlingMiddleware`` synthesizes ``AIMessage``s tagged with
    ``additional_kwargs["deerflow_error_fallback"]`` (e.g. "The configured LLM
    provider is temporarily unavailable…") purely to show the user a graceful
    notice. They are persisted in the checkpoint, so on the next turn they would
    be replayed to the model as if they were genuine assistant output — and when
    one is the last message, it forms an illegal assistant *prefill* that
    Anthropic OAuth rejects with "the conversation must end with a user message",
    which itself produces another fallback and poisons the thread permanently.

    These messages are never real model input, so we strip them on every request.
    Returns the filtered list and the number removed. If filtering would empty
    the list, the original is returned unchanged (defensive; never happens in
    practice because a system prompt + user turn always precede a fallback).
    """
    filtered = [m for m in messages if not (isinstance(getattr(m, "additional_kwargs", None), dict) and m.additional_kwargs.get("deerflow_error_fallback"))]
    removed = len(messages) - len(filtered)
    if removed and not filtered:
        return messages, 0
    return filtered, removed


def _strip_trailing_assistant_messages(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    """Ensure the request does not end on an assistant message.

    Claude Code OAuth tokens do not support assistant message prefill: the
    conversation must end with a user (or tool-result) turn. DeerFlow's agent
    loop never intends to prefill, so a trailing ``AIMessage`` is always an
    accident (a poisoned checkpoint, a mis-rolled-back regenerate, etc.). Drop
    trailing assistant messages so the payload ends on the preceding user/tool
    turn and the model generates a fresh response. Returns the trimmed list and
    the number removed; never trims below one message.
    """
    trimmed = list(messages)
    removed = 0
    while len(trimmed) > 1 and isinstance(trimmed[-1], AIMessage):
        trimmed.pop()
        removed += 1
    return trimmed, removed


# Billing header required by Anthropic API for OAuth token access.
# Must be the first system prompt block. Format mirrors Claude Code CLI.
# Override with ANTHROPIC_BILLING_HEADER env var if the hardcoded version drifts.
_DEFAULT_BILLING_HEADER = "x-anthropic-billing-header: cc_version=2.1.85.351; cc_entrypoint=cli; cch=6c6d5;"
OAUTH_BILLING_HEADER = os.environ.get("ANTHROPIC_BILLING_HEADER", _DEFAULT_BILLING_HEADER)


class ClaudeChatModel(ChatAnthropic):
    """ChatAnthropic with OAuth Bearer auth, prompt caching, and smart thinking.

    Config example:
        - name: claude-sonnet-4.6
          use: deerflow.models.claude_provider:ClaudeChatModel
          model: claude-sonnet-4-6
          max_tokens: 16384
          enable_prompt_caching: true
    """

    # Custom fields
    enable_prompt_caching: bool = True
    prompt_cache_size: int = 3
    auto_thinking_budget: bool = True
    retry_max_attempts: int = MAX_RETRIES
    _is_oauth: bool = PrivateAttr(default=False)
    _oauth_access_token: str = PrivateAttr(default="")

    model_config = {"arbitrary_types_allowed": True}

    def _validate_retry_config(self) -> None:
        if self.retry_max_attempts < 1:
            raise ValueError("retry_max_attempts must be >= 1")

    def model_post_init(self, __context: Any) -> None:
        """Auto-load credentials and configure OAuth if needed."""
        from pydantic import SecretStr

        from deerflow.models.credential_loader import (
            OAUTH_ANTHROPIC_BETAS,
            is_oauth_token,
            load_claude_code_credential,
        )

        self._validate_retry_config()

        # Extract actual key value (SecretStr.str() returns '**********')
        current_key = ""
        if self.anthropic_api_key:
            if hasattr(self.anthropic_api_key, "get_secret_value"):
                current_key = self.anthropic_api_key.get_secret_value()
            else:
                current_key = str(self.anthropic_api_key)

        # Try the explicit Claude Code OAuth handoff sources if no valid key.
        if not current_key or current_key in ("your-anthropic-api-key",):
            cred = load_claude_code_credential()
            if cred:
                current_key = cred.access_token
                logger.info(f"Using Claude Code CLI credential (source: {cred.source})")
            else:
                logger.warning("No Anthropic API key or explicit Claude Code OAuth credential found.")

        # Detect OAuth token and configure Bearer auth
        if is_oauth_token(current_key):
            self._is_oauth = True
            self._oauth_access_token = current_key
            # Set the token as api_key temporarily (will be swapped to auth_token on client)
            self.anthropic_api_key = SecretStr(current_key)
            # Add required beta headers for OAuth
            self.default_headers = {
                **(self.default_headers or {}),
                "anthropic-beta": OAUTH_ANTHROPIC_BETAS,
            }
            # OAuth tokens have a limit of 4 cache_control blocks — disable prompt caching
            self.enable_prompt_caching = False
            logger.info("OAuth token detected — will use Authorization: Bearer header")
        else:
            if current_key:
                self.anthropic_api_key = SecretStr(current_key)

        # Ensure api_key is SecretStr
        if isinstance(self.anthropic_api_key, str):
            self.anthropic_api_key = SecretStr(self.anthropic_api_key)

        super().model_post_init(__context)

        # Patch clients immediately after creation for OAuth Bearer auth.
        # This must happen after super() because clients are lazily created.
        if self._is_oauth:
            self._patch_client_oauth(self._client)
            self._patch_client_oauth(self._async_client)

    def _patch_client_oauth(self, client: Any) -> None:
        """Swap api_key → auth_token on an Anthropic SDK client for OAuth Bearer auth."""
        if hasattr(client, "api_key") and hasattr(client, "auth_token"):
            client.api_key = None
            client.auth_token = self._oauth_access_token

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Override to inject prompt caching, thinking budget, and OAuth billing."""
        messages = self._convert_input(input_).to_messages()

        # Drop DeerFlow error-fallback notices — they are UI-only messages that
        # must never be replayed as assistant turns. Persisted at the tail of a
        # failed turn, they otherwise form an illegal assistant prefill that
        # poisons the thread (see _strip_error_fallback_messages).
        messages, dropped_fallbacks = _strip_error_fallback_messages(messages)
        if dropped_fallbacks:
            logger.warning("ClaudeChatModel: stripped %d DeerFlow error-fallback message(s) before sending to the model", dropped_fallbacks)

        # OAuth tokens reject assistant prefill, so the request must not end on an
        # assistant message. Defensively trim any trailing AIMessage left by a
        # poisoned checkpoint or a mis-rolled-back regenerate.
        if self._is_oauth:
            messages, trimmed = _strip_trailing_assistant_messages(messages)
            if trimmed:
                logger.warning("ClaudeChatModel: trimmed %d trailing assistant message(s) to satisfy OAuth no-prefill constraint", trimmed)

        # Anthropic requires all system content to be consecutive at the front.
        # Normalize so any mid-conversation SystemMessage (summary push-down,
        # poisoned checkpoint, etc.) cannot raise "multiple non-consecutive system
        # messages" inside super()._get_request_payload -> _format_messages.
        messages, demoted = _ensure_consecutive_system_messages(messages)
        if demoted:
            logger.warning("ClaudeChatModel: demoted %d non-leading SystemMessage(s) to HumanMessage to keep system content consecutive", demoted)

        # Repair human-language text that leaked into history as literal \uXXXX
        # escapes (e.g. a tool-call arg "Tìm" instead of "Tìm"). Left alone,
        # the model mirrors the pattern into new tool calls and prose, forming a
        # self-reinforcing poison loop (see _decode_stray_unicode_escapes).
        messages, repaired = _decode_stray_unicode_escapes(messages)
        if repaired:
            logger.warning("ClaudeChatModel: decoded stray \\uXXXX escape sequences in %d message(s) to break the escape poison loop", repaired)

        payload = super()._get_request_payload(messages, stop=stop, **kwargs)

        if self._is_oauth:
            self._apply_oauth_billing(payload)

        if self.enable_prompt_caching:
            self._apply_prompt_caching(payload)

        if self.auto_thinking_budget:
            self._apply_thinking_budget(payload)

        return payload

    def _apply_oauth_billing(self, payload: dict) -> None:
        """Inject the billing header block required for all OAuth requests.

        The billing block is always placed first in the system list, removing any
        existing occurrence to avoid duplication or out-of-order positioning.
        """
        billing_block = {"type": "text", "text": OAUTH_BILLING_HEADER}

        system = payload.get("system")
        if isinstance(system, list):
            # Remove any existing billing blocks, then insert a single one at index 0.
            filtered = [b for b in system if not (isinstance(b, dict) and OAUTH_BILLING_HEADER in b.get("text", ""))]
            payload["system"] = [billing_block] + filtered
        elif isinstance(system, str):
            if OAUTH_BILLING_HEADER in system:
                payload["system"] = [billing_block]
            else:
                payload["system"] = [billing_block, {"type": "text", "text": system}]
        else:
            payload["system"] = [billing_block]

        # Add metadata.user_id required by the API for OAuth billing validation
        if not isinstance(payload.get("metadata"), dict):
            payload["metadata"] = {}
        if "user_id" not in payload["metadata"]:
            # Generate a stable device_id from the machine's hostname
            hostname = socket.gethostname()
            device_id = hashlib.sha256(f"deerflow-{hostname}".encode()).hexdigest()
            session_id = str(uuid.uuid4())
            payload["metadata"]["user_id"] = json.dumps(
                {
                    "device_id": device_id,
                    "account_uuid": "deerflow",
                    "session_id": session_id,
                }
            )

    def _apply_prompt_caching(self, payload: dict) -> None:
        """Apply ephemeral cache_control to system, recent messages, and last tool definition.

        Uses a budget of MAX_CACHE_BREAKPOINTS (4) breakpoints — the hard limit
        enforced by both the Anthropic API and AWS Bedrock.  Breakpoints are
        placed on the *last* eligible blocks because later breakpoints cover a
        larger prefix and yield better cache hit rates.

        The system prompt is expected to be fully static (no per-user memory or
        current date).  Dynamic context is injected per-turn via
        DynamicContextMiddleware as a <system-reminder> in the first HumanMessage.
        """
        MAX_CACHE_BREAKPOINTS = 4

        # Collect candidate blocks in document order:
        #   1. system text blocks
        #   2. content blocks of the last prompt_cache_size messages
        #   3. the last tool definition
        candidates: list[dict] = []

        # 1. System blocks
        system = payload.get("system")
        if system and isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    candidates.append(block)
        elif system and isinstance(system, str):
            new_block: dict = {"type": "text", "text": system}
            payload["system"] = [new_block]
            candidates.append(new_block)

        # 2. Recent message blocks
        messages = payload.get("messages", [])
        cache_start = max(0, len(messages) - self.prompt_cache_size)
        for i in range(cache_start, len(messages)):
            msg = messages[i]
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        candidates.append(block)
            elif isinstance(content, str) and content:
                new_block = {"type": "text", "text": content}
                msg["content"] = [new_block]
                candidates.append(new_block)

        # 3. Last tool definition
        tools = payload.get("tools", [])
        if tools and isinstance(tools[-1], dict):
            candidates.append(tools[-1])

        # Apply cache_control only to the last MAX_CACHE_BREAKPOINTS candidates
        # to stay within the API limit.
        for block in candidates[-MAX_CACHE_BREAKPOINTS:]:
            block["cache_control"] = {"type": "ephemeral"}

    def _apply_thinking_budget(self, payload: dict) -> None:
        """Auto-allocate thinking budget (80% of max_tokens)."""
        thinking = payload.get("thinking")
        if not thinking or not isinstance(thinking, dict):
            return
        if thinking.get("type") != "enabled":
            return
        if thinking.get("budget_tokens"):
            return

        max_tokens = payload.get("max_tokens", 8192)
        thinking["budget_tokens"] = int(max_tokens * THINKING_BUDGET_RATIO)

    @staticmethod
    def _strip_cache_control(payload: dict) -> None:
        """Remove cache_control markers before OAuth requests reach Anthropic."""
        for section in ("system", "messages"):
            items = payload.get(section)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                item.pop("cache_control", None)
                content = item.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            block.pop("cache_control", None)

        tools = payload.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    tool.pop("cache_control", None)

    def _create(self, payload: dict) -> Any:
        if self._is_oauth:
            self._strip_cache_control(payload)
        return super()._create(payload)

    async def _acreate(self, payload: dict) -> Any:
        if self._is_oauth:
            self._strip_cache_control(payload)
        return await super()._acreate(payload)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        """Override with OAuth patching and retry logic."""
        if self._is_oauth:
            self._patch_client_oauth(self._client)

        last_error = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return super()._generate(messages, stop=stop, **kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Rate limited, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                time.sleep(wait_ms / 1000)
            except anthropic.InternalServerError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Server error, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                time.sleep(wait_ms / 1000)
        raise last_error

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        """Async override with OAuth patching and retry logic."""
        import asyncio

        if self._is_oauth:
            self._patch_client_oauth(self._async_client)

        last_error = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return await super()._agenerate(messages, stop=stop, **kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Rate limited, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                await asyncio.sleep(wait_ms / 1000)
            except anthropic.InternalServerError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Server error, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                await asyncio.sleep(wait_ms / 1000)
        raise last_error

    @staticmethod
    def _calc_backoff_ms(attempt: int, error: Exception) -> int:
        """Exponential backoff with a fixed 20% buffer."""
        backoff_ms = 2000 * (1 << (attempt - 1))
        jitter_ms = int(backoff_ms * 0.2)
        total_ms = backoff_ms + jitter_ms

        if hasattr(error, "response") and error.response is not None:
            retry_after = error.response.headers.get("Retry-After")
            if retry_after:
                try:
                    total_ms = int(retry_after) * 1000
                except (ValueError, TypeError):
                    pass

        return total_ms
