"""Regression tests for the assistant-prefill poison loop.

A failed turn persists a DeerFlow error-fallback ``AIMessage`` at the tail of the
conversation. On the next turn that trailing assistant message is replayed to the
model, which Claude Code OAuth rejects with "this model does not support
assistant message prefill — the conversation must end with a user message". The
rejection produces another fallback, which re-poisons the thread permanently.

The provider must drop error-fallback messages and never end an OAuth request on
an assistant message.
"""

from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deerflow.models.claude_provider import (
    ClaudeChatModel,
    _strip_error_fallback_messages,
    _strip_trailing_assistant_messages,
)


def _make_model(*, oauth: bool = True) -> ClaudeChatModel:
    with mock.patch.object(ClaudeChatModel, "model_post_init"):
        m = ClaudeChatModel(model="claude-sonnet-4-6", anthropic_api_key="sk-ant-oat-fake-token")  # type: ignore[call-arg]
    m._is_oauth = oauth
    m._oauth_access_token = "sk-ant-oat-fake-token"
    return m


@pytest.fixture()
def model() -> ClaudeChatModel:
    return _make_model()


def _fallback(text: str) -> AIMessage:
    return AIMessage(content=text, additional_kwargs={"deerflow_error_fallback": True, "error_reason": "transient"})


# ---------------------------------------------------------------------------
# _strip_error_fallback_messages
# ---------------------------------------------------------------------------


def test_strips_error_fallback_messages_anywhere():
    msgs = [
        SystemMessage(content="prompt"),
        HumanMessage(content="hi"),
        _fallback("temporarily unavailable"),
        ToolMessage(content="OK", tool_call_id="t1"),
        _fallback("LLM request failed"),
    ]
    out, removed = _strip_error_fallback_messages(msgs)
    assert removed == 2
    assert [m.type for m in out] == ["system", "human", "tool"]


def test_keeps_normal_assistant_messages():
    msgs = [HumanMessage(content="hi"), AIMessage(content="real answer")]
    out, removed = _strip_error_fallback_messages(msgs)
    assert removed == 0
    assert out == msgs


def test_does_not_empty_list_when_all_fallbacks():
    msgs = [_fallback("a"), _fallback("b")]
    out, removed = _strip_error_fallback_messages(msgs)
    assert removed == 0
    assert out == msgs


# ---------------------------------------------------------------------------
# _strip_trailing_assistant_messages
# ---------------------------------------------------------------------------


def test_strips_trailing_assistant_messages():
    msgs = [
        HumanMessage(content="hi"),
        ToolMessage(content="OK", tool_call_id="t1"),
        AIMessage(content="trailing 1"),
        AIMessage(content="trailing 2"),
    ]
    out, removed = _strip_trailing_assistant_messages(msgs)
    assert removed == 2
    assert [m.type for m in out] == ["human", "tool"]


def test_keeps_request_ending_in_user_or_tool():
    msgs = [HumanMessage(content="hi"), ToolMessage(content="OK", tool_call_id="t1")]
    out, removed = _strip_trailing_assistant_messages(msgs)
    assert removed == 0
    assert out == msgs


def test_never_trims_below_one_message():
    msgs = [AIMessage(content="only")]
    out, removed = _strip_trailing_assistant_messages(msgs)
    assert removed == 0
    assert len(out) == 1


# ---------------------------------------------------------------------------
# End-to-end payload: the poisoned-thread scenario
# ---------------------------------------------------------------------------


def test_get_request_payload_recovers_poisoned_thread(model):
    """Tail = [..., ToolMessage, fallback, fallback, fallback] must produce a
    request that ends on the tool result (user role), not an assistant prefill."""
    msgs = [
        SystemMessage(content="You are DeerFlow."),
        HumanMessage(content="build the auth module", name="user-input"),
        AIMessage(content="", additional_kwargs={}, tool_calls=[{"id": "t1", "name": "write_todos", "args": {}}]),
        ToolMessage(content="Updated todo list", tool_call_id="t1"),
        _fallback("The configured LLM provider is temporarily unavailable after multiple retries."),
        _fallback("LLM request failed: Error code: 400 - assistant message prefill"),
        _fallback("LLM request failed: Error code: 400 - assistant message prefill"),
    ]

    payload = model._get_request_payload(msgs)

    roles = [m["role"] for m in payload["messages"]]
    assert roles[-1] == "user", f"request must end with a user turn, got {roles}"
    # None of the error-fallback text leaks back to the model.
    serialized = repr(payload["messages"])
    assert "temporarily unavailable" not in serialized
    assert "assistant message prefill" not in serialized


def test_get_request_payload_keeps_normal_conversation(model):
    msgs = [
        SystemMessage(content="You are DeerFlow."),
        HumanMessage(content="2+2?", name="user-input"),
    ]
    payload = model._get_request_payload(msgs)
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["user"]
