"""Regression tests for the \\uXXXX escape poison loop.

A tool-call argument (or any string) can enter the conversation as *literal*
backslash-u characters — ``"description": "T\\u00ecm"`` (six characters) instead
of the real ``Tìm``. That state is a stable fixed point: json round-trips
preserve it, and the model, seeing its own history rendered as ``\\uXXXX``,
reproduces the pattern in new tool calls and in prose (writing ``C\\u00e1c``
while still emitting emoji raw). The provider must decode these stray escapes at
the request boundary so the loop cannot sustain — while leaving ASCII escapes and
genuine code/regex snippets untouched.
"""

from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deerflow.models.claude_provider import (
    ClaudeChatModel,
    _decode_stray_unicode_escapes,
    _repair_obj,
    _repair_text,
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


# ---------------------------------------------------------------------------
# _repair_text — the core string transform
# ---------------------------------------------------------------------------


def test_decodes_vietnamese_diacritics():
    assert _repair_text("T\\u00ecm \\u0111\\u1ecbnh ngh\\u0129a h\\u00e0m") == "Tìm định nghĩa hàm"


def test_decodes_uppercase_hex_and_astral_escape():
    assert _repair_text("\\u00C0") == "À"
    # Astral (emoji) resolves above the threshold and decodes too.
    assert _repair_text("hi \\U0001F600") == "hi 😀"


def test_decodes_utf16_surrogate_pair():
    assert _repair_text("\\ud83d\\ude00") == "😀"


def test_leaves_ascii_escapes_untouched():
    # A == 'A' (< U+00A0): a legitimate code/regex literal, not leaked text.
    assert _repair_text("regex: \\u0041 matches A") == "regex: \\u0041 matches A"


def test_leaves_c1_control_and_lone_surrogate_untouched():
    assert _repair_text("esc \\u001b here") == "esc \\u001b here"
    assert _repair_text("\\ud83d alone") == "\\ud83d alone"


def test_no_escape_is_identity():
    s = "plain text with 🔀 emoji and Tiếng Việt"
    assert _repair_text(s) is s


def test_repair_obj_recurses_dict_and_list():
    obj = {"a": ["T\\u00ecm", {"b": "\\u0111\\u1ecbnh"}], "n": 3, "keep": "\\u0041"}
    assert _repair_obj(obj) == {"a": ["Tìm", {"b": "định"}], "n": 3, "keep": "\\u0041"}


# ---------------------------------------------------------------------------
# _decode_stray_unicode_escapes — message-level pass
# ---------------------------------------------------------------------------


def test_repairs_tool_call_args():
    msg = AIMessage(
        content="",
        tool_calls=[{"id": "t1", "name": "grep", "args": {"description": "T\\u00ecm h\\u00e0m", "pattern": "^def "}}],
    )
    out, count = _decode_stray_unicode_escapes([msg])
    assert count == 1
    assert out[0].tool_calls[0]["args"] == {"description": "Tìm hàm", "pattern": "^def "}


def test_repairs_string_and_list_content():
    str_msg = AIMessage(content="C\\u00e1c CSV th\\u1ecb tr\\u01b0\\u1eddng")
    list_msg = AIMessage(content=[{"type": "text", "text": "ch\\u1 ecked"}, {"type": "text", "text": "th\\u1ecb"}])
    out, count = _decode_stray_unicode_escapes([str_msg, list_msg])
    assert count == 2
    assert out[0].content == "Các CSV thị trường"
    assert out[1].content[1]["text"] == "thị"


def test_unpoisoned_messages_pass_through_unchanged():
    msgs = [
        SystemMessage(content="You are DeerFlow."),
        HumanMessage(content="fetch VNINDEX.csv"),
        AIMessage(content="done", tool_calls=[{"id": "t1", "name": "ls", "args": {"path": "/mnt/user-data"}}]),
    ]
    out, count = _decode_stray_unicode_escapes(msgs)
    assert count == 0
    # Same objects returned (no needless copies).
    assert all(a is b for a, b in zip(out, msgs))


# ---------------------------------------------------------------------------
# End-to-end: the poisoned request payload
# ---------------------------------------------------------------------------


def test_get_request_payload_decodes_poisoned_tool_args(model):
    msgs = [
        SystemMessage(content="You are DeerFlow."),
        HumanMessage(content="tìm hàm", name="user-input"),
        AIMessage(
            content="",
            tool_calls=[{"id": "t1", "name": "grep", "args": {"description": "T\\u00ecm \\u0111\\u1ecbnh ngh\\u0129a h\\u00e0m"}}],
        ),
        # A tool-call turn is always followed by its result; this also keeps the
        # OAuth request from ending on an assistant prefill.
        ToolMessage(content="1 match", tool_call_id="t1"),
    ]
    payload = model._get_request_payload(msgs)
    serialized = repr(payload["messages"])
    assert "Tìm định nghĩa hàm" in serialized
    assert "\\u00ec" not in serialized


def test_get_request_payload_preserves_ascii_escape_in_code(model):
    msgs = [
        SystemMessage(content="You are DeerFlow."),
        HumanMessage(content="explain \\u0041", name="user-input"),
    ]
    payload = model._get_request_payload(msgs)
    assert "\\u0041" in repr(payload["messages"])
