"""Tests for StreamEvent and vision capability detection."""
import pytest
from synapse.ai.clients.base import (
    StreamEvent, LLMClient, is_vision_model, VISION_MODELS,
)


def test_stream_event_text_kind():
    ev = StreamEvent(kind="text", text="hello")
    assert ev.kind == "text"
    assert ev.text == "hello"
    assert ev.tool_call is None
    assert ev.error is None


def test_stream_event_done_kind():
    ev = StreamEvent(kind="done")
    assert ev.kind == "done"
    assert ev.text is None


def test_stream_event_error_kind():
    ev = StreamEvent(kind="error", error="boom")
    assert ev.kind == "error"
    assert ev.error == "boom"


def test_stream_event_tool_call_kind():
    tc = {"name": "inspect_canvas", "input": {"node_ids": ["n1"]}}
    ev = StreamEvent(kind="tool_call", tool_call=tc)
    assert ev.kind == "tool_call"
    assert ev.tool_call == tc
    assert ev.text is None
    assert ev.error is None


def test_stream_event_rejects_unknown_kind():
    with pytest.raises(ValueError):
        StreamEvent(kind="not_a_kind")


def test_is_vision_model_known_vision():
    assert is_vision_model("claude-sonnet-4-20250514") is True
    assert is_vision_model("gpt-4o") is True
    assert is_vision_model("gemini-2.5-flash") is True
    assert is_vision_model("llava:13b") is True


def test_is_vision_model_known_text_only():
    assert is_vision_model("llama-3.3-70b-versatile") is False
    assert is_vision_model("gemma3:2b") is False


def test_is_vision_model_unknown_defaults_false():
    assert is_vision_model("some-future-model-xyz") is False


def test_llm_client_abstract():
    with pytest.raises(TypeError):
        LLMClient()  # abstract methods not implemented
