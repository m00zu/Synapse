"""Phase-1 non-regression checks — no network required."""
from synapse.llm_assistant import (
    OllamaClient, OpenAIClient, ClaudeClient,
    GeminiClient, GroqClient, LlamaCppClient, RunPodClient,
)
from synapse.ai.clients import (
    OllamaClient as OllamaNew, LLMClient, StreamEvent,
)


def test_reexported_classes_are_same():
    # Re-exports from llm_assistant resolve to the new-home classes.
    assert OllamaClient is OllamaNew


def test_all_streaming_clients_are_subclasses_of_base():
    for cls in (OllamaClient, OpenAIClient, ClaudeClient,
                GeminiClient, GroqClient, LlamaCppClient, RunPodClient):
        assert issubclass(cls, LLMClient), f"{cls.__name__} is not LLMClient"


def test_each_client_has_chat_with_tools_stream():
    for cls in (OllamaClient, OpenAIClient, ClaudeClient,
                GeminiClient, GroqClient, LlamaCppClient, RunPodClient):
        assert callable(getattr(cls, "chat_with_tools_stream", None)), \
            f"{cls.__name__} missing chat_with_tools_stream"


def test_legacy_chat_signature_preserved():
    # The existing app.py / AIChatPanel code calls .chat(system, user).
    # That signature MUST still work on the five streaming providers.
    import inspect
    for cls in (OllamaClient, OpenAIClient, ClaudeClient, GeminiClient, GroqClient):
        sig = inspect.signature(cls.chat)
        params = list(sig.parameters)
        assert "system" in params and "user" in params, \
            f"{cls.__name__}.chat lost its legacy signature"


def test_vision_flag_differs_per_model():
    assert ClaudeClient(api_key="x", model="claude-sonnet-4-20250514").supports_vision is True
    assert GroqClient(api_key="x", model="llama-3.3-70b-versatile").supports_vision is False


def test_llamacpp_stream_stub_yields_error_only():
    client = LlamaCppClient("/nonexistent/path.gguf")
    events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert len(events) == 1
    assert events[0].kind == "error"
    assert "Phase 1" in events[0].error


def test_runpod_stream_stub_yields_error_only():
    client = RunPodClient(api_key="x", endpoint_id="y")
    events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert len(events) == 1
    assert events[0].kind == "error"
    assert "Phase 1" in events[0].error
