"""Streaming tests for OpenRouterClient — mirrors OpenAI (same wire protocol)."""
from unittest.mock import patch, MagicMock

from synapse.ai.clients.openrouter import OpenRouterClient


def _fake_sse_response(contents: list[str]):
    lines: list[bytes] = []
    for c in contents:
        lines.append(
            f'data: {{"choices":[{{"delta":{{"content":"{c}"}}}}]}}'.encode()
        )
    lines.append(b'data: [DONE]')
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_openrouter_stream_concatenates_deltas():
    client = OpenRouterClient(api_key="or-test", model="openrouter/free")
    with patch("synapse.ai.clients.openrouter.requests.post",
               return_value=_fake_sse_response(["Hel", "lo"])) as pm:
        events = list(client.chat_with_tools_stream(
            system="s", messages=[{"role": "user", "content": "hi"}]))
    assert [e.kind for e in events] == ["text", "text", "done"]
    assert "".join(e.text for e in events if e.kind == "text") == "Hello"
    args, kwargs = pm.call_args
    assert kwargs["json"]["stream"] is True
    # Attribution headers travel with every request.
    headers = kwargs["headers"]
    assert "Authorization" in headers
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers


def test_openrouter_stream_error_emits_error_event():
    client = OpenRouterClient(api_key="or-test")
    with patch("synapse.ai.clients.openrouter.requests.post",
               side_effect=RuntimeError("timeout")):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert events[-1].kind == "error"
    assert "timeout" in events[-1].error


def test_openrouter_stream_emits_tool_call_event_when_tools_enabled():
    """OpenRouter uses OpenAI-compatible tool-calls (same delta.tool_calls shape)."""
    from synapse.ai.tools import TOOLS
    client = OpenRouterClient(api_key="or-test")
    lines = [
        b'data: {"choices":[{"delta":{"tool_calls":['
        b'{"index":0,"id":"call_1","type":"function",'
        b'"function":{"name":"inspect_canvas","arguments":""}}]}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":['
        b'{"index":0,"function":{"arguments":"{\\"node_ids\\":"}}]}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":['
        b'{"index":0,"function":{"arguments":" [\\"a\\"]}"}}]}}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        b'data: [DONE]',
    ]
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with patch("synapse.ai.clients.openrouter.requests.post", return_value=resp) as pm:
        events = list(client.chat_with_tools_stream(
            system="s", messages=[], tools=TOOLS,
        ))
    _, kwargs = pm.call_args
    assert "tools" in kwargs["json"]
    assert kwargs["json"]["tools"][0]["type"] == "function"
    tcs = [e for e in events if e.kind == "tool_call"]
    assert len(tcs) == 1
    assert tcs[0].tool_call["id"] == "call_1"
    assert tcs[0].tool_call["name"] == "inspect_canvas"
    assert tcs[0].tool_call["input"] == {"node_ids": ["a"]}


def test_openrouter_retries_on_429_then_succeeds():
    """Free-tier models often 429 during back-to-back calls. Retry should
    smooth over transient 429s with backoff."""
    import synapse.ai.clients.openrouter as mod
    # Patch the delay tuple so the test doesn't actually sleep.
    with patch.object(mod, "_RETRY_DELAYS", (0.0, 0.0)):
        client = OpenRouterClient(api_key="or-test")
        fail = MagicMock()
        fail.status_code = 429
        fail.headers = {}
        fail.close.return_value = None
        fail.raise_for_status.side_effect = Exception("should not be called on 429")

        success = _fake_sse_response(["ok"])
        success.status_code = 200
        success.headers = {}

        with patch("synapse.ai.clients.openrouter.requests.post",
                   side_effect=[fail, success]) as pm:
            events = list(client.chat_with_tools_stream(system="s", messages=[]))

        assert pm.call_count == 2
        assert any(e.kind == "text" and e.text == "ok" for e in events)


def test_openrouter_gives_up_after_final_retry():
    """After exhausting retries, surface the 429 as an error event."""
    import synapse.ai.clients.openrouter as mod
    with patch.object(mod, "_RETRY_DELAYS", (0.0, 0.0)):
        client = OpenRouterClient(api_key="or-test")
        fail = MagicMock()
        fail.status_code = 429
        fail.headers = {}
        fail.close.return_value = None
        fail.__enter__.return_value = fail
        fail.__exit__.return_value = False
        fail.raise_for_status.side_effect = Exception("429 Too Many Requests")
        fail.iter_lines.return_value = iter([])

        with patch("synapse.ai.clients.openrouter.requests.post",
                   return_value=fail):
            events = list(client.chat_with_tools_stream(system="s", messages=[]))

        assert events[-1].kind == "error"
        assert "429" in events[-1].error


def test_openrouter_lists_models_places_free_tier_first():
    client = OpenRouterClient()
    fake = MagicMock()
    fake.json.return_value = {"data": [
        {"id": "openai/gpt-4o"},
        {"id": "meta-llama/llama-3.3-70b-instruct:free"},
        {"id": "anthropic/claude-sonnet-4"},
        {"id": "google/gemma-3-27b-it:free"},
    ]}
    fake.raise_for_status.return_value = None
    with patch("synapse.ai.clients.openrouter.requests.get", return_value=fake):
        models = client.list_models()
    assert models[:2] == [
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ]
    assert "openai/gpt-4o" in models
