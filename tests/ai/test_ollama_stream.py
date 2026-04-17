"""Streaming tests for OllamaClient using mocked HTTP."""
from unittest.mock import patch, MagicMock
from synapse.ai.clients.ollama import OllamaClient
from synapse.ai.clients.base import StreamEvent


def _fake_ndjson_response(chunks: list[str], done: bool = True,
                           status_code: int = 200):
    """Build a MagicMock resembling requests.Response with iter_lines().
    ``status_code`` defaults to 200; retry logic in OllamaClient checks this
    before returning the response."""
    lines: list[bytes] = []
    for c in chunks:
        lines.append(f'{{"message":{{"content":"{c}"}},"done":false}}'.encode())
    if done:
        lines.append(b'{"message":{"content":""},"done":true}')
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {}
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    # Configure context manager to return self (for 'with' statement)
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    resp.close.return_value = None
    return resp


def test_ollama_stream_yields_text_then_done():
    client = OllamaClient(model="gemma3:2b")
    with patch("synapse.ai.clients.ollama.requests.post",
               return_value=_fake_ndjson_response(["Hel", "lo", " world"])) as pm:
        events = list(client.chat_with_tools_stream(
            system="sys", messages=[{"role": "user", "content": "hi"}]))
    pm.assert_called_once()
    assert [e.kind for e in events] == ["text", "text", "text", "done"]
    assert "".join(e.text or "" for e in events if e.kind == "text") == "Hello world"


def test_ollama_stream_sends_stream_true():
    client = OllamaClient()
    with patch("synapse.ai.clients.ollama.requests.post",
               return_value=_fake_ndjson_response([])) as pm:
        list(client.chat_with_tools_stream(system="s", messages=[]))
    _, kwargs = pm.call_args
    assert kwargs["json"]["stream"] is True


def test_ollama_stream_error_emits_error_event():
    client = OllamaClient()
    def boom(*a, **k):
        raise ConnectionError("network down")
    with patch("synapse.ai.clients.ollama.requests.post", side_effect=boom):
        events = list(client.chat_with_tools_stream(
            system="s", messages=[{"role": "user", "content": "hi"}]))
    assert events[-1].kind == "error"
    assert "network down" in events[-1].error


def test_ollama_stream_sends_custom_user_agent_and_accept():
    """Ollama Cloud's WAF rejects default python-requests UA on /api/chat.
    Confirm we send a non-default User-Agent + Accept header."""
    client = OllamaClient(api_key="oc-test")
    with patch("synapse.ai.clients.ollama.requests.post",
               return_value=_fake_ndjson_response([])) as pm:
        list(client.chat_with_tools_stream(system="s", messages=[]))
    _, kwargs = pm.call_args
    headers = kwargs["headers"]
    assert "User-Agent" in headers
    assert "python-requests" not in headers["User-Agent"]
    assert headers.get("Accept") == "application/json"
    assert headers.get("Authorization") == "Bearer oc-test"


def test_ollama_stream_detects_tool_call_from_ndjson():
    from synapse.ai.tools import TOOLS
    client = OllamaClient(model="gemma3:2b")
    lines = [
        b'{"message":{"content":"<tool_call>{\\"name\\":"},"done":false}',
        b'{"message":{"content":"\\"inspect_canvas\\",\\"input\\":{}}</tool_call>"},"done":false}',
        b'{"message":{"content":""},"done":true}',
    ]
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with patch("synapse.ai.clients.ollama.requests.post", return_value=resp) as pm:
        events = list(client.chat_with_tools_stream(
            system="sys", messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
        ))
    # System prompt was augmented with the fallback protocol addendum.
    _, kwargs = pm.call_args
    sys_msg = kwargs["json"]["messages"][0]
    assert sys_msg["role"] == "system"
    assert "<tool_call>" in sys_msg["content"]
    # One tool_call event.
    tcs = [e for e in events if e.kind == "tool_call"]
    assert len(tcs) == 1
    assert tcs[0].tool_call["name"] == "inspect_canvas"
    assert tcs[0].tool_call["input"] == {}


def test_ollama_retries_on_500_then_succeeds():
    """Ollama Cloud's free-tier returns transient 500s. The client should
    back off, retry, and recover without surfacing an error event."""
    import synapse.ai.clients.ollama as mod
    # No-op the backoff delays so tests don't sleep.
    with patch.object(mod, "_RETRY_DELAYS", (0.0, 0.0)):
        client = OllamaClient(model="gemma3:2b")
        fail = _fake_ndjson_response([], status_code=500)
        good = _fake_ndjson_response(["hi"])
        with patch("synapse.ai.clients.ollama.requests.post",
                   side_effect=[fail, good]) as pm:
            events = list(client.chat_with_tools_stream(
                system="s", messages=[{"role": "user", "content": "hi"}]))
        assert pm.call_count == 2
        assert any(e.kind == "text" and e.text == "hi" for e in events)
        assert events[-1].kind == "done"


def test_ollama_gives_up_after_repeated_500s():
    """After exhausting retries, surface the 500 as an error event."""
    import synapse.ai.clients.ollama as mod
    with patch.object(mod, "_RETRY_DELAYS", (0.0, 0.0)):
        client = OllamaClient(model="gemma3:2b")
        fail = _fake_ndjson_response([], status_code=500)
        fail.raise_for_status.side_effect = Exception("500 Server Error")
        with patch("synapse.ai.clients.ollama.requests.post", return_value=fail):
            events = list(client.chat_with_tools_stream(
                system="s", messages=[{"role": "user", "content": "hi"}]))
        assert events[-1].kind == "error"
        assert "500" in events[-1].error
