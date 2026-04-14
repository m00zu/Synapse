"""Streaming tests for OllamaClient using mocked HTTP."""
from unittest.mock import patch, MagicMock
from synapse.ai.clients.ollama import OllamaClient
from synapse.ai.clients.base import StreamEvent


def _fake_ndjson_response(chunks: list[str], done: bool = True):
    """Build a MagicMock resembling requests.Response with iter_lines()."""
    lines: list[bytes] = []
    for c in chunks:
        lines.append(f'{{"message":{{"content":"{c}"}},"done":false}}'.encode())
    if done:
        lines.append(b'{"message":{"content":""},"done":true}')
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    # Configure context manager to return self (for 'with' statement)
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
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
