from unittest.mock import patch, MagicMock
from synapse.ai.clients.claude import ClaudeClient


def _fake_anthropic_sse(pieces: list[str]):
    lines: list[bytes] = [b"event: message_start",
                          b'data: {"type":"message_start"}', b""]
    for p in pieces:
        lines.append(b"event: content_block_delta")
        lines.append(
            f'data: {{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{p}"}}}}'.encode()
        )
        lines.append(b"")
    lines.append(b"event: message_stop")
    lines.append(b'data: {"type":"message_stop"}')
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_claude_stream_concatenates_text_deltas():
    client = ClaudeClient(api_key="sk-ant-test")
    with patch("synapse.ai.clients.claude.requests.post",
               return_value=_fake_anthropic_sse(["Hi", " there"])):
        events = list(client.chat_with_tools_stream(
            system="s", messages=[{"role": "user", "content": "hi"}]))
    text_joined = "".join(e.text for e in events if e.kind == "text")
    assert text_joined == "Hi there"
    assert events[-1].kind == "done"


def test_claude_stream_ignores_non_text_events():
    client = ClaudeClient(api_key="sk-ant-test")
    resp = MagicMock()
    resp.iter_lines.return_value = iter([
        b"event: ping", b'data: {"type":"ping"}', b"",
        b"event: content_block_delta",
        b'data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{}"}}',
        b"event: message_stop", b'data: {"type":"message_stop"}',
    ])
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with patch("synapse.ai.clients.claude.requests.post", return_value=resp):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    text_events = [e for e in events if e.kind == "text"]
    assert text_events == []
    assert events[-1].kind == "done"


def test_claude_stream_error_emits_error_event():
    client = ClaudeClient(api_key="sk-ant-test")
    with patch("synapse.ai.clients.claude.requests.post",
               side_effect=RuntimeError("bad key")):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert events[-1].kind == "error"
    assert "bad key" in events[-1].error
