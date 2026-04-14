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


def test_claude_stream_emits_tool_call_event_when_tools_enabled():
    from synapse.ai.tools import TOOLS
    client = ClaudeClient(api_key="sk-ant-test")
    lines = [
        b"event: message_start",
        b'data: {"type":"message_start"}',
        b"",
        b"event: content_block_start",
        b'data: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"tool_use","id":"toolu_1","name":"inspect_canvas","input":{}}}',
        b"",
        b"event: content_block_delta",
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"input_json_delta","partial_json":"{\\"node_ids\\":"}}',
        b"",
        b"event: content_block_delta",
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"input_json_delta","partial_json":" [\\"a\\"]}"}}',
        b"",
        b"event: content_block_stop",
        b'data: {"type":"content_block_stop","index":0}',
        b"",
        b"event: message_stop",
        b'data: {"type":"message_stop"}',
    ]
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with patch("synapse.ai.clients.claude.requests.post", return_value=resp) as pm:
        events = list(client.chat_with_tools_stream(
            system="s", messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
        ))
    _, kwargs = pm.call_args
    payload = kwargs["json"]
    assert "tools" in payload
    assert any(t["name"] == "inspect_canvas" for t in payload["tools"])
    tcs = [e for e in events if e.kind == "tool_call"]
    assert len(tcs) == 1
    assert tcs[0].tool_call["name"] == "inspect_canvas"
    assert tcs[0].tool_call["id"] == "toolu_1"
    assert tcs[0].tool_call["input"] == {"node_ids": ["a"]}
    assert events[-1].kind == "done"
