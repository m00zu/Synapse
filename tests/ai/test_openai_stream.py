from unittest.mock import patch, MagicMock
from synapse.ai.clients.openai import OpenAIClient


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


def test_openai_stream_concatenates_deltas():
    client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
    with patch("synapse.ai.clients.openai.requests.post",
               return_value=_fake_sse_response(["Hel", "lo"])) as pm:
        events = list(client.chat_with_tools_stream(
            system="s", messages=[{"role": "user", "content": "hi"}]))
    assert [e.kind for e in events] == ["text", "text", "done"]
    assert "".join(e.text for e in events if e.kind == "text") == "Hello"
    _, kwargs = pm.call_args
    assert kwargs["json"]["stream"] is True


def test_openai_stream_ignores_blank_and_keepalive_lines():
    client = OpenAIClient(api_key="sk-test")
    resp = MagicMock()
    resp.iter_lines.return_value = iter([b"", b": keepalive",
        b'data: {"choices":[{"delta":{"content":"X"}}]}',
        b'data: [DONE]'])
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with patch("synapse.ai.clients.openai.requests.post", return_value=resp):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert [e.kind for e in events] == ["text", "done"]
    assert events[0].text == "X"


def test_openai_stream_error_emits_error_event():
    client = OpenAIClient(api_key="sk-test")
    with patch("synapse.ai.clients.openai.requests.post",
               side_effect=RuntimeError("timeout")):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert events[-1].kind == "error"
    assert "timeout" in events[-1].error
