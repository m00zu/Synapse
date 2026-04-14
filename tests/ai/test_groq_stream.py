from unittest.mock import patch, MagicMock
from synapse.ai.clients.groq import GroqClient


def _fake_sse(contents: list[str]):
    lines: list[bytes] = []
    for c in contents:
        lines.append(f'data: {{"choices":[{{"delta":{{"content":"{c}"}}}}]}}'.encode())
    lines.append(b'data: [DONE]')
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_groq_stream_concatenates_deltas():
    client = GroqClient(api_key="gsk-test")
    with patch("synapse.ai.clients.groq.requests.post",
               return_value=_fake_sse(["Foo", "Bar"])):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert "".join(e.text for e in events if e.kind == "text") == "FooBar"
    assert events[-1].kind == "done"


def test_groq_stream_error_event():
    client = GroqClient(api_key="gsk-test")
    with patch("synapse.ai.clients.groq.requests.post",
               side_effect=RuntimeError("rate limit")):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert events[-1].kind == "error"
    assert "rate limit" in events[-1].error
