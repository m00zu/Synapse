from unittest.mock import patch, MagicMock
from synapse.ai.clients.gemini import GeminiClient


def _fake_sse(pieces: list[str]):
    lines: list[bytes] = []
    for p in pieces:
        lines.append(
            f'data: {{"candidates":[{{"content":{{"parts":[{{"text":"{p}"}}]}}}}]}}'.encode()
        )
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_gemini_stream_concatenates_parts():
    client = GeminiClient(api_key="gk-test")
    with patch("synapse.ai.clients.gemini.requests.post",
               return_value=_fake_sse(["He", "llo"])) as pm:
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert "".join(e.text for e in events if e.kind == "text") == "Hello"
    assert events[-1].kind == "done"
    args, kwargs = pm.call_args
    assert ":streamGenerateContent" in args[0]


def test_gemini_stream_handles_empty_candidates():
    client = GeminiClient(api_key="gk-test")
    resp = MagicMock()
    resp.iter_lines.return_value = iter([
        b'data: {"candidates":[]}',
        b'data: {"candidates":[{"content":{"parts":[{"text":"X"}]}}]}',
    ])
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with patch("synapse.ai.clients.gemini.requests.post", return_value=resp):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert any(e.kind == "text" and e.text == "X" for e in events)
    assert events[-1].kind == "done"


def test_gemini_stream_error_event():
    client = GeminiClient(api_key="gk-test")
    with patch("synapse.ai.clients.gemini.requests.post",
               side_effect=RuntimeError("quota exceeded")):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert events[-1].kind == "error"
    assert "quota exceeded" in events[-1].error
