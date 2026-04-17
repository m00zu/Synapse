from synapse.ai.clients._fallback_parser import StreamedToolCallParser


def test_passthrough_text():
    p = StreamedToolCallParser()
    assert p.feed("hello ") == [("text", "hello ")]
    assert p.feed("world.") == [("text", "world.")]
    assert p.finish() == []


def test_detects_complete_tool_call():
    p = StreamedToolCallParser()
    events = p.feed('<tool_call>{"name": "inspect_canvas", "input": {}}</tool_call>')
    kinds = [e[0] for e in events]
    assert kinds == ["tool_call"]
    tc = events[0][1]
    assert tc == {"name": "inspect_canvas", "input": {}}


def test_detects_tool_call_split_across_chunks():
    p = StreamedToolCallParser()
    out = []
    out += p.feed('some reply <tool_call>{"name"')
    out += p.feed(': "explain_node", "input":')
    out += p.feed(' {"node_type": "X"}}</tool_call> trailing')
    text_chunks = [e[1] for e in out if e[0] == "text"]
    assert "some reply " in "".join(text_chunks)
    tool_calls = [e[1] for e in out if e[0] == "tool_call"]
    assert tool_calls == [{"name": "explain_node", "input": {"node_type": "X"}}]


def test_malformed_json_inside_markers_emits_error():
    p = StreamedToolCallParser()
    events = p.feed('<tool_call>{broken json</tool_call>')
    kinds = [e[0] for e in events]
    assert "error" in kinds
    err_msg = next(e[1] for e in events if e[0] == "error")
    assert "JSON" in err_msg or "decode" in err_msg.lower()


def test_no_tool_call_finish_is_noop():
    p = StreamedToolCallParser()
    p.feed("hi there")
    assert p.finish() == []


def test_partial_opener_not_yet_a_tool_call():
    p = StreamedToolCallParser()
    assert p.feed("<tool_c") == []
    events = p.feed('all>{"name":"inspect_canvas","input":{}}</tool_call>')
    assert [e[0] for e in events] == ["tool_call"]
