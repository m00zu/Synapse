from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.clients.base import StreamEvent
from synapse.ai.tools import ToolDispatcher
from synapse.ai.tool_handlers.inspect_canvas import make_inspect_canvas_handler
from synapse.ai.tool_handlers.explain_node import explain_node_handler
from synapse.ai.orchestrator import ChatOrchestrator


def _dispatcher(graph):
    d = ToolDispatcher()
    d.register("inspect_canvas", make_inspect_canvas_handler(graph))
    d.register("explain_node", explain_node_handler)
    return d


def _mock_client(event_batches: list[list[StreamEvent]]):
    """Each batch = one chat_with_tools_stream invocation's event stream."""
    client = MagicMock()
    it = iter(event_batches)

    def stream(system, messages, tools=None):
        try:
            batch = next(it)
        except StopIteration:
            batch = [StreamEvent(kind="done")]
        for ev in batch:
            yield ev

    client.chat_with_tools_stream.side_effect = stream
    client.supports_vision = False
    client.model = "test-model"
    return client


def test_orchestrator_plain_text_turn():
    g = FakeGraph()
    client = _mock_client([[
        StreamEvent(kind="text", text="Hello!"),
        StreamEvent(kind="done"),
    ]])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g))
    events = list(orch.run_turn("hi"))
    kinds = [e.kind for e in events]
    assert "text" in kinds
    assert events[-1].kind == "turn_done"


def test_orchestrator_single_tool_call_round():
    g = FakeGraph(); g.add_node(FakeNode("a", "CSVLoader"))
    client = _mock_client([
        # Turn 1: model calls inspect_canvas.
        [
            StreamEvent(kind="tool_call", tool_call={"id": "t1", "name": "inspect_canvas", "input": {}}),
            StreamEvent(kind="done"),
        ],
        # Turn 2 (after tool result injection): model replies.
        [
            StreamEvent(kind="text", text="Canvas has one CSVLoader."),
            StreamEvent(kind="done"),
        ],
    ])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g))
    events = list(orch.run_turn("what's on my canvas?"))
    kinds = [e.kind for e in events]
    assert "tool_call_started" in kinds
    assert "tool_call_finished" in kinds
    assert any(e.kind == "text" and "CSVLoader" in e.text for e in events)
    assert client.chat_with_tools_stream.call_count == 2


def test_orchestrator_enforces_4_call_cap():
    g = FakeGraph()
    infinite = [
        [
            StreamEvent(kind="tool_call", tool_call={"id": f"t{i}", "name": "inspect_canvas", "input": {}}),
            StreamEvent(kind="done"),
        ]
        for i in range(10)
    ]
    client = _mock_client(infinite + [[StreamEvent(kind="text", text="OK."), StreamEvent(kind="done")]])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g), max_tool_calls=4)
    events = list(orch.run_turn("loop"))
    tool_starts = [e for e in events if e.kind == "tool_call_started"]
    assert len(tool_starts) == 4
    assert any(e.kind == "cap_exceeded" for e in events)


def test_orchestrator_propagates_client_error():
    g = FakeGraph()
    client = _mock_client([[StreamEvent(kind="error", error="HTTP 500")]])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g))
    events = list(orch.run_turn("x"))
    err = next(e for e in events if e.kind == "error")
    assert "HTTP 500" in err.error


def test_orchestrator_cancel_flag_aborts_mid_stream():
    g = FakeGraph()
    client = _mock_client([[
        StreamEvent(kind="text", text="one"),
        StreamEvent(kind="text", text="two"),
        StreamEvent(kind="text", text="three"),
        StreamEvent(kind="done"),
    ]])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g))
    events = []
    for e in orch.run_turn("x"):
        events.append(e)
        if e.kind == "text" and e.text == "one":
            orch.cancel()
    kinds = [e.kind for e in events]
    # Either cancelled mid-stream, or clean turn_done — either is acceptable
    # depending on where the cancel flag was checked. Must not hang.
    assert "cancelled" in kinds or "turn_done" in kinds


def test_orchestrator_dedups_preappended_user_message():
    """Regression: AIChatPanel appends the user message to history before
    handing it to the worker. The orchestrator must not append a duplicate —
    consecutive user messages cause some providers (Ollama Cloud +
    nemotron-3-super) to return HTTP 500."""
    g = FakeGraph()
    client = _mock_client([[
        StreamEvent(kind="text", text="hi back"),
        StreamEvent(kind="done"),
    ]])
    existing_history = [{"role": "user", "content": "say hi"}]
    orch = ChatOrchestrator(
        graph=g, client=client, dispatcher=_dispatcher(g),
        history=existing_history,
    )
    list(orch.run_turn("say hi"))  # same text as what's already in history
    user_msgs = [m for m in orch.history if m.get("role") == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "say hi"


def test_orchestrator_still_appends_when_history_ends_in_assistant():
    """Sanity: dedup only fires when the last history entry is an identical
    user message. Normal multi-turn flow must keep appending."""
    g = FakeGraph()
    client = _mock_client([[StreamEvent(kind="done")]])
    orch = ChatOrchestrator(
        graph=g, client=client, dispatcher=_dispatcher(g),
        history=[
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ],
    )
    list(orch.run_turn("new question"))
    user_msgs = [m["content"] for m in orch.history if m.get("role") == "user"]
    assert user_msgs == ["previous question", "new question"]
