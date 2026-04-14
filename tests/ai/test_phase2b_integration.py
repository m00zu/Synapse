"""End-to-end Phase 2b test: user turn → orchestrator → tool_call →
dispatcher → real handler → tool_result → next turn → final text."""
from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.clients.base import StreamEvent
from synapse.ai.tools import ToolDispatcher
from synapse.ai.tool_handlers.inspect_canvas import make_inspect_canvas_handler
from synapse.ai.tool_handlers.explain_node import explain_node_handler
from synapse.ai.orchestrator import ChatOrchestrator


def _turn_streams(batches):
    """Factory returning a fresh iterator each chat_with_tools_stream call."""
    it = iter(batches)

    def stream(system, messages, tools=None):
        try:
            batch = next(it)
        except StopIteration:
            batch = [StreamEvent(kind="done")]
        for ev in batch:
            yield ev

    return stream


def test_full_turn_with_real_inspect_canvas_dispatch():
    graph = FakeGraph()
    graph.add_node(FakeNode("a", "CSVLoader", {"path": "data.csv"}))
    graph.add_node(FakeNode("b", "SortTable", {"column": "value"}))

    client = MagicMock()
    client.chat_with_tools_stream.side_effect = _turn_streams([
        # Turn 1: model calls inspect_canvas.
        [
            StreamEvent(kind="tool_call",
                        tool_call={"id": "t1", "name": "inspect_canvas", "input": {}}),
            StreamEvent(kind="done"),
        ],
        # Turn 2 (after tool result): model replies with text.
        [
            StreamEvent(kind="text", text="Your canvas has a CSVLoader and a SortTable."),
            StreamEvent(kind="done"),
        ],
    ])
    client.model = "test-model"

    dispatcher = ToolDispatcher()
    dispatcher.register("inspect_canvas", make_inspect_canvas_handler(graph))
    dispatcher.register("explain_node", explain_node_handler)

    orch = ChatOrchestrator(graph=graph, client=client, dispatcher=dispatcher)
    events = list(orch.run_turn("what's on my canvas?"))
    kinds = [e.kind for e in events]

    assert kinds.count("tool_call_started") == 1
    assert kinds.count("tool_call_finished") == 1
    assert any(e.kind == "text" and "CSVLoader" in e.text for e in events)
    assert kinds[-1] == "turn_done"

    # The tool_result injected into history must actually contain the real
    # inspect_canvas output. Since the mock client is not ClaudeClient or
    # OpenAIClient, the orchestrator falls through to the inline-user-message
    # path — look for that synthetic message.
    tool_result_user_msgs = [
        m for m in orch.history
        if m.get("role") == "user" and "Tool result for" in str(m.get("content", ""))
    ]
    assert len(tool_result_user_msgs) == 1
    content = tool_result_user_msgs[0]["content"]
    assert "CSVLoader" in content and "SortTable" in content


def test_full_turn_with_write_python_script_and_modify_workflow():
    """Orchestrator → modify_workflow(add PythonScriptNode) → write_python_script → text."""
    from synapse.ai.tool_handlers.write_python_script import make_write_python_script_handler
    from synapse.ai.tool_handlers.modify_workflow import make_modify_workflow_handler

    graph = FakeGraph()

    client = MagicMock()
    # The write_python_script handler calls client.chat_multi (the sub-LLM path)
    # while the orchestrator's outer loop calls client.chat_with_tools_stream.
    client.chat_multi.return_value = "out_1 = in_1  # passthrough"
    client.chat_with_tools_stream.side_effect = _turn_streams([
        # Turn 1: add a PythonScriptNode.
        [
            StreamEvent(kind="tool_call", tool_call={
                "id": "t1", "name": "modify_workflow",
                "input": {"operations":
                          [{"op": "add_node",
                            "type": "PythonScriptNode",
                            "id": "py1"}]}}),
            StreamEvent(kind="done"),
        ],
        # Turn 2: write code to it.
        [
            StreamEvent(kind="tool_call", tool_call={
                "id": "t2", "name": "write_python_script",
                "input": {"node_id": "py1",
                          "description": "passthrough"}}),
            StreamEvent(kind="done"),
        ],
        # Turn 3: final text.
        [
            StreamEvent(kind="text", text="Done! I added a PythonScriptNode."),
            StreamEvent(kind="done"),
        ],
    ])
    client.model = "m"

    dispatcher = ToolDispatcher()
    dispatcher.register("modify_workflow", make_modify_workflow_handler(
        graph, node_factory=lambda t, i: FakeNode(i, t)))
    dispatcher.register("write_python_script",
                        make_write_python_script_handler(graph, client))

    orch = ChatOrchestrator(graph=graph, client=client, dispatcher=dispatcher)
    events = list(orch.run_turn("add a passthrough python script"))

    # End state: py1 exists, has script_code, final text reached.
    assert graph.get_node_by_id("py1") is not None
    assert "out_1 = in_1" in graph.get_node_by_id("py1").model.custom_properties.get("script_code", "")
    assert any(e.kind == "text" and "Done" in e.text for e in events)
    assert events[-1].kind == "turn_done"
