from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tools import ToolDispatcher, TOOL_NAMES
from synapse.ai.tool_handlers.inspect_canvas import make_inspect_canvas_handler
from synapse.ai.tool_handlers.explain_node import explain_node_handler
from synapse.ai.tool_handlers.read_node_output import make_read_node_output_handler
from synapse.ai.tool_handlers.generate_workflow import make_generate_workflow_handler
from synapse.ai.tool_handlers.modify_workflow import make_modify_workflow_handler
from synapse.ai.tool_handlers.write_python_script import make_write_python_script_handler


def _all_tool_dispatcher():
    graph = FakeGraph()
    client = MagicMock()
    client.chat_multi.side_effect = [
        '{"nodes":["CSVLoader"]}',
        '{"nodes":[{"id":1,"type":"CSVLoader"}],"edges":[]}',
        "out_1 = in_1",  # for write_python_script
    ]
    d = ToolDispatcher()
    d.register("inspect_canvas", make_inspect_canvas_handler(graph))
    d.register("explain_node", explain_node_handler)
    d.register("read_node_output", make_read_node_output_handler(graph, lambda: False))
    d.register("generate_workflow", make_generate_workflow_handler(graph, client))
    d.register("modify_workflow", make_modify_workflow_handler(
        graph, node_factory=lambda t, i: FakeNode(i, t)))
    d.register("write_python_script", make_write_python_script_handler(graph, client))
    return d, graph


def test_dispatcher_knows_all_six_tools():
    d, _ = _all_tool_dispatcher()
    assert set(d.registered_names()) == set(TOOL_NAMES)


def test_inspect_canvas_empty_graph():
    d, _ = _all_tool_dispatcher()
    out = d.dispatch("inspect_canvas", {})
    assert out == {"nodes": [], "edges": [], "truncated": False}


def test_explain_node_unknown():
    d, _ = _all_tool_dispatcher()
    out = d.dispatch("explain_node", {"node_type": "NoSuchNode"})
    assert "error" in out


def test_end_to_end_modify_then_read():
    d, graph = _all_tool_dispatcher()
    d.dispatch("modify_workflow", {"operations": [
        {"op": "add_node", "type": "PythonScriptNode", "id": "py1"},
    ]})
    out_write = d.dispatch("write_python_script", {
        "node_id": "py1", "description": "passthrough",
        "n_inputs": 1, "n_outputs": 1,
    })
    assert "error" not in out_write
    out_inspect = d.dispatch("inspect_canvas", {})
    ids = [n["id"] for n in out_inspect["nodes"]]
    assert "py1" in ids


def test_bad_tool_name_returns_wrapped_error():
    d, _ = _all_tool_dispatcher()
    out = d.dispatch("summon_demon", {})
    assert "error" in out
