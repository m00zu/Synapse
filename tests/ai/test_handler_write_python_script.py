from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tool_handlers.write_python_script import make_write_python_script_handler


def _python_node(node_id="n1") -> FakeNode:
    return FakeNode(node_id, "PythonScriptNode")


def test_writes_code_to_existing_python_script_node():
    g = FakeGraph(); n = _python_node(); g.add_node(n)
    client = MagicMock()
    client.chat_multi.return_value = "out_1 = in_1.copy()\nout_1['log_a'] = np.log2(in_1['a'])"
    handler = make_write_python_script_handler(graph=g, client=client)
    out = handler({
        "node_id": "n1",
        "description": "log2 of column a",
        "n_inputs": 1,
        "n_outputs": 1,
    })
    assert "error" not in out
    assert out["target_node_id"] == "n1"
    assert n.model.custom_properties["script_code"].startswith("out_1 = in_1.copy()")
    assert out["line_count"] == 2
    assert out["assigned_outputs"] == ["out_1"]


def test_refuses_wrong_node_type():
    g = FakeGraph(); n = FakeNode("n1", "CSVLoader"); g.add_node(n)
    handler = make_write_python_script_handler(graph=g, client=MagicMock())
    out = handler({"node_id": "n1", "description": "x"})
    assert "error" in out and "PythonScriptNode" in out["error"]


def test_refuses_missing_node():
    handler = make_write_python_script_handler(graph=FakeGraph(), client=MagicMock())
    out = handler({"node_id": "ghost", "description": "x"})
    assert "error" in out


def test_missing_description_returns_error():
    g = FakeGraph(); g.add_node(_python_node())
    handler = make_write_python_script_handler(graph=g, client=MagicMock())
    out = handler({"node_id": "n1"})
    assert "error" in out and "description" in out["error"]


def test_strips_markdown_fences_from_llm_output():
    g = FakeGraph(); n = _python_node(); g.add_node(n)
    client = MagicMock()
    client.chat_multi.return_value = "```python\nout_1 = in_1\n```"
    handler = make_write_python_script_handler(graph=g, client=client)
    out = handler({"node_id": "n1", "description": "passthrough"})
    assert "error" not in out
    assert n.model.custom_properties["script_code"] == "out_1 = in_1"


def test_resizes_ports_via_set_property():
    g = FakeGraph(); n = _python_node(); g.add_node(n)
    client = MagicMock()
    client.chat_multi.return_value = "out_1 = in_1\nout_2 = in_2"
    handler = make_write_python_script_handler(graph=g, client=client)
    handler({
        "node_id": "n1",
        "description": "passthrough two inputs",
        "n_inputs": 2,
        "n_outputs": 2,
    })
    assert n.model.custom_properties.get("n_inputs") == 2
    assert n.model.custom_properties.get("n_outputs") == 2
