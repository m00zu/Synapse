from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tool_handlers.modify_workflow import make_modify_workflow_handler


class _Factory:
    """Tiny stand-in for NodeGraphQt's create_node. Tests record each call."""
    def __init__(self):
        self.created = []

    def __call__(self, type_name: str, node_id: str) -> FakeNode:
        n = FakeNode(node_id, type_name)
        self.created.append(n)
        return n


def _setup():
    g = FakeGraph(); factory = _Factory()
    handler = make_modify_workflow_handler(graph=g, node_factory=factory)
    return g, factory, handler


def test_add_node_applies_and_reports_success():
    g, factory, handler = _setup()
    out = handler({"operations": [
        {"op": "add_node", "type": "CSVLoader", "id": "n1"},
    ]})
    assert out["applied"] == [{"op": "add_node", "id": "n1"}]
    assert out["failed"] == []
    assert g.get_node_by_id("n1") is not None


def test_connect_disconnect_roundtrip():
    g, factory, handler = _setup()
    out = handler({"operations": [
        {"op": "add_node", "type": "CSVLoader", "id": "a"},
        {"op": "add_node", "type": "SortTable", "id": "b"},
        {"op": "connect", "src": "a", "src_port": "out_1",
                          "dst": "b", "dst_port": "in_1"},
    ]})
    assert out["failed"] == []
    a = g.get_node_by_id("a"); b = g.get_node_by_id("b")
    assert a.outputs()["out_1"].connected_ports()[0].node().id == "b"

    out2 = handler({"operations": [
        {"op": "disconnect", "src": "a", "src_port": "out_1",
                             "dst": "b", "dst_port": "in_1"},
    ]})
    assert out2["failed"] == []
    assert a.outputs()["out_1"].connected_ports() == []


def test_set_prop_updates_custom_property():
    g, factory, handler = _setup()
    handler({"operations": [{"op": "add_node", "type": "CSVLoader", "id": "n1"}]})
    out = handler({"operations": [
        {"op": "set_prop", "id": "n1", "prop": "path", "value": "/data.csv"},
    ]})
    assert out["failed"] == []
    assert g.get_node_by_id("n1").model.custom_properties["path"] == "/data.csv"


def test_remove_node():
    g, factory, handler = _setup()
    handler({"operations": [{"op": "add_node", "type": "X", "id": "n1"}]})
    out = handler({"operations": [{"op": "remove_node", "id": "n1"}]})
    assert out["failed"] == []
    assert g.get_node_by_id("n1") is None


def test_partial_success_reports_failures_individually():
    g, factory, handler = _setup()
    out = handler({"operations": [
        {"op": "add_node", "type": "CSVLoader", "id": "a"},
        {"op": "set_prop", "id": "nonexistent", "prop": "x", "value": 1},
        {"op": "add_node", "type": "SortTable", "id": "b"},
    ]})
    applied_ids = [a["id"] for a in out["applied"]]
    assert "a" in applied_ids and "b" in applied_ids
    assert len(out["failed"]) == 1
    assert out["failed"][0]["op"]["id"] == "nonexistent"


def test_unknown_op_kind_reports_failure():
    g, factory, handler = _setup()
    out = handler({"operations": [{"op": "do_magic"}]})
    assert out["applied"] == []
    assert len(out["failed"]) == 1
    assert "unknown op" in out["failed"][0]["reason"].lower()
