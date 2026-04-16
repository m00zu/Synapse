from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tool_handlers.modify_workflow import make_modify_workflow_handler


class _Factory:
    """Tiny stand-in for NodeGraphQt's create_node. Like NodeGraphQt, this
    factory is responsible for registering the new node in the graph —
    modify_workflow's add_node op no longer calls graph.add_node() itself."""
    def __init__(self, graph):
        self.graph = graph
        self.created = []

    def __call__(self, type_name: str, node_id: str) -> FakeNode:
        n = FakeNode(node_id, type_name)
        self.graph.add_node(n)
        self.created.append(n)
        return n


def _setup():
    g = FakeGraph(); factory = _Factory(g)
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


# --- auto-layout for new nodes -------------------------------------------

class _PositionableNode(FakeNode):
    """FakeNode with pos()/set_pos() — stand-in for NodeGraphQt nodes."""
    def __init__(self, node_id: str, type_name: str, x: float = 0, y: float = 0):
        super().__init__(node_id, type_name)
        self._x, self._y = float(x), float(y)

    def pos(self):
        return (self._x, self._y)

    def set_pos(self, x, y):
        self._x, self._y = float(x), float(y)


def test_auto_layout_places_new_node_downstream_of_existing():
    """When a new node is connected from an existing positioned node, it
    should appear to the right of that node (not at origin)."""
    g = FakeGraph()
    # Existing node at (100, 200).
    existing = _PositionableNode("e1", "CSVLoader", x=100, y=200)
    g.add_node(existing)

    def _pos_factory(type_name: str, node_id: str):
        n = _PositionableNode(node_id, type_name)
        g.add_node(n)
        return n

    handler = make_modify_workflow_handler(graph=g, node_factory=_pos_factory)
    handler({"operations": [
        {"op": "add_node", "type": "SortTable", "id": "new1"},
        {"op": "connect", "src": "e1", "dst": "new1"},
    ]})

    new_node = g.get_node_by_id("new1")
    x, y = new_node.pos()
    assert x == 400  # 100 + X_PAD(300)
    assert y == 200  # same row


def test_auto_layout_chains_through_new_nodes():
    """new1 attaches to existing, new2 attaches to new1 — both should flow
    rightward in a chain."""
    g = FakeGraph()
    existing = _PositionableNode("e1", "CSVLoader", x=0, y=0)
    g.add_node(existing)

    def _pos_factory(t, i):
        n = _PositionableNode(i, t); g.add_node(n); return n

    handler = make_modify_workflow_handler(graph=g, node_factory=_pos_factory)
    handler({"operations": [
        {"op": "add_node", "type": "A", "id": "new1"},
        {"op": "add_node", "type": "B", "id": "new2"},
        {"op": "connect", "src": "e1", "dst": "new1"},
        {"op": "connect", "src": "new1", "dst": "new2"},
    ]})

    assert g.get_node_by_id("new1").pos() == (300.0, 0.0)
    assert g.get_node_by_id("new2").pos() == (600.0, 0.0)


def test_auto_layout_stacks_siblings_vertically():
    """Two new nodes connecting from the same source should not overlap."""
    g = FakeGraph()
    src = _PositionableNode("e1", "Source", x=0, y=0)
    g.add_node(src)

    def _pos_factory(t, i):
        n = _PositionableNode(i, t); g.add_node(n); return n

    handler = make_modify_workflow_handler(graph=g, node_factory=_pos_factory)
    handler({"operations": [
        {"op": "add_node", "type": "A", "id": "a"},
        {"op": "add_node", "type": "B", "id": "b"},
        {"op": "connect", "src": "e1", "dst": "a"},
        {"op": "connect", "src": "e1", "dst": "b"},
    ]})

    a_pos = g.get_node_by_id("a").pos()
    b_pos = g.get_node_by_id("b").pos()
    assert a_pos[0] == b_pos[0] == 300.0
    assert a_pos[1] != b_pos[1]  # stacked vertically


def test_auto_layout_noop_on_nodes_without_pos_methods():
    """FakeNode (no pos/set_pos) must not crash. Tests the silent no-op path."""
    g, factory, handler = _setup()  # FakeNode factory
    out = handler({"operations": [
        {"op": "add_node", "type": "X", "id": "n1"},
    ]})
    assert out["failed"] == []
    # No pos/set_pos called — nothing to assert beyond "didn't crash".


def test_auto_layout_orphan_new_node_cascades_below_canvas():
    """A new node with no connections drops into an origin cascade below
    the existing canvas bounds (not a literal stack at 0,0)."""
    g = FakeGraph()
    existing = _PositionableNode("e1", "Source", x=0, y=500)
    g.add_node(existing)

    def _pos_factory(t, i):
        n = _PositionableNode(i, t); g.add_node(n); return n

    handler = make_modify_workflow_handler(graph=g, node_factory=_pos_factory)
    handler({"operations": [
        {"op": "add_node", "type": "Orphan", "id": "orph"},
    ]})
    x, y = g.get_node_by_id("orph").pos()
    assert x == 0.0
    assert y > 500.0  # below the existing node
