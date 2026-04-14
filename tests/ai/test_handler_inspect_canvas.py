from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tool_handlers.inspect_canvas import make_inspect_canvas_handler


def _graph_with_two_nodes():
    g = FakeGraph()
    a = FakeNode("a", "CSVLoader", {"path": "/x.csv"})
    b = FakeNode("b", "SortTable", {"column": "value"})
    g.add_node(a); g.add_node(b)
    a.add_output("out_1").connect_to(b.add_input("in_1"))
    return g


def test_inspect_returns_all_nodes_when_node_ids_empty():
    g = _graph_with_two_nodes()
    handler = make_inspect_canvas_handler(g)
    out = handler({})
    assert set(n["type"] for n in out["nodes"]) == {"CSVLoader", "SortTable"}
    assert len(out["edges"]) == 1


def test_inspect_filters_to_requested_nodes():
    g = _graph_with_two_nodes()
    handler = make_inspect_canvas_handler(g)
    out = handler({"node_ids": ["a"]})
    assert [n["type"] for n in out["nodes"]] == ["CSVLoader"]
    assert out["edges"] == [] or all(
        e_src in {"a"} or e_dst in {"a"}
        for e_src, e_dst in (tuple(e) for e in out["edges"])
    )


def test_inspect_omits_props_when_include_props_false():
    g = _graph_with_two_nodes()
    handler = make_inspect_canvas_handler(g)
    out = handler({"include_props": False})
    for n in out["nodes"]:
        assert "props" not in n


def test_inspect_sets_truncated_flag_when_over_budget():
    g = FakeGraph()
    for i in range(50):
        g.add_node(FakeNode(f"n{i}", "BigNode", {"blob": "x" * 500}))
    handler = make_inspect_canvas_handler(g, token_cap=200)
    out = handler({})
    assert out["truncated"] is True
    assert isinstance(out["nodes"], list)
