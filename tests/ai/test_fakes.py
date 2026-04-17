from tests.ai.fakes import FakeGraph, FakeNode


def test_node_basic_identity():
    n = FakeNode("n1", "CSVLoader", {"path": "x.csv"})
    assert n.id == "n1"
    assert n.type_name == "CSVLoader"
    assert n.model.custom_properties["path"] == "x.csv"


def test_node_ports_and_connection():
    a = FakeNode("a", "A"); b = FakeNode("b", "B")
    out = a.add_output("out_1")
    in_ = b.add_input("in_1")
    out.connect_to(in_)
    assert b.inputs()["in_1"].connected_ports()[0].node().id == "a"
    assert a.outputs()["out_1"].connected_ports()[0].node().id == "b"


def test_graph_add_remove_lookup():
    g = FakeGraph()
    n = FakeNode("n1", "X")
    g.add_node(n)
    assert g.get_node_by_id("n1") is n
    assert len(g.all_nodes()) == 1
    g.remove_node(n)
    assert g.get_node_by_id("n1") is None


def test_set_property_updates_custom_properties():
    n = FakeNode("n", "Script")
    n.set_property("script_code", "out_1 = in_1")
    assert n.model.custom_properties["script_code"] == "out_1 = in_1"
