import pytest
pytest.importorskip("PySide6")



def test_session_starts_empty():
    from synapse.server.session import SessionState
    s = SessionState()
    assert s.graph.all_nodes() == []


def test_session_can_add_and_remove_node():
    from synapse.server.session import SessionState
    s = SessionState()
    nid = s.graph.add_node("GaussianBlurNode")
    assert len(s.graph.all_nodes()) == 1
    s.graph.remove_node(nid)
    assert s.graph.all_nodes() == []


def test_session_rejects_unknown_node_type():
    from synapse.server.session import SessionState
    s = SessionState()
    with pytest.raises(ValueError, match="unknown node type"):
        s.graph.add_node("NotARealNode")


def test_session_set_prop_updates_node():
    from synapse.server.session import SessionState
    s = SessionState()
    nid = s.graph.add_node("GaussianBlurNode")
    s.graph.set_prop(nid, "sigma", 2.5)
    n = s.graph.get_node(nid)
    assert n.get_property("sigma") == 2.5


def test_session_connect_nodes():
    from synapse.server.session import SessionState
    s = SessionState()
    src = s.graph.add_node("ImageReadNode")
    dst = s.graph.add_node("BinaryThresholdNode")
    # Auto-wire by type; first compatible port pair.
    s.graph.connect(src, dst)
    assert s.graph.is_connected(src, dst)


def test_session_export_import_roundtrip():
    from synapse.server.session import SessionState
    s = SessionState()
    a = s.graph.add_node("ImageReadNode")
    b = s.graph.add_node("BinaryThresholdNode")
    s.graph.set_prop(b, "thresh_state", [128, 1])
    s.graph.connect(a, b)
    workflow = s.graph.export()
    s2 = SessionState()
    s2.graph.import_(workflow)
    nodes = s2.graph.all_nodes()
    assert len(nodes) == 2
