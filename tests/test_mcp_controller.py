"""Test the GraphController abstraction with a fake implementation."""
import pytest

from synapse.mcp.controller import FakeGraphController, NodeInfo, NodeRecord


def test_fake_list_registered_returns_seeded_nodes():
    ctl = FakeGraphController(registered=[
        NodeInfo('plugins.ML.Classification', 'Random Forest',
                 'plugins.ML.Classification.RandomForestClassifierNode',
                 ['target_column', 'feature_columns'],
                 ['train'], ['model', 'result'],
                 'Trains a Random Forest classifier.'),
    ])
    out = ctl.list_registered()
    assert len(out) == 1
    assert out[0].name == 'Random Forest'


def test_fake_add_node_assigns_id():
    ctl = FakeGraphController(registered=[
        NodeInfo('plugins.ML.Classification', 'Random Forest',
                 'plugins.ML.Classification.RandomForestClassifierNode',
                 [], [], [], 'docs'),
    ])
    nid = ctl.add_node(
        'plugins.ML.Classification.RandomForestClassifierNode')
    assert nid == 'n1'
    nid2 = ctl.add_node(
        'plugins.ML.Classification.RandomForestClassifierNode')
    assert nid2 == 'n2'
    assert {n.id for n in ctl.list_active()} == {'n1', 'n2'}


def test_fake_add_node_unknown_type_raises():
    ctl = FakeGraphController(registered=[])
    with pytest.raises(KeyError):
        ctl.add_node('not.a.real.type')


def test_fake_set_property_round_trips():
    ctl = FakeGraphController(registered=[
        NodeInfo('cat', 'N', 'cat.N', ['x'], [], [], 'd'),
    ])
    nid = ctl.add_node('cat.N')
    ctl.set_property(nid, 'x', 42)
    rec = ctl.get_node(nid)
    assert rec.properties['x'] == 42


def test_fake_connect_records_link():
    ctl = FakeGraphController(registered=[
        NodeInfo('cat', 'A', 'cat.A', [], [], ['out'], 'd'),
        NodeInfo('cat', 'B', 'cat.B', [], ['in'], [], 'd'),
    ])
    a = ctl.add_node('cat.A')
    b = ctl.add_node('cat.B')
    ctl.connect(a, 'out', b, 'in')
    conns = ctl.list_connections()
    assert (a, 'out', b, 'in') in conns


def test_fake_disconnect_removes_link():
    ctl = FakeGraphController(registered=[
        NodeInfo('cat', 'A', 'cat.A', [], [], ['out'], 'd'),
        NodeInfo('cat', 'B', 'cat.B', [], ['in'], [], 'd'),
    ])
    a = ctl.add_node('cat.A')
    b = ctl.add_node('cat.B')
    ctl.connect(a, 'out', b, 'in')
    ctl.disconnect(a, 'out', b, 'in')
    assert ctl.list_connections() == []


def test_fake_delete_node_drops_node_and_connections():
    ctl = FakeGraphController(registered=[
        NodeInfo('cat', 'A', 'cat.A', [], [], ['out'], 'd'),
        NodeInfo('cat', 'B', 'cat.B', [], ['in'], [], 'd'),
    ])
    a = ctl.add_node('cat.A')
    b = ctl.add_node('cat.B')
    ctl.connect(a, 'out', b, 'in')
    ctl.delete_node(a)
    assert {n.id for n in ctl.list_active()} == {b}
    assert ctl.list_connections() == []  # edges to/from 'a' are gone


def test_fake_delete_node_unknown_id_raises():
    ctl = FakeGraphController(registered=[])
    with pytest.raises(KeyError):
        ctl.delete_node('nonexistent')


def test_fake_run_node_returns_canned_result():
    ctl = FakeGraphController(registered=[
        NodeInfo('cat', 'A', 'cat.A', [], [], [], 'd'),
    ])
    a = ctl.add_node('cat.A')
    ctl.set_run_result(a, success=True, message='ok', duration_ms=12.3)
    result = ctl.run_node(a)
    assert result == {'success': True, 'message': 'ok', 'duration_ms': 12.3}
