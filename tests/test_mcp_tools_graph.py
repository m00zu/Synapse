"""Tests for graph-manipulation tools."""
import pytest

from synapse.mcp.controller import FakeGraphController, NodeInfo
from synapse.mcp.tools.graph import (
    describe_graph, add_node, delete_node, set_property,
    connect, disconnect,
)


@pytest.fixture
def ctl():
    return FakeGraphController(registered=[
        NodeInfo('Chem.IO', 'MolTable Reader',
                 'plugins.Chem.IO.MolTableReaderNode',
                 ['file_path', 'smiles_col'], [], ['mol_table'], 'docs'),
        NodeInfo('ML.Class', 'Random Forest',
                 'plugins.ML.Class.RandomForestClassifierNode',
                 ['target_column'], ['train'], ['model'], 'docs'),
    ])


def test_describe_graph_empty(ctl):
    snap = describe_graph(ctl)
    assert snap == {'nodes': [], 'connections': []}


def test_describe_graph_after_add(ctl):
    add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode',
             properties={'smiles_col': 'smi'})
    snap = describe_graph(ctl)
    assert len(snap['nodes']) == 1
    n = snap['nodes'][0]
    assert n['type'] == 'plugins.Chem.IO.MolTableReaderNode'
    assert n['properties']['smiles_col'] == 'smi'


def test_add_node_returns_id_and_initial_props(ctl):
    result = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode',
                      properties={'smiles_col': 'smi'})
    assert 'node_id' in result
    assert result['inputs'] == []
    assert result['outputs'] == ['mol_table']


def test_add_node_unknown_type_raises_actionable(ctl):
    with pytest.raises(ValueError) as exc:
        add_node(ctl, 'bogus.type')
    assert 'bogus.type' in str(exc.value)
    assert 'list_nodes' in str(exc.value)


def test_set_property_persists(ctl):
    r = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')
    set_property(ctl, r['node_id'], 'smiles_col', 'canonical_smiles')
    snap = describe_graph(ctl)
    assert snap['nodes'][0]['properties']['smiles_col'] == 'canonical_smiles'


def test_connect_roundtrips_into_snapshot(ctl):
    a = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')['node_id']
    b = add_node(ctl, 'plugins.ML.Class.RandomForestClassifierNode')['node_id']
    connect(ctl, a, 'mol_table', b, 'train')
    snap = describe_graph(ctl)
    assert {'src_node_id': a, 'src_port': 'mol_table',
            'dst_node_id': b, 'dst_port': 'train'} in snap['connections']


def test_connect_unknown_node_raises_actionable(ctl):
    a = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')['node_id']
    with pytest.raises(ValueError) as exc:
        connect(ctl, a, 'mol_table', 'missing', 'train')
    assert 'missing' in str(exc.value)
    assert 'describe_graph' in str(exc.value)


def test_disconnect_round_trips(ctl):
    a = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')['node_id']
    b = add_node(ctl, 'plugins.ML.Class.RandomForestClassifierNode')['node_id']
    connect(ctl, a, 'mol_table', b, 'train')
    disconnect(ctl, a, 'mol_table', b, 'train')
    assert describe_graph(ctl)['connections'] == []


def test_disconnect_no_such_edge_raises_actionable(ctl):
    a = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')['node_id']
    b = add_node(ctl, 'plugins.ML.Class.RandomForestClassifierNode')['node_id']
    with pytest.raises(ValueError) as exc:
        disconnect(ctl, a, 'mol_table', b, 'train')
    assert 'describe_graph' in str(exc.value)


def test_delete_node_removes_node_and_edges(ctl):
    a = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')['node_id']
    b = add_node(ctl, 'plugins.ML.Class.RandomForestClassifierNode')['node_id']
    connect(ctl, a, 'mol_table', b, 'train')
    delete_node(ctl, a)
    snap = describe_graph(ctl)
    assert {n['id'] for n in snap['nodes']} == {b}
    assert snap['connections'] == []  # attached edges also gone


def test_delete_node_unknown_raises_actionable(ctl):
    with pytest.raises(ValueError) as exc:
        delete_node(ctl, 'bogus')
    assert 'bogus' in str(exc.value)
    assert 'describe_graph' in str(exc.value)
