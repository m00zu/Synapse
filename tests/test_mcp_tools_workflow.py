"""Tests for workflow-I/O tools (new/save/load)."""
import json

import pytest

from synapse.mcp.controller import FakeGraphController, NodeInfo
from synapse.mcp.tools.workflow import (
    new_workflow, save_workflow, load_workflow,
)
from synapse.mcp.tools.graph import add_node, connect, describe_graph


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


def test_new_workflow_clears_everything(ctl):
    a = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')['node_id']
    b = add_node(ctl,
                  'plugins.ML.Class.RandomForestClassifierNode')['node_id']
    connect(ctl, a, 'mol_table', b, 'train')
    assert len(describe_graph(ctl)['nodes']) == 2

    result = new_workflow(ctl)
    assert result == {'cleared': True}
    snap = describe_graph(ctl)
    assert snap == {'nodes': [], 'connections': []}


def test_save_workflow_writes_json(ctl, tmp_path):
    a = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode',
                 properties={'smiles_col': 'smi'})['node_id']
    target = tmp_path / 'pipeline.json'
    result = save_workflow(ctl, str(target))
    assert result == {'saved_to': str(target)}
    assert target.is_file()
    data = json.loads(target.read_text())
    assert any(n['id'] == a for n in data['nodes'])
    # The recorded property survives.
    assert data['nodes'][0]['properties']['smiles_col'] == 'smi'


def test_save_workflow_creates_parent_dirs(ctl, tmp_path):
    add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')
    target = tmp_path / 'a' / 'b' / 'c' / 'wf.json'
    save_workflow(ctl, str(target))
    assert target.is_file()


def test_save_then_load_round_trips(ctl, tmp_path):
    a = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode',
                 properties={'smiles_col': 'x'})['node_id']
    b = add_node(ctl,
                  'plugins.ML.Class.RandomForestClassifierNode')['node_id']
    connect(ctl, a, 'mol_table', b, 'train')

    target = tmp_path / 'wf.json'
    save_workflow(ctl, str(target))

    # Mutate the graph; load should clobber.
    new_workflow(ctl)
    assert describe_graph(ctl) == {'nodes': [], 'connections': []}

    load_workflow(ctl, str(target))
    snap = describe_graph(ctl)
    assert len(snap['nodes']) == 2
    assert len(snap['connections']) == 1
    # ids preserved.
    ids = {n['id'] for n in snap['nodes']}
    assert ids == {a, b}


def test_load_workflow_missing_file_raises_actionable(ctl, tmp_path):
    with pytest.raises(ValueError) as exc:
        load_workflow(ctl, str(tmp_path / 'nope.json'))
    assert 'No such file' in str(exc.value)


def test_load_workflow_unknown_type_raises_actionable(ctl, tmp_path):
    # Hand-write a session referencing a type that's not registered.
    target = tmp_path / 'wf.json'
    target.write_text(json.dumps({
        'nodes': [{'id': 'n1', 'type_id': 'gone.NodeType',
                   'name': 'X', 'properties': {}}],
        'connections': [],
    }))
    with pytest.raises(ValueError) as exc:
        load_workflow(ctl, str(target))
    assert 'gone.NodeType' in str(exc.value)
    assert 'list_nodes' in str(exc.value)


def test_load_workflow_resets_id_counter(ctl, tmp_path):
    """After load, the next add_node should produce an id that doesn't
    collide with loaded ids."""
    add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')
    add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')   # n1, n2
    target = tmp_path / 'wf.json'
    save_workflow(ctl, str(target))

    new_workflow(ctl)
    load_workflow(ctl, str(target))
    nid3 = add_node(ctl, 'plugins.Chem.IO.MolTableReaderNode')['node_id']
    existing = {n['id'] for n in describe_graph(ctl)['nodes']}
    assert nid3 not in (existing - {nid3})
    # nid3 must not equal n1 or n2.
    assert nid3 not in ('n1', 'n2')
