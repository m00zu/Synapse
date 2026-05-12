"""Tests for create_workflow: validation, atomic rollback, optional run."""
import pytest

from synapse.mcp.controller import FakeGraphController, NodeInfo
from synapse.mcp.tools.compose import create_workflow


@pytest.fixture
def ctl():
    return FakeGraphController(registered=[
        NodeInfo('Chem.IO', 'MolTable Reader',
                 'plugins.Chem.IO.MolTableReaderNode',
                 ['file_path', 'smiles_col'], [], ['mol_table'], 'docs'),
        NodeInfo('Chem.Batch', 'Fingerprint',
                 'plugins.Chem.Batch.FingerprintColumnNode',
                 [], ['mol_table'], ['mol_table'], 'docs'),
        NodeInfo('ML.Cluster', 'K-Means',
                 'plugins.ML.Cluster.KMeansNode',
                 ['n_clusters'], ['table'], ['table'], 'docs'),
    ])


def _good_definition():
    return {
        'nodes': [
            {'id': 'a',
             'type': 'plugins.Chem.IO.MolTableReaderNode',
             'properties': {'file_path': '/x.csv', 'smiles_col': 'smi'}},
            {'id': 'b',
             'type': 'plugins.Chem.Batch.FingerprintColumnNode'},
            {'id': 'c',
             'type': 'plugins.ML.Cluster.KMeansNode',
             'properties': {'n_clusters': 5}},
        ],
        'connections': [
            {'src': 'a', 'src_port': 'mol_table',
             'dst': 'b', 'dst_port': 'mol_table'},
            {'src': 'b', 'src_port': 'mol_table',
             'dst': 'c', 'dst_port': 'table'},
        ],
    }


def test_happy_path_returns_id_mapping(ctl):
    result = create_workflow(ctl, _good_definition())
    assert set(result['created_ids']) == {'a', 'b', 'c'}
    # All aliases got real ids.
    assert all(isinstance(v, str) and v for v in result['created_ids'].values())
    # Graph reflects 3 nodes + 2 connections.
    active = ctl.list_active()
    assert len(active) == 3
    assert len(ctl.list_connections()) == 2


def test_run_false_by_default_no_execution(ctl):
    result = create_workflow(ctl, _good_definition())
    assert 'run_results' not in result


def test_run_true_executes_terminal_nodes(ctl):
    # Pre-program a run result for the eventual id of node 'c'.
    # We need to know what id 'c' ends up as — assume FakeGraphController
    # numbers from n1 in insertion order.
    defn = _good_definition()
    # Pre-set run result for the third node to be created (n3).
    ctl.set_run_result('n3', success=True, message='5 clusters',
                       duration_ms=12.0)
    result = create_workflow(ctl, defn, run=True)
    # Terminal = nodes with no outgoing connections; here that's 'c'.
    assert 'run_results' in result
    assert 'c' in result['run_results']  # keyed by alias
    assert result['run_results']['c']['success'] is True


def test_validation_unknown_type_rolls_back_everything(ctl):
    defn = _good_definition()
    defn['nodes'][1]['type'] = 'plugins.Chem.Batch.FingerprintColumnTypo'
    with pytest.raises(ValueError) as exc:
        create_workflow(ctl, defn)
    # Atomic: nothing was created.
    assert ctl.list_active() == []
    assert ctl.list_connections() == []
    # Error names the offending alias + the bad type.
    msg = str(exc.value)
    assert "'b'" in msg
    assert 'FingerprintColumnTypo' in msg


def test_validation_duplicate_alias_rolls_back(ctl):
    defn = _good_definition()
    defn['nodes'].append({'id': 'a',  # duplicate
                          'type': 'plugins.ML.Cluster.KMeansNode'})
    with pytest.raises(ValueError) as exc:
        create_workflow(ctl, defn)
    assert ctl.list_active() == []
    assert 'duplicate' in str(exc.value).lower()
    assert "'a'" in str(exc.value)


def test_validation_unknown_alias_in_connection_rolls_back(ctl):
    defn = _good_definition()
    defn['connections'][0]['src'] = 'phantom'
    with pytest.raises(ValueError) as exc:
        create_workflow(ctl, defn)
    assert ctl.list_active() == []
    assert 'phantom' in str(exc.value)


def test_existing_graph_state_unchanged_on_failure(ctl):
    # Seed the graph with one pre-existing node.
    pre_id = ctl.add_node('plugins.ML.Cluster.KMeansNode')
    pre_count = len(ctl.list_active())

    defn = _good_definition()
    defn['nodes'][0]['type'] = 'bogus.Type'

    with pytest.raises(ValueError):
        create_workflow(ctl, defn)

    # Original node still there; nothing from the failed batch leaked in.
    assert len(ctl.list_active()) == pre_count
    assert ctl.get_node(pre_id) is not None
