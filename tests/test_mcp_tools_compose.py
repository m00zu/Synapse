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


# ── Partial-success semantics for connections ────────────────────────────

def test_happy_path_reports_connections_made_and_no_failures(ctl):
    result = create_workflow(ctl, _good_definition())
    assert len(result['connections_made']) == 2
    assert result['connections_failed'] == []
    # Each made connection records resolved port names.
    for m in result['connections_made']:
        assert {'src', 'src_port', 'dst', 'dst_port', 'fuzzy_matched'} <= set(m)


def test_bad_port_name_does_not_rollback_nodes(ctl):
    defn = _good_definition()
    # Typo: 'output' is not a port on the MolTable Reader (it's 'mol_table').
    defn['connections'][0]['src_port'] = 'output'
    result = create_workflow(ctl, defn)
    # All three nodes were still created.
    assert len(ctl.list_active()) == 3
    assert set(result['created_ids']) == {'a', 'b', 'c'}
    # The bad connection is reported in connections_failed, not raised.
    assert len(result['connections_failed']) == 1
    failure = result['connections_failed'][0]
    assert failure['attempted']['src_port'] == 'output'
    assert 'mol_table' in failure['available_src_ports']
    # The other (good) connection still succeeded.
    assert len(result['connections_made']) == 1
    assert len(ctl.list_connections()) == 1


def test_case_insensitive_port_match(ctl):
    defn = _good_definition()
    # 'Mol_Table' vs 'mol_table' -- case-insensitive resolution should win.
    defn['connections'][0]['src_port'] = 'Mol_Table'
    result = create_workflow(ctl, defn)
    assert result['connections_failed'] == []
    assert len(result['connections_made']) == 2
    # The match is flagged as fuzzy so the LLM knows it wasn't exact.
    fuzzy = [m for m in result['connections_made'] if m['fuzzy_matched']]
    assert len(fuzzy) == 1
    assert fuzzy[0]['src_port'] == 'mol_table'  # resolved to canonical case


def test_run_skipped_when_connection_failed(ctl):
    defn = _good_definition()
    defn['connections'][0]['src_port'] = 'nonexistent'
    result = create_workflow(ctl, defn, run=True)
    # Skip is reported; run_results is absent.
    assert 'run_skipped' in result
    assert 'run_results' not in result


def test_failed_connection_includes_available_ports_for_both_ends(ctl):
    defn = _good_definition()
    # Both src and dst are wrong.
    defn['connections'][0]['src_port'] = 'bogus_src'
    defn['connections'][0]['dst_port'] = 'bogus_dst'
    result = create_workflow(ctl, defn)
    failure = result['connections_failed'][0]
    # Both available port lists are present so the LLM can fix the wire
    # in one follow-up call.
    assert 'available_src_ports' in failure
    assert 'available_dst_ports' in failure
    assert 'mol_table' in failure['available_src_ports']
    assert 'mol_table' in failure['available_dst_ports']
