"""Tests for discovery tools using FakeGraphController."""
import pytest

from synapse.mcp.controller import FakeGraphController, NodeInfo
from synapse.mcp.tools.discovery import (
    list_nodes, describe_node, search_nodes,
)


@pytest.fixture
def ctl():
    return FakeGraphController(registered=[
        NodeInfo('Cheminformatics.IO', 'MolTable Reader',
                 'plugins.Chem.IO.MolTableReaderNode',
                 ['file_path', 'smiles_col'], [], ['mol_table'],
                 'Read a tabular file and parse a SMILES column to a MolTable.'),
        NodeInfo('ML.Classification', 'Random Forest',
                 'plugins.ML.Classification.RandomForestClassifierNode',
                 ['target_column', 'n_estimators'], ['train'],
                 ['model', 'result'],
                 'Trains a Random Forest classifier.'),
        NodeInfo('ML.Clustering', 'K-Means',
                 'plugins.ML.Clustering.KMeansNode',
                 ['n_clusters'], ['table'], ['table', 'sklearn_model'],
                 'Clusters data using K-Means algorithm.'),
    ])


def test_list_nodes_returns_summary_per_node(ctl):
    out = list_nodes(ctl)
    assert len(out) == 3
    assert {n['name'] for n in out} == {
        'MolTable Reader', 'Random Forest', 'K-Means'}
    sample = next(n for n in out if n['name'] == 'K-Means')
    assert sample['category'] == 'ML.Clustering'
    assert sample['type'] == 'plugins.ML.Clustering.KMeansNode'
    assert 'K-Means' in sample['summary']


def test_describe_node_full_info(ctl):
    info = describe_node(
        ctl, 'plugins.ML.Classification.RandomForestClassifierNode')
    assert info['name'] == 'Random Forest'
    assert info['properties'] == ['target_column', 'n_estimators']
    assert info['inputs'] == ['train']
    assert info['outputs'] == ['model', 'result']


def test_describe_node_unknown_type_raises_actionable_error(ctl):
    with pytest.raises(ValueError) as exc:
        describe_node(ctl, 'not.a.real.type')
    assert 'not.a.real.type' in str(exc.value)
    # Actionable error mentions list_nodes as next step.
    assert 'list_nodes' in str(exc.value)


def test_search_nodes_keyword_hits_summary_and_name(ctl):
    # 'classifier' should match the RF summary text.
    out = search_nodes(ctl, 'classifier')
    assert any(n['name'] == 'Random Forest' for n in out)


def test_search_nodes_case_insensitive(ctl):
    out = search_nodes(ctl, 'SMILES')
    assert any(n['name'] == 'MolTable Reader' for n in out)


def test_search_nodes_returns_at_most_top_k(ctl):
    # 'a' is in every name/summary; top_k=2 should cap.
    out = search_nodes(ctl, 'a', top_k=2)
    assert len(out) == 2


def test_search_nodes_empty_query_returns_empty(ctl):
    assert search_nodes(ctl, '') == []
    assert search_nodes(ctl, '   ') == []
