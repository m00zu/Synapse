"""Tests for execution tools."""
import pytest

from synapse.mcp.controller import FakeGraphController, NodeInfo
from synapse.mcp.tools.execution import run_node, get_node_status, get_node_output


@pytest.fixture
def ctl():
    c = FakeGraphController(registered=[
        NodeInfo('cat', 'Foo', 'cat.Foo', [], [], ['out'], 'docs'),
    ])
    return c


def test_run_node_success_round_trip(ctl):
    nid = ctl.add_node('cat.Foo')
    ctl.set_run_result(nid, success=True, message='ok', duration_ms=42.0)
    out = run_node(ctl, nid)
    assert out == {'success': True, 'message': 'ok', 'duration_ms': 42.0}


def test_run_node_failure_returned_not_raised(ctl):
    nid = ctl.add_node('cat.Foo')
    ctl.set_run_result(nid, success=False, message='boom', duration_ms=5.0)
    out = run_node(ctl, nid)
    assert out['success'] is False
    assert out['message'] == 'boom'


def test_run_node_unknown_id_raises_actionable(ctl):
    with pytest.raises(ValueError) as exc:
        run_node(ctl, 'nonsense')
    assert 'nonsense' in str(exc.value)
    assert 'describe_graph' in str(exc.value)


def test_get_node_status_reads_record(ctl):
    nid = ctl.add_node('cat.Foo')
    # Default: pending, no message
    status = get_node_status(ctl, nid)
    assert status['node_id'] == nid
    assert status['status'] == 'pending'
    assert status['last_message'] is None


def test_get_node_output_dataframe(ctl):
    import pandas as pd
    nid = ctl.add_node('cat.Foo')
    df = pd.DataFrame({'area': [42, 17, 99], 'mean': [1.1, 2.2, 3.3]})
    ctl.set_output(nid, 'result', df)
    out = get_node_output(ctl, nid, port_name='result')
    assert out['kind'] == 'table'
    assert out['n_rows'] == 3
    assert out['columns'] == ['area', 'mean']
    assert out['head'][0]['area'] == 42


def test_get_node_output_scalar(ctl):
    nid = ctl.add_node('cat.Foo')
    ctl.set_output(nid, 'count', 42)
    out = get_node_output(ctl, nid, port_name='count')
    assert out['kind'] == 'scalar'
    assert out['value'] == 42


def test_get_node_output_unknown_node_raises_actionable(ctl):
    with pytest.raises(ValueError) as exc:
        get_node_output(ctl, 'nonsense', port_name='out')
    assert 'nonsense' in str(exc.value)
    assert 'describe_graph' in str(exc.value)


def test_get_node_output_port_not_evaluated_raises_actionable(ctl):
    nid = ctl.add_node('cat.Foo')
    with pytest.raises(ValueError) as exc:
        get_node_output(ctl, nid, port_name='out')
    # Error should hint at run_node().
    assert 'run_node' in str(exc.value).lower()
