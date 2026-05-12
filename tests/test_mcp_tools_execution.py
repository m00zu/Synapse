"""Tests for execution tools."""
import pytest

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

from synapse.mcp.controller import FakeGraphController, NodeInfo
from synapse.mcp.tools.execution import run_node, get_node_status, get_node_output


@pytest.fixture
def ctl():
    c = FakeGraphController(registered=[
        NodeInfo('cat', 'Foo', 'cat.Foo', [], [], ['out'], 'docs'),
    ])
    return c


# ── Helper fixture: 100-row DataFrame for mode tests ─────────────────────────

@pytest.fixture
def df_node(ctl):
    """Return (ctl, node_id, df) — a node with a pre-seeded 100-row DataFrame."""
    if not _HAS_PANDAS:
        pytest.skip("pandas not available")
    df = pd.DataFrame({
        'area':  list(range(100)),
        'score': [i * 0.5 for i in range(100)],
        'label': ['a' if i % 2 else 'b' for i in range(100)],
    })
    nid = ctl.add_node('cat.Foo')
    ctl.set_output(nid, 'out', df)
    return ctl, nid, df


# ── Existing tests (unchanged) ────────────────────────────────────────────────

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


# ── New mode tests ────────────────────────────────────────────────────────────

def test_get_node_output_describe_mode(df_node):
    ctl, nid, _ = df_node
    out = get_node_output(ctl, nid, mode='describe')
    assert out['kind'] == 'describe'
    assert out['n_rows'] == 100
    assert 'area' in out['summary']
    assert 'mean' in out['summary']['area']


def test_get_node_output_range_mode(df_node):
    ctl, nid, _ = df_node
    out = get_node_output(ctl, nid, mode='range', start=10, end=15)
    assert out['kind'] == 'range'
    assert out['n_returned'] == 5
    assert out['rows'][0]['area'] == 10
    assert out['rows'][-1]['area'] == 14


def test_get_node_output_range_caps_at_500(df_node):
    ctl, nid, _ = df_node
    big = pd.DataFrame({'i': range(2000)})
    ctl.set_output(nid, 'out', big)
    out = get_node_output(ctl, nid, mode='range', start=0, end=2000)
    assert out['n_returned'] == 500   # capped


def test_get_node_output_columns_mode(df_node):
    ctl, nid, _ = df_node
    out = get_node_output(ctl, nid,
                          mode='columns', columns=['area', 'label'],
                          sample=5)
    assert out['kind'] == 'columns'
    assert out['columns'] == ['area', 'label']
    assert out['n_returned'] == 5
    assert set(out['head'][0].keys()) == {'area', 'label'}


def test_get_node_output_columns_unknown_raises_actionable(df_node):
    ctl, nid, _ = df_node
    with pytest.raises(ValueError) as exc:
        get_node_output(ctl, nid,
                        mode='columns', columns=['nope'])
    msg = str(exc.value)
    assert 'nope' in msg
    assert 'area' in msg   # Available list mentioned.


def test_get_node_output_filter_mode(df_node):
    ctl, nid, _ = df_node
    out = get_node_output(ctl, nid, mode='filter',
                          query='area >= 50 and label == "b"')
    assert out['kind'] == 'filter'
    assert out['n_matched'] > 0
    for row in out['rows']:
        assert row['area'] >= 50
        assert row['label'] == 'b'


def test_get_node_output_filter_bad_query_raises_actionable(df_node):
    ctl, nid, _ = df_node
    with pytest.raises(ValueError) as exc:
        get_node_output(ctl, nid,
                        mode='filter', query='not valid python @#$')
    assert 'Invalid filter query' in str(exc.value)


def test_get_node_output_full_mode(df_node):
    ctl, nid, _ = df_node
    out = get_node_output(ctl, nid, mode='full')
    assert out['kind'] == 'full'
    assert out['n_rows'] == 100
    assert len(out['rows']) == 100


def test_get_node_output_full_refuses_huge_table(df_node):
    ctl, nid, _ = df_node
    huge = pd.DataFrame({'i': range(6000)})
    ctl.set_output(nid, 'out', huge)
    with pytest.raises(ValueError) as exc:
        get_node_output(ctl, nid, mode='full')
    assert 'filter' in str(exc.value).lower() or 'range' in str(exc.value).lower()


def test_get_node_output_unsupported_mode_on_scalar(ctl):
    nid = ctl.add_node('cat.Foo')
    ctl.set_output(nid, 'out', 42)
    out = get_node_output(ctl, nid, mode='describe')
    assert out['kind'] == 'unsupported_mode'
    assert out['mode'] == 'describe'


def test_get_node_output_unknown_mode_raises(ctl):
    nid = ctl.add_node('cat.Foo')
    ctl.set_output(nid, 'out', 'x')  # any payload — caught before dispatch
    with pytest.raises(ValueError) as exc:
        get_node_output(ctl, nid, mode='whoops')
    assert 'whoops' in str(exc.value)
