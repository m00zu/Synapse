"""Tests for the auto-layout algorithm (synapse/layout.py).

Tests use simple dicts to model the graph -- no Qt -- so the algorithm
is verified in isolation.  The Qt wrapper (``auto_organize``) is just
a thin adapter over ``compute_layout`` and isn't exercised here.
"""
from __future__ import annotations

import pytest

from synapse.layout import compute_layout, _DEFAULT_NODE_SIZE, _DEFAULT_COL_PADDING


def _make_graph(edges: list[tuple[str, str]], extra_nodes: list[str] | None = None):
    """Build a tiny graph from a list of edges + optional isolated nodes.

    Returns (node_ids, parents_fn, children_fn) suitable for
    ``compute_layout``.
    """
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {}
    all_nodes: set[str] = set()
    for src, dst in edges:
        all_nodes.add(src)
        all_nodes.add(dst)
        children.setdefault(src, []).append(dst)
        parents.setdefault(dst, []).append(src)
    for n in extra_nodes or []:
        all_nodes.add(n)
    nodes = sorted(all_nodes)
    return (
        nodes,
        lambda nid: parents.get(nid, []),
        lambda nid: children.get(nid, []),
    )


def _col_of(x: float) -> int:
    """Return the integer column index for an x-position.

    Layout uses variable column widths (each column = max-width of its
    nodes + col_padding).  With default node sizes the column stride
    is ``default_node_width + col_padding``.
    """
    stride = _DEFAULT_NODE_SIZE[0] + _DEFAULT_COL_PADDING
    return round(x / stride)


# ── Basic shapes ────────────────────────────────────────────────────────


def test_empty_graph():
    """An empty input returns an empty output."""
    assert compute_layout([], lambda _: [], lambda _: []) == {}


def test_single_node():
    """One node sits at (0, 0)."""
    nodes, p, c = _make_graph([], extra_nodes=['only'])
    pos = compute_layout(nodes, p, c)
    assert pos['only'] == (0.0, 0.0)


def test_linear_chain():
    """A -> B -> C -> D should be a single row, columns 0, 1, 2, 3."""
    nodes, p, c = _make_graph([('A', 'B'), ('B', 'C'), ('C', 'D')])
    pos = compute_layout(nodes, p, c)
    assert _col_of(pos['A'][0]) == 0
    assert _col_of(pos['B'][0]) == 1
    assert _col_of(pos['C'][0]) == 2
    assert _col_of(pos['D'][0]) == 3
    # All on the same row (within a row-height tolerance).
    ys = {pos[n][1] for n in 'ABCD'}
    assert len(ys) == 1, f"Linear chain should share one row, got y values {ys}"


# ── Fan-in: the user's headline example ─────────────────────────────────


def test_fan_in_two_readers_into_concat():
    """The user's 'two CSV readers -> concat' case.

    Both readers in column 0, concat in column 1.  The concat sits
    at the midpoint y of the two readers (barycenter).
    """
    nodes, p, c = _make_graph([('reader1', 'concat'),
                                ('reader2', 'concat')])
    pos = compute_layout(nodes, p, c)
    # Columns.
    assert _col_of(pos['reader1'][0]) == 0
    assert _col_of(pos['reader2'][0]) == 0
    assert _col_of(pos['concat'][0]) == 1
    # The two readers must be at different y (stacked).
    assert pos['reader1'][1] != pos['reader2'][1]
    # The concat's y is between the two readers (centered via barycenter).
    r1, r2 = sorted([pos['reader1'][1], pos['reader2'][1]])
    assert r1 <= pos['concat'][1] <= r2, (
        f"Concat y={pos['concat'][1]} should sit between "
        f"reader y values [{r1}, {r2}]")


# ── Fan-out ─────────────────────────────────────────────────────────────


def test_fan_out_centered_source():
    """A -> B, A -> C: A should be centered between B and C vertically."""
    nodes, p, c = _make_graph([('A', 'B'), ('A', 'C')])
    pos = compute_layout(nodes, p, c)
    assert _col_of(pos['A'][0]) == 0
    assert _col_of(pos['B'][0]) == 1
    assert _col_of(pos['C'][0]) == 1
    b_y, c_y = sorted([pos['B'][1], pos['C'][1]])
    assert b_y <= pos['A'][1] <= c_y, (
        f"A y={pos['A'][1]} should sit between B/C y [{b_y}, {c_y}]")


# ── Diamond ─────────────────────────────────────────────────────────────


def test_diamond_a_to_bc_to_d():
    """A -> B -> D, A -> C -> D.

    Columns 0/1/1/2.  A and D should be centered between B and C.
    """
    nodes, p, c = _make_graph([
        ('A', 'B'), ('A', 'C'),
        ('B', 'D'), ('C', 'D'),
    ])
    pos = compute_layout(nodes, p, c)
    assert _col_of(pos['A'][0]) == 0
    assert _col_of(pos['B'][0]) == 1
    assert _col_of(pos['C'][0]) == 1
    assert _col_of(pos['D'][0]) == 2
    b_y, c_y = sorted([pos['B'][1], pos['C'][1]])
    assert b_y <= pos['A'][1] <= c_y
    assert b_y <= pos['D'][1] <= c_y


# ── Disconnected subgraphs ──────────────────────────────────────────────


def test_two_disconnected_chains_stack_vertically():
    """Two unrelated DAGs should be stacked in y (no horizontal overlap)."""
    nodes, p, c = _make_graph([
        ('A1', 'B1'), ('B1', 'C1'),    # first chain
        ('A2', 'B2'), ('B2', 'C2'),    # second chain
    ])
    pos = compute_layout(nodes, p, c)
    # Both chains use columns 0/1/2.
    for label in '12':
        assert _col_of(pos[f'A{label}'][0]) == 0
        assert _col_of(pos[f'B{label}'][0]) == 1
        assert _col_of(pos[f'C{label}'][0]) == 2
    # Second chain sits BELOW the first (greater y).
    first_max_y = max(pos[n][1] for n in ('A1', 'B1', 'C1'))
    second_min_y = min(pos[n][1] for n in ('A2', 'B2', 'C2'))
    assert second_min_y > first_max_y, (
        f"Second chain should be below the first; got "
        f"first_max_y={first_max_y}, second_min_y={second_min_y}")


def test_isolated_node_among_chain():
    """A floating node (no edges) goes in its own subgraph below."""
    nodes, p, c = _make_graph([('A', 'B'), ('B', 'C')],
                              extra_nodes=['lone'])
    pos = compute_layout(nodes, p, c)
    chain_max_y = max(pos[n][1] for n in 'ABC')
    assert pos['lone'][1] > chain_max_y


# ── Overlap-safe stacking with variable heights ─────────────────────────


def test_variable_widths_no_horizontal_overlap():
    """A wide node in column 0 must not horizontally overlap its
    downstream neighbour in column 1.

    Regression: the original implementation used a fixed column width
    of 320 px, so a 500-px-wide ROI/viewer node would spill into the
    next column.  Now column x is computed from per-column max widths.
    """
    nodes, p, c = _make_graph([('big', 'next')])
    sizes = {
        'big':  (600.0, 100.0),   # very wide upstream
        'next': (250.0, 100.0),
    }
    pos = compute_layout(nodes, p, c, size_fn=lambda nid: sizes[nid])
    big_right = pos['big'][0] + sizes['big'][0]
    next_left = pos['next'][0]
    assert next_left >= big_right, (
        f"Column 1 must start at or after column 0's right edge: "
        f"big_right={big_right}, next_left={next_left}")


def test_variable_heights_no_overlap():
    """Nodes of different heights stacked in the same column don't overlap."""
    nodes, p, c = _make_graph([
        ('R1', 'sink'), ('R2', 'sink'), ('R3', 'sink'),
    ])
    # Custom sizes: R2 is twice as tall as R1/R3.
    sizes = {
        'R1': (250.0, 100.0),
        'R2': (250.0, 200.0),
        'R3': (250.0, 100.0),
        'sink': (250.0, 100.0),
    }
    pos = compute_layout(nodes, p, c, size_fn=lambda nid: sizes[nid])
    # All three readers in column 0.
    for r in ('R1', 'R2', 'R3'):
        assert _col_of(pos[r][0]) == 0
    # Check non-overlap: sort by y, ensure each one starts after the
    # previous ends.
    by_y = sorted(['R1', 'R2', 'R3'], key=lambda nid: pos[nid][1])
    prev_bottom = -float('inf')
    for r in by_y:
        top = pos[r][1]
        bottom = top + sizes[r][1]
        assert top >= prev_bottom, (
            f"{r} top={top} overlaps previous bottom={prev_bottom}")
        prev_bottom = bottom


# ── Sanity: order stability ─────────────────────────────────────────────


def test_layout_is_deterministic():
    """Running compute_layout twice with the same input returns the same output."""
    nodes, p, c = _make_graph([
        ('A', 'B'), ('B', 'C'),
        ('X', 'B'),     # extra parent on B
        ('B', 'D'), ('B', 'E'),
    ])
    p1 = compute_layout(nodes, p, c)
    p2 = compute_layout(nodes, p, c)
    assert p1 == p2
