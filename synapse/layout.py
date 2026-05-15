"""Auto-layout algorithm for DAG-style node graphs.

The core function ``compute_layout`` takes a graph by accessor functions
(rather than a NodeGraphQt object) so it's pure-Python and testable
without Qt.  The Qt wrapper ``auto_organize`` applies the result to a
running NodeGraph and wraps the move in an undo macro.

Algorithm:

  1. **Connected components**: nodes that aren't reachable from each
     other go in separate sub-layouts, stacked vertically.
  2. **X from topological depth**: roots at column 0, then each node at
     ``1 + max(depth(parents))``.
  3. **Y from barycenter heuristic**: a node's y is the average of its
     neighbours' y, refined by alternating left->right and right->left
     sweeps for a handful of iterations.
  4. **Overlap-safe stacking**: within each column, nodes are sorted by
     their barycenter-computed y, then placed top-to-bottom using each
     node's actual height (no overlaps even when node sizes vary).

Conventions:

  - X grows rightwards; Y grows downwards (matches Qt's coordinate
    system).
  - The leftmost root sits at ``x = 0``; the topmost node sits at
    ``y = 0``.  The caller can shift the result to fit the viewport.
  - Disconnected sub-graphs are stacked top-to-bottom with a fixed
    padding between them.
"""
from __future__ import annotations

from typing import Callable, Iterable, TypeVar


NodeId = TypeVar("NodeId")

# Defaults tuned for typical Synapse node card sizes.
# Column x positions are computed from per-column max widths plus
# ``col_padding`` (not a fixed column width) so that nodes carrying
# big embedded widgets -- 3D viewers, table previews, ROI canvases --
# get the horizontal space they need without overlapping neighbours.
_DEFAULT_COL_PADDING = 80.0
_DEFAULT_ROW_PADDING = 60.0
_DEFAULT_SUBGRAPH_PADDING = 120.0
_DEFAULT_NODE_SIZE = (250.0, 100.0)
_BARYCENTER_ITERATIONS = 5


# ── Public API ─────────────────────────────────────────────────────────

def compute_layout(
    node_ids: Iterable[NodeId],
    parents_fn: Callable[[NodeId], list[NodeId]],
    children_fn: Callable[[NodeId], list[NodeId]],
    size_fn: Callable[[NodeId], tuple[float, float]] | None = None,
    *,
    col_padding: float = _DEFAULT_COL_PADDING,
    row_padding: float = _DEFAULT_ROW_PADDING,
    subgraph_padding: float = _DEFAULT_SUBGRAPH_PADDING,
) -> dict[NodeId, tuple[float, float]]:
    """Compute ``(x, y)`` positions for every node in a DAG.

    ``parents_fn`` and ``children_fn`` return the upstream / downstream
    neighbours of a given node ID.  ``size_fn`` returns ``(width,
    height)`` in pixels for a node (used for overlap-safe stacking);
    when not supplied, a constant default is used.

    The output is a flat dict ``{node_id: (x, y)}``.  Disconnected
    components are stacked vertically with ``subgraph_padding``
    between them.
    """
    nodes = list(node_ids)
    if not nodes:
        return {}
    if size_fn is None:
        size_fn = lambda _nid: _DEFAULT_NODE_SIZE

    positions: dict[NodeId, tuple[float, float]] = {}
    y_offset = 0.0
    for component in _connected_components(nodes, parents_fn, children_fn):
        comp_pos = _layout_one_component(
            component, parents_fn, children_fn, size_fn,
            col_padding=col_padding, row_padding=row_padding,
        )
        # Shift this component down by ``y_offset`` and track its
        # bottom edge for the next component.
        max_bottom = 0.0
        for nid, (x, y) in comp_pos.items():
            shifted_y = y + y_offset
            positions[nid] = (x, shifted_y)
            _, h = size_fn(nid)
            max_bottom = max(max_bottom, shifted_y + h)
        y_offset = max_bottom + subgraph_padding

    return positions


# ── Internals ──────────────────────────────────────────────────────────

def _connected_components(
    nodes: list[NodeId],
    parents_fn: Callable[[NodeId], list[NodeId]],
    children_fn: Callable[[NodeId], list[NodeId]],
) -> list[list[NodeId]]:
    """Group nodes into connected components (edges treated as undirected).

    Components are returned in input order of their first-encountered
    node so the visible result is stable across runs.
    """
    node_set = set(nodes)
    visited: set[NodeId] = set()
    components: list[list[NodeId]] = []
    for start in nodes:
        if start in visited:
            continue
        component: list[NodeId] = []
        # BFS through both upstream + downstream neighbours.
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            component.append(n)
            for neighbour in parents_fn(n) + children_fn(n):
                if neighbour in node_set and neighbour not in visited:
                    stack.append(neighbour)
        components.append(component)
    return components


def _layout_one_component(
    nodes: list[NodeId],
    parents_fn: Callable[[NodeId], list[NodeId]],
    children_fn: Callable[[NodeId], list[NodeId]],
    size_fn: Callable[[NodeId], tuple[float, float]],
    *,
    col_padding: float,
    row_padding: float,
) -> dict[NodeId, tuple[float, float]]:
    """Lay out one connected DAG component."""
    node_set = set(nodes)

    # 1. Topological depth -> x column.
    depths = _compute_depths(nodes, parents_fn, node_set)
    max_depth = max(depths.values()) if depths else 0
    columns: list[list[NodeId]] = [[] for _ in range(max_depth + 1)]
    for n in nodes:
        columns[depths[n]].append(n)

    # 2. Initial y per column (arbitrary but deterministic).
    y_pos: dict[NodeId, float] = {}
    for col in columns:
        for i, n in enumerate(col):
            y_pos[n] = float(i)

    # 3. Barycenter sweeps.  Alternating directions, ~5 iterations is
    # plenty for typical scientific workflow sizes.
    for _ in range(_BARYCENTER_ITERATIONS):
        # Left -> right: pull child toward parents' mean y.
        for d in range(1, max_depth + 1):
            for n in columns[d]:
                parents = [p for p in parents_fn(n) if p in node_set]
                if parents:
                    y_pos[n] = sum(y_pos[p] for p in parents) / len(parents)
        # Right -> left: pull parent toward children's mean y.
        for d in range(max_depth - 1, -1, -1):
            for n in columns[d]:
                children = [c for c in children_fn(n) if c in node_set]
                if children:
                    y_pos[n] = sum(y_pos[c] for c in children) / len(children)

    # 4. Compute per-column x positions from each column's max node width.
    # Fixed-width columns caused horizontal overlap for graphs with wide
    # nodes (e.g. ROI mask viewer, table preview, 3D volume viewer).
    col_max_w = [
        max((size_fn(n)[0] for n in col), default=0.0)
        for col in columns
    ]
    col_x: list[float] = [0.0]
    for w in col_max_w[:-1]:
        col_x.append(col_x[-1] + w + col_padding)

    # 5. Overlap-safe stacking within each column.  Sort by computed
    # y, then place top-to-bottom respecting actual node heights.
    positions: dict[NodeId, tuple[float, float]] = {}
    for d, col in enumerate(columns):
        col_sorted = sorted(col, key=lambda n: y_pos[n])
        cur_y = 0.0
        for n in col_sorted:
            _, h = size_fn(n)
            positions[n] = (col_x[d], cur_y)
            cur_y += h + row_padding

    return positions


def _compute_depths(
    nodes: list[NodeId],
    parents_fn: Callable[[NodeId], list[NodeId]],
    node_set: set[NodeId],
) -> dict[NodeId, int]:
    """Topological depth from roots, using memoised recursion."""
    depths: dict[NodeId, int] = {}

    def depth(n: NodeId) -> int:
        if n in depths:
            return depths[n]
        parents = [p for p in parents_fn(n) if p in node_set]
        result = 0 if not parents else 1 + max(depth(p) for p in parents)
        depths[n] = result
        return result

    for n in nodes:
        depth(n)
    return depths


# ── Qt wrapper ─────────────────────────────────────────────────────────

def auto_organize(graph, *, push_undo: bool = True,
                  center_view: bool = True) -> int:
    """Re-position every node in ``graph`` for left-to-right readability.

    The whole canvas is laid out -- this is the explicit "re-organize
    everything" action.  Wraps the moves in an undo macro so the user
    can ``Cmd+Z`` to restore prior positions.

    Returns the number of nodes that were moved.
    """
    nodes = list(graph.all_nodes())
    if not nodes:
        return 0
    id_to_node = {n.id: n for n in nodes}

    def parents_fn(nid):
        node = id_to_node[nid]
        result = []
        for p in node.input_ports():
            for cp in p.connected_ports():
                result.append(cp.node().id)
        return result

    def children_fn(nid):
        node = id_to_node[nid]
        result = []
        for p in node.output_ports():
            for cp in p.connected_ports():
                result.append(cp.node().id)
        return result

    def size_fn(nid):
        """Return the rendered (width, height) of a node.

        NodeGraphQt's ``boundingRect()`` returns the default (~100x80)
        until the view has been laid out for the current widget/port
        set.  Force a ``draw_node()`` first so we measure the actual
        on-screen size -- without this, the result is too small and
        adjacent nodes end up slightly overlapping.
        """
        node = id_to_node[nid]
        try:
            view = getattr(node, 'view', None)
            if view is not None and hasattr(view, 'draw_node'):
                try:
                    view.draw_node()
                except Exception:
                    pass
            if view is not None and hasattr(view, 'boundingRect'):
                br = view.boundingRect()
                w, h = float(br.width()), float(br.height())
                if w > 0 and h > 0:
                    return (w, h)
        except Exception:
            pass
        return _DEFAULT_NODE_SIZE

    positions = compute_layout(
        node_ids=list(id_to_node.keys()),
        parents_fn=parents_fn,
        children_fn=children_fn,
        size_fn=size_fn,
    )

    undo_stack = None
    if push_undo:
        try:
            undo_stack = graph.undo_stack()
            undo_stack.beginMacro("Auto-organize layout")
        except Exception:
            undo_stack = None

    moved = 0
    for nid, (x, y) in positions.items():
        try:
            id_to_node[nid].set_pos(x, y)
            moved += 1
        except Exception:
            continue

    if undo_stack is not None:
        try:
            undo_stack.endMacro()
        except Exception:
            pass

    if center_view:
        try:
            graph.viewer().center_selection()
        except Exception:
            pass

    return moved
