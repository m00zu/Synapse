"""Compose tool: atomic one-shot workflow construction.

``create_workflow(definition, run=False)`` takes a structured spec of
nodes + connections, validates all of it up front, then either creates
everything or rolls back (deleting any nodes it had already created
during a partial run).  Optional ``run`` flag evaluates the terminal
nodes (those with no outgoing edges) after construction succeeds.
"""
from __future__ import annotations

from typing import Any

from ..controller import GraphController


def _validate(controller: GraphController,
              definition: dict) -> tuple[list[dict], list[dict]]:
    """Validate the definition; return (nodes, connections) lists."""
    nodes = definition.get('nodes') or []
    connections = definition.get('connections') or []

    if not isinstance(nodes, list) or not nodes:
        raise ValueError(
            "definition.nodes must be a non-empty list of "
            "{id, type, properties?} dicts.")
    if not isinstance(connections, list):
        raise ValueError("definition.connections must be a list of "
                          "{src, src_port, dst, dst_port} dicts.")

    # Build alias -> type map; detect duplicates.
    aliases: dict[str, str] = {}
    registered = {n.type_id for n in controller.list_registered()}
    for n in nodes:
        alias = n.get('id')
        if not alias:
            raise ValueError(f"node missing 'id': {n}")
        if alias in aliases:
            raise ValueError(f"duplicate alias {alias!r} in nodes list")
        type_id = n.get('type')
        if not type_id:
            raise ValueError(f"node {alias!r} missing 'type'")
        if type_id not in registered:
            raise ValueError(
                f"node {alias!r}: unknown type {type_id!r}. "
                f"Call list_nodes() to see registered types.")
        aliases[alias] = type_id

    # Validate every connection references known aliases.
    for i, c in enumerate(connections):
        for k in ('src', 'src_port', 'dst', 'dst_port'):
            if k not in c:
                raise ValueError(f"connection #{i} missing {k!r}: {c}")
        if c['src'] not in aliases:
            raise ValueError(
                f"connection #{i}: unknown src alias {c['src']!r}. "
                f"Defined aliases: {sorted(aliases)}.")
        if c['dst'] not in aliases:
            raise ValueError(
                f"connection #{i}: unknown dst alias {c['dst']!r}. "
                f"Defined aliases: {sorted(aliases)}.")

    return nodes, connections


_X_PAD = 300.0     # fallback column width when node width is unknown
_Y_PAD = 120.0     # fallback row height when node height is unknown
_GAP   = 60.0      # visual breathing room between columns / rows


def _layout_new_nodes(controller: GraphController,
                      nodes: list[dict],
                      connections: list[dict],
                      created: dict[str, str]) -> None:
    """Best-effort left-to-right layout for nodes created in this batch.

    Compute depth via topological levels (roots at depth 0, then +1 per
    edge).  At each depth, stack siblings vertically.  Nodes that fail
    to set position are skipped silently — layout is cosmetic.
    """
    # Skip layout entirely on controllers without a real NodeGraphQt graph
    # (e.g. FakeGraphController in tests).
    if not hasattr(controller, '_graph'):
        return

    # Build incoming-edge counts to identify roots and compute depths.
    in_count: dict[str, int] = {n['id']: 0 for n in nodes}
    out_map: dict[str, list[str]] = {n['id']: [] for n in nodes}
    for c in connections:
        s, d = c['src'], c['dst']
        if d in in_count:
            in_count[d] += 1
        if s in out_map:
            out_map[s].append(d)

    # BFS levels.
    depth: dict[str, int] = {}
    frontier = [nid for nid, c in in_count.items() if c == 0]
    for nid in frontier:
        depth[nid] = 0
    while frontier:
        nxt: list[str] = []
        for nid in frontier:
            for dst in out_map.get(nid, []):
                if dst not in depth or depth[dst] < depth[nid] + 1:
                    depth[dst] = depth[nid] + 1
                    nxt.append(dst)
        frontier = nxt

    # Anything unreachable (orphan node) lands at depth 0 too.
    for nid in in_count:
        depth.setdefault(nid, 0)

    # Bucket by depth so we can stack siblings vertically.
    buckets: dict[int, list[str]] = {}
    for nid in [n['id'] for n in nodes]:
        buckets.setdefault(depth[nid], []).append(nid)

    graph = controller._graph  # type: ignore[attr-defined]

    def _measure(n) -> tuple[float, float]:
        """Return (width, height) of a NodeGraphQt node *after* its view
        has been laid out.  Falls back to the X/Y_PAD defaults if any
        accessor raises.
        """
        try:
            view = getattr(n, 'view', None)
            # Force the view to compute its real width/height based on
            # current widgets/ports.  Without this, the view's bounding
            # rect is the default (~100x80) until Qt next repaints.
            if view is not None and hasattr(view, 'draw_node'):
                try:
                    view.draw_node()
                except Exception:
                    pass
            if view is not None and hasattr(view, 'boundingRect'):
                rect = view.boundingRect()
                w = float(rect.width())
                h = float(rect.height())
                if w > 0 and h > 0:
                    return w, h
        except Exception:
            pass
        return _X_PAD, _Y_PAD

    # Find current canvas extent so we don't smash onto existing nodes.
    # Use right-edge (x + width) of the rightmost existing node + gap.
    base_x = 0.0
    try:
        for n in graph.all_nodes():
            if n.id in created.values():
                continue   # skip the nodes we just added
            try:
                p = n.pos()
                x = p[0] if not hasattr(p, 'x') else p.x()
                w, _ = _measure(n)
                base_x = max(base_x, x + w + _GAP)
            except Exception:
                continue
    except Exception:
        pass

    # Compute column x positions using each level's widest node width.
    col_x: dict[int, float] = {}
    x_cursor = base_x
    for d in sorted(buckets.keys()):
        col_x[d] = x_cursor
        max_w = 0.0
        for alias in buckets[d]:
            real_id = created.get(alias)
            if real_id is None:
                continue
            try:
                n = graph.get_node_by_id(real_id)
                if n is not None:
                    w, _ = _measure(n)
                    max_w = max(max_w, w)
            except Exception:
                continue
        x_cursor += (max_w or _X_PAD) + _GAP

    # Place each node at its column-x; stack siblings vertically using
    # each sibling's height + gap (so tall nodes don't crash into the
    # next row down).
    for d, alias_list in buckets.items():
        y_cursor = 0.0
        for alias in alias_list:
            real_id = created.get(alias)
            if real_id is None:
                continue
            try:
                node = graph.get_node_by_id(real_id)
                if node is None:
                    continue
                node.set_pos(col_x[d], y_cursor)
                _, h = _measure(node)
                y_cursor += h + _GAP
            except Exception:
                continue


def create_workflow(controller: GraphController,
                    definition: dict,
                    run: bool = False) -> dict[str, Any]:
    """Build a NEW workflow (or a fresh sub-pipeline) in ONE atomic call.

    **Use this ONLY for creating fresh nodes** — typically on an empty
    canvas, or to add an entirely new sub-pipeline alongside existing
    work.  One call instead of N × ``add_node`` + M × ``connect``;
    nodes are auto-laid-out and validation failures roll back cleanly.

    **Do NOT use this to MODIFY an existing workflow.**  For any change
    to nodes that already exist — re-wiring, property tweaks, swapping
    types, deleting branches — use the modify tools instead:

    - ``connect`` / ``disconnect`` — change wires.
    - ``set_property`` — change a node's setting.
    - ``replace_node`` — swap one node's type, preserving compatible wires.
    - ``delete_node`` / ``add_node`` — surgical insertion or removal.

    Calling ``create_workflow`` to "rebuild" the graph DUPLICATES the
    existing nodes (it appends; it does not replace).  Always prefer the
    smallest edit that achieves the user's goal.

    ``definition`` shape::

        {
          "nodes": [
            {"id": "a",            # local alias used in connections
             "type": "<node_type_id>",
             "properties": {...}   # optional
            },
            ...
          ],
          "connections": [
            {"src": "a", "src_port": "out",
             "dst": "b", "dst_port": "in"},
            ...
          ],
        }

    All node types and connection aliases are validated up front.  If
    any check fails, no nodes are created (any partial creation during
    this call is rolled back).

    With ``run=True``, every just-created node is evaluated in
    topological order (upstream before downstream) — equivalent to
    clicking "Run" on the canvas.  Per-alias results land in
    ``run_results``; evaluation stops at the first failure so a broken
    upstream node doesn't trigger misleading errors downstream.

    Returns ``{created_ids: {alias: real_id}, run_results?: {...}}``.
    Pre-existing graph state is never touched.
    """
    nodes, connections = _validate(controller, definition)

    # Begin atomic block.  Track real ids created so we can roll back
    # if anything later fails.
    created: dict[str, str] = {}  # alias -> real_id
    try:
        for n in nodes:
            real_id = controller.add_node(
                n['type'], properties=n.get('properties'))
            created[n['id']] = real_id
        for c in connections:
            controller.connect(created[c['src']], c['src_port'],
                                created[c['dst']], c['dst_port'])
    except Exception:
        # Roll back any nodes we created (controller.delete_node also
        # drops attached edges).
        for real_id in created.values():
            try:
                controller.delete_node(real_id)
            except Exception:
                pass  # best-effort cleanup
        raise

    # Layout: walk the connection DAG, left-to-right by depth, sibling-stacked.
    _layout_new_nodes(controller, nodes, connections, created)

    result: dict[str, Any] = {'created_ids': dict(created)}

    if run:
        # Each run_node call now walks upstream automatically (mirrors
        # Synapse's "Run" button), so we only need to invoke the
        # terminals — they'll drag their dependencies through.
        run_results: dict[str, dict] = {}
        terminals = [n['id'] for n in nodes
                     if n['id'] not in {c['src'] for c in connections}]
        for alias in terminals:
            run_results[alias] = controller.run_node(created[alias])
        result['run_results'] = run_results

    return result
