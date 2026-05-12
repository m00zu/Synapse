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


def _terminal_aliases(nodes: list[dict],
                      connections: list[dict]) -> list[str]:
    """Aliases with no outgoing connections — these are the run targets."""
    has_outgoing = {c['src'] for c in connections}
    return [n['id'] for n in nodes if n['id'] not in has_outgoing]


_X_PAD = 300.0
_Y_PAD = 120.0


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

    # Find current canvas extent so we don't smash onto existing nodes.
    base_x = 0.0
    try:
        for n in controller._graph.all_nodes():  # type: ignore[attr-defined]
            real_id = controller.get_node(n.id).id
            if real_id in created.values():
                continue   # skip the nodes we just added
            try:
                p = n.pos()
                x = p[0] if not hasattr(p, 'x') else p.x()
                base_x = max(base_x, x + _X_PAD)
            except Exception:
                continue
    except Exception:
        pass  # If we can't introspect, just start at 0,0.

    for d, alias_list in buckets.items():
        for i, alias in enumerate(alias_list):
            real_id = created.get(alias)
            if real_id is None:
                continue
            try:
                node = controller._graph.get_node_by_id(real_id)  # type: ignore[attr-defined]
                if node is not None:
                    node.set_pos(base_x + d * _X_PAD, i * _Y_PAD)
            except Exception:
                pass


def create_workflow(controller: GraphController,
                    definition: dict,
                    run: bool = False) -> dict[str, Any]:
    """Build a whole workflow in ONE call — preferred over add_node + connect.

    **Use this whenever you're building two or more nodes that share
    connections.**  It's one atomic call instead of N add_node + M connect
    round-trips, the nodes get auto-laid-out so they don't overlap on the
    canvas, and any validation failure rolls back cleanly (no half-built
    graphs left behind).

    Only fall back to ``add_node`` / ``connect`` for surgical edits to an
    existing graph (e.g. "add a Murcko Scaffold step between nodes X and Y").

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

    With ``run=True``, terminal nodes (no outgoing edges) are evaluated
    after creation; per-alias results are returned in ``run_results``.

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
        run_results: dict[str, dict] = {}
        for alias in _terminal_aliases(nodes, connections):
            run_results[alias] = controller.run_node(created[alias])
        result['run_results'] = run_results

    return result
