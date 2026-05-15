"""Compose tool: one-shot workflow construction with partial-success semantics.

``create_workflow(definition, run=False)`` takes a structured spec of
nodes + connections, validates the node-level shape up front, then:

  1. Creates every node atomically (rolls back if any node creation
     fails -- almost never happens since type ids were validated).
  2. Attempts each connection independently.  A connection failure
     does NOT roll back -- the node is created, the failure is
     reported, and the LLM (or human) can fix wiring afterwards via
     the ``connect`` tool.

Connection failures are reported with the actual port names available
on the node, so the LLM has enough info to fix the wire in one
follow-up call without needing a separate ``describe_node``.  Port
name resolution is case-insensitive: ``src_port='Output'`` will
silently resolve to ``'output'`` if that's what the node exposes.

Optional ``run`` flag evaluates terminal nodes when ALL connections
succeed.  If any connection failed, ``run`` is skipped and a
``run_skipped`` note is included in the response.
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
    to set position are skipped silently -- layout is cosmetic.
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


def _resolve_port(name: str, available: list[str]) -> str | None:
    """Pick the best match for a port name; case-insensitive.

    Returns the actual port name to use, or None if no match.
    """
    if name in available:
        return name
    lower_map = {p.lower(): p for p in available}
    return lower_map.get(name.lower())


def _node_ports(controller: GraphController, node_id: str,
                ) -> tuple[list[str], list[str]]:
    """Return (input_ports, output_ports) for a node by id.

    Uses ``get_node`` + ``describe_registered`` so this works against
    both the live NodeGraphController and FakeGraphController without
    poking at internals.
    """
    record = controller.get_node(node_id)
    info = controller.describe_registered(record.type_id)
    return list(info.input_ports), list(info.output_ports)


def _port_types_of(controller: GraphController, node_id: str
                   ) -> dict[str, str]:
    """Return the {port-name: type-name} map for a live node, if any.

    Used to enrich connection-failure reports so the LLM knows the
    type of each available port, not just the name.  Falls back to an
    empty dict when the controller can't expose this (e.g.
    FakeGraphController in tests) -- the failure report just omits
    type info in that case.
    """
    # The NodeGraphController exposes the live NodeGraphQt node via a
    # private method get_node_by_id on the underlying graph.  We avoid
    # poking that here -- the live controller patches NodeGraphQt's
    # Port.connect_to so type errors surface naturally through the
    # exception path below, and the failure report already includes
    # the exception message naming both port types.
    return {}


def create_workflow(controller: GraphController,
                    definition: dict,
                    run: bool = False) -> dict[str, Any]:
    """Build a NEW workflow (or a fresh sub-pipeline) in one call.

    **Use this ONLY for creating fresh nodes** -- typically on an empty
    canvas, or to add an entirely new sub-pipeline alongside existing
    work.  One call instead of N x ``add_node`` + M x ``connect``;
    nodes are auto-laid-out.

    **Do NOT use this to MODIFY an existing workflow.**  For any change
    to nodes that already exist -- re-wiring, property tweaks, swapping
    types, deleting branches -- use the modify tools instead:

    - ``connect`` / ``disconnect`` -- change wires.
    - ``set_property`` -- change a node's setting.
    - ``replace_node`` -- swap one node's type, preserving compatible wires.
    - ``delete_node`` / ``add_node`` -- surgical insertion or removal.

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

    **Partial-success semantics for connections.**  Node creation is
    atomic (rolls back if it fails -- almost never happens since
    type_ids are validated).  But each connection is attempted
    independently: a failure on connection #5 does NOT delete nodes or
    drop connections #1-4.  Failed connections are reported with the
    actual port names available on each end, so the next call can fix
    the wire without needing ``describe_node`` first.  Port-name
    matching is case-insensitive.

    With ``run=True`` AND all connections succeeding, every just-
    created node is evaluated in topological order (upstream before
    downstream) -- equivalent to clicking "Run" on the canvas.  If any
    connection failed, ``run`` is skipped (the graph is incomplete)
    and a ``run_skipped`` note is included instead.

    Returns::

        {
          "created_ids":        {alias: real_id, ...},
          "connections_made":   [{src, src_port, dst, dst_port,
                                  fuzzy_matched: bool}, ...],
          "connections_failed": [
            {
              "attempted": {src, src_port, dst, dst_port},
              "reason":    "<human-readable error>",
              "available_src_ports": [...],
              "available_dst_ports": [...]
            },
            ...
          ],
          "run_results"?:  {alias: {...}},   # if run=True and clean
          "run_skipped"?:  "<reason>",       # if run=True but failures
        }

    Pre-existing graph state is never touched.
    """
    nodes, connections = _validate(controller, definition)

    # Phase 1: create nodes (atomic -- rollback if any fails).
    created: dict[str, str] = {}
    try:
        for n in nodes:
            real_id = controller.add_node(
                n['type'], properties=n.get('properties'))
            created[n['id']] = real_id
    except Exception:
        for real_id in created.values():
            try:
                controller.delete_node(real_id)
            except Exception:
                pass
        raise

    # Phase 2: attempt each connection independently.
    connections_made: list[dict[str, Any]] = []
    connections_failed: list[dict[str, Any]] = []

    for c in connections:
        src_alias = c['src']
        dst_alias = c['dst']
        src_id = created[src_alias]
        dst_id = created[dst_alias]
        src_port_req = c['src_port']
        dst_port_req = c['dst_port']

        src_ins, src_outs = _node_ports(controller, src_id)
        dst_ins, dst_outs = _node_ports(controller, dst_id)

        resolved_src = _resolve_port(src_port_req, src_outs)
        resolved_dst = _resolve_port(dst_port_req, dst_ins)

        attempted = {
            'src': src_alias, 'src_port': src_port_req,
            'dst': dst_alias, 'dst_port': dst_port_req,
        }

        if resolved_src is None or resolved_dst is None:
            reasons = []
            if resolved_src is None:
                reasons.append(
                    f"src node {src_alias!r} (type "
                    f"{controller.get_node(src_id).type_id!r}) "
                    f"has no output port {src_port_req!r}")
            if resolved_dst is None:
                reasons.append(
                    f"dst node {dst_alias!r} (type "
                    f"{controller.get_node(dst_id).type_id!r}) "
                    f"has no input port {dst_port_req!r}")
            connections_failed.append({
                'attempted': attempted,
                'reason': '; '.join(reasons),
                'available_src_ports': src_outs,
                'available_dst_ports': dst_ins,
            })
            continue

        try:
            controller.connect(src_id, resolved_src, dst_id, resolved_dst)
        except Exception as exc:
            connections_failed.append({
                'attempted': attempted,
                'reason': f"{type(exc).__name__}: {exc}",
                'available_src_ports': src_outs,
                'available_dst_ports': dst_ins,
            })
            continue

        connections_made.append({
            'src': src_alias, 'src_port': resolved_src,
            'dst': dst_alias, 'dst_port': resolved_dst,
            'fuzzy_matched': (resolved_src != src_port_req
                              or resolved_dst != dst_port_req),
        })

    # Phase 3: layout (do this even when some connections failed --
    # the partial wiring is still useful information for the user).
    _layout_new_nodes(controller, nodes, connections, created)

    result: dict[str, Any] = {
        'created_ids': dict(created),
        'connections_made': connections_made,
        'connections_failed': connections_failed,
    }

    # Phase 4: optionally run.  Skip if any connection failed -- the
    # graph is incomplete and running terminals would produce
    # misleading errors.
    if run and connections_failed:
        result['run_skipped'] = (
            f"Skipped run: {len(connections_failed)} connection(s) "
            f"failed; fix them via 'connect' first, then 'run_node' "
            f"the terminals.")
    elif run:
        run_results: dict[str, dict] = {}
        terminals = [n['id'] for n in nodes
                     if n['id'] not in {c['src'] for c in connections}]
        for alias in terminals:
            run_results[alias] = controller.run_node(created[alias])
        result['run_results'] = run_results

    return result
