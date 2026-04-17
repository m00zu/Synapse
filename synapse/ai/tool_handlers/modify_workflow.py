"""modify_workflow tool handler — apply a batch of graph operations."""
from __future__ import annotations

from typing import Callable

# Layout constants for newly-added nodes. Matches WorkflowLoader's spacing so
# modify_workflow-grown workflows look consistent with generate_workflow ones.
X_PAD = 300
Y_PAD = 120


def _type_name(node) -> str:
    """Human-readable node class name for error messages."""
    return getattr(node, "type_name", None) or type(node).__name__


def _resolve_port(ports: dict, hint: str):
    """Fuzzy-match *hint* to a port in *ports*. Mirrors WorkflowLoader's
    resolver: exact → prefix → case-insensitive prefix. Returns port or None.
    """
    if not hint:
        return None
    if hint in ports:
        return ports[hint]
    hint_lower = hint.lower()
    for name, port in ports.items():
        if name.startswith(hint) or name.lower().startswith(hint_lower):
            return port
    return None


def _port_type(port) -> str:
    """Best-effort port type string. Real NodeGraphQt ports carry a color
    tuple; FakeNode ports have no type info. Returns 'any' when unknown."""
    color = getattr(port, "color", None)
    if color is None:
        return "any"
    try:
        from synapse.nodes.base import PORT_COLORS
        c2t = {tuple(v): k for k, v in PORT_COLORS.items()}
        return c2t.get(tuple(color), "any")
    except Exception:
        return "any"


def _auto_wire(src, dst):
    """Pick the first (src_out, dst_in) pair of matching type. Unconnected
    inputs preferred; falls back to first unconnected pair of any type."""
    for sp in src.outputs().values():
        st = _port_type(sp)
        for dp in dst.inputs().values():
            if dp.connected_ports():
                continue
            dt = _port_type(dp)
            if st == dt or st == "any" or dt == "any":
                return sp, dp
    for sp in src.outputs().values():
        for dp in dst.inputs().values():
            if not dp.connected_ports():
                return sp, dp
    return None, None


def _try_get_pos(node):
    """Best-effort position read. NodeGraphQt nodes expose ``pos()``/``set_pos()``;
    FakeNode (tests) does not. Returns (x, y) tuple or None if unsupported."""
    pos_fn = getattr(node, "pos", None)
    if not callable(pos_fn):
        return None
    try:
        p = pos_fn()
        if hasattr(p, "x") and hasattr(p, "y"):
            return (float(p.x()), float(p.y()))  # QPointF
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return (float(p[0]), float(p[1]))
    except Exception:
        return None
    return None


def _try_set_pos(node, x: float, y: float) -> bool:
    """Set node position if the node supports it. No-op on FakeNode."""
    set_pos_fn = getattr(node, "set_pos", None)
    if not callable(set_pos_fn):
        return False
    try:
        set_pos_fn(float(x), float(y))
        return True
    except Exception:
        return False


def _auto_layout_new_nodes(graph, applied: list[dict], lookup: Callable):
    """Place newly-added nodes downstream of whichever node they connect from.

    Algorithm:
      * Collect new node ids (from add_node records).
      * Build src -> [dsts] map from connect records.
      * Fixed-point pass: for each unplaced new node, look for a connect
        whose src already has a known position (existing canvas node OR
        an already-placed new node). Place the new node at
        ``(src.x + X_PAD, src.y + k*Y_PAD)`` where k is how many peers
        already occupy that x column from the same source.
      * Any leftover new nodes (no predecessor with position) get dropped
        into an origin cascade so they're at least visible.
    """
    new_ids = [r["id"] for r in applied if r.get("op") == "add_node"]
    if not new_ids:
        return
    # src -> dsts (only for connect ops in this batch).
    edges: dict[str, list[str]] = {}
    for r in applied:
        if r.get("op") == "connect":
            edges.setdefault(r["src"], []).append(r["dst"])

    placed: set[str] = set()
    # Track slot count per "column" so multiple new nodes from the same
    # source stack vertically instead of overlapping.
    slot_usage: dict[tuple[float, float], int] = {}

    def _known_pos(node_id: str):
        if node_id in new_ids and node_id not in placed:
            return None
        n = lookup(node_id)
        if n is None:
            return None
        return _try_get_pos(n)

    # Fixed-point iteration — placement can cascade (A→B→C, both new).
    progressed = True
    while progressed:
        progressed = False
        for new_id in new_ids:
            if new_id in placed:
                continue
            # Look for any connect in applied where dst is new_id and src has a pos.
            parent_pos = None
            for src, dsts in edges.items():
                if new_id not in dsts:
                    continue
                p = _known_pos(src)
                if p is not None:
                    parent_pos = p
                    break
            if parent_pos is None:
                continue
            sx, sy = parent_pos
            # Stack siblings vertically.
            slot_key = (sx + X_PAD, sy)
            slot = slot_usage.get(slot_key, 0)
            slot_usage[slot_key] = slot + 1
            target_x = sx + X_PAD
            target_y = sy + slot * Y_PAD
            node = lookup(new_id)
            if node is not None and _try_set_pos(node, target_x, target_y):
                placed.add(new_id)
                progressed = True

    # Fallback cascade for orphaned new nodes (no predecessor with known pos).
    remaining = [nid for nid in new_ids if nid not in placed]
    if remaining:
        # Origin below existing canvas bounds (best-effort).
        max_y = 0.0
        for n in graph.all_nodes():
            p = _try_get_pos(n)
            if p is not None:
                max_y = max(max_y, p[1])
        for i, nid in enumerate(remaining):
            node = lookup(nid)
            if node is None:
                continue
            _try_set_pos(node, 0.0, max_y + Y_PAD + i * Y_PAD)


def make_modify_workflow_handler(graph, node_factory: Callable[[str, str], object]):
    """Bind a handler to (graph, node_factory).

    ``node_factory(type_name: str, node_id: str)`` must create AND register
    the node in ``graph``, then return it. NodeGraphQt's ``create_node``
    does both; test factories yielding FakeNode need to call
    ``graph.add_node(n)`` themselves.
    """

    def _lookup(node_id: str):
        for n in graph.all_nodes():
            if getattr(n, "id", None) == node_id:
                return n
            if getattr(n, "_llm_id", None) == node_id:
                return n
        return None

    def _apply_one(op: dict) -> tuple[bool, str, dict | None]:
        kind = op.get("op")
        if kind == "add_node":
            if not op.get("type") or not op.get("id"):
                return False, "add_node requires 'type' and 'id'", None
            if _lookup(op["id"]) is not None:
                return False, f"node id already exists: {op['id']}", None
            try:
                node = node_factory(op["type"], op["id"])
            except Exception as e:
                return False, f"{type(e).__name__}: {e}", None
            # Factory is responsible for adding to the graph — NodeGraphQt's
            # create_node already registers the node; calling graph.add_node
            # again would double-register and corrupt internal bookkeeping
            # (seen as ``KeyError: '_TEMP_property_widget_types'`` on real
            # NodeGraphQt). Test factories that yield FakeNode must add
            # inside the factory body.
            if node is None:
                return False, "node factory returned None", None
            return True, "", {"op": "add_node", "id": op["id"]}
        if kind == "remove_node":
            nid = op.get("id")
            node = _lookup(nid or "")
            if node is None:
                return False, f"no such node: {nid}", None
            graph.remove_node(node)
            return True, "", {"op": "remove_node", "id": nid}
        if kind == "set_prop":
            node = _lookup(op.get("id") or "")
            if node is None:
                return False, f"no such node: {op.get('id')}", None
            prop = op.get("prop")
            if not prop:
                return False, "set_prop requires 'prop'", None
            try:
                node.set_property(prop, op.get("value"))
            except Exception as e:
                return False, f"{type(e).__name__}: {e}", None
            return True, "", {"op": "set_prop", "id": node.id, "prop": prop}
        if kind == "connect":
            src = _lookup(op.get("src") or ""); dst = _lookup(op.get("dst") or "")
            if not src or not dst:
                return False, "connect requires existing src and dst node ids", None
            src_hint = op.get("src_port") or ""
            dst_hint = op.get("dst_port") or ""
            sport = _resolve_port(src.outputs(), src_hint) if src_hint else None
            dport = _resolve_port(dst.inputs(), dst_hint) if dst_hint else None
            # If a hint was given and didn't match, refuse — do NOT create a
            # new port on the fly. That used to silently hallucinate ports
            # like "A" and "B" on ImageMathNode when the LLM got the name
            # wrong. Surface the valid names so the LLM can retry.
            if src_hint and sport is None:
                valid = list(src.outputs().keys())
                return False, (
                    f"src_port {src_hint!r} not on {_type_name(src)}; "
                    f"valid outputs: {valid}"
                ), None
            if dst_hint and dport is None:
                valid = list(dst.inputs().keys())
                return False, (
                    f"dst_port {dst_hint!r} not on {_type_name(dst)}; "
                    f"valid inputs: {valid}"
                ), None
            # No hint on one/both sides — fall back to type-based auto-wire.
            if sport is None or dport is None:
                auto_s, auto_d = _auto_wire(src, dst)
                sport = sport or auto_s
                dport = dport or auto_d
            if sport is None or dport is None:
                return False, (
                    f"no compatible ports between {_type_name(src)} and "
                    f"{_type_name(dst)}"
                ), None
            sport.connect_to(dport)
            return True, "", {"op": "connect", "src": src.id, "dst": dst.id}
        if kind == "disconnect":
            src = _lookup(op.get("src") or ""); dst = _lookup(op.get("dst") or "")
            if not src or not dst:
                return False, "disconnect requires existing src and dst node ids", None
            sport = src.outputs().get(op.get("src_port") or "")
            dport = dst.inputs().get(op.get("dst_port") or "")
            if sport and dport and dport in sport.connected_ports():
                sport.peers.remove(dport); dport.peers.remove(sport)
                return True, "", {"op": "disconnect", "src": src.id, "dst": dst.id}
            return False, "no such connection", None
        return False, f"unknown op kind: {kind}", None

    def _handler(tool_input: dict) -> dict:
        ops = (tool_input or {}).get("operations")
        if not isinstance(ops, list):
            return {"error": "modify_workflow requires 'operations' (array)."}
        applied: list[dict] = []
        failed: list[dict] = []
        for op in ops:
            ok, reason, record = _apply_one(op)
            if ok and record is not None:
                applied.append(record)
            else:
                failed.append({"op": op, "reason": reason})
        # Auto-layout new nodes downstream of their predecessors. No-op on
        # FakeNode (no pos/set_pos methods) so tests are unaffected.
        _auto_layout_new_nodes(graph, applied, _lookup)
        return {"applied": applied, "failed": failed}

    return _handler
