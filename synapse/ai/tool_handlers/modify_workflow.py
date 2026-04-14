"""modify_workflow tool handler — apply a batch of graph operations."""
from __future__ import annotations

from typing import Callable


def make_modify_workflow_handler(graph, node_factory: Callable[[str, str], object]):
    """Bind a handler to (graph, node_factory).

    ``node_factory(type_name: str, node_id: str)`` creates a new node of the
    requested type and returns it. Phase 2b will supply a factory backed by
    NodeGraphQt's ``create_node`` + the class registry; Phase 2a tests supply
    a trivial callable that yields FakeNode instances.
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
            node = node_factory(op["type"], op["id"])
            graph.add_node(node)
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
            sport = src.outputs().get(op.get("src_port") or "")
            dport = dst.inputs().get(op.get("dst_port") or "")
            if not sport:
                sport = src.add_output(op.get("src_port") or "out_1")
            if not dport:
                dport = dst.add_input(op.get("dst_port") or "in_1")
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
        return {"applied": applied, "failed": failed}

    return _handler
