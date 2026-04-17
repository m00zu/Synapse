"""inspect_canvas tool handler — read-only canvas dump with a token cap."""
from __future__ import annotations

import json
from typing import Callable

from synapse.ai.context import estimate_tokens


DEFAULT_TOKEN_CAP = 2000


def make_inspect_canvas_handler(graph, token_cap: int = DEFAULT_TOKEN_CAP) -> Callable[[dict], dict]:
    """Bind a handler to this graph + cap so it can be registered with ToolDispatcher."""

    def _handler(tool_input: dict) -> dict:
        node_id_filter: set[str] | None = None
        if tool_input.get("node_ids"):
            node_id_filter = set(tool_input["node_ids"])
        include_props = tool_input.get("include_props", True)

        nodes_out: list[dict] = []
        for node in graph.all_nodes():
            if node_id_filter is not None and node.id not in node_id_filter:
                continue
            entry: dict = {"id": node.id, "type": getattr(node, "type_name", type(node).__name__)}
            if include_props:
                try:
                    props = {
                        k: v for k, v in node.model.custom_properties.items()
                        if not k.startswith("_")
                    }
                except Exception:
                    props = {}
                if props:
                    entry["props"] = props
            nodes_out.append(entry)

        edges_out: list[list[str]] = []
        keep = {n["id"] for n in nodes_out}
        for node in graph.all_nodes():
            src_id = node.id
            for _, port in node.outputs().items():
                for connected in port.connected_ports():
                    dst_id = connected.node().id
                    if src_id in keep or dst_id in keep:
                        if (src_id in keep and dst_id in keep) or node_id_filter is None:
                            edges_out.append([src_id, dst_id])

        # Token cap — serialise, trim nodes until under budget.
        result = {"nodes": nodes_out, "edges": edges_out, "truncated": False}
        approx = estimate_tokens(json.dumps(result))
        while approx > token_cap and nodes_out:
            nodes_out.pop()
            edges_out[:] = [
                e for e in edges_out
                if e[0] in {n["id"] for n in nodes_out} and e[1] in {n["id"] for n in nodes_out}
            ]
            result = {"nodes": nodes_out, "edges": edges_out, "truncated": True}
            approx = estimate_tokens(json.dumps(result))
        return result

    return _handler
