"""GraphController: abstract facade over NodeGraph + fake for tests.

Real impl lives in the same file (NodeGraphController) and is wired by
``server.py`` once Synapse has constructed its NodeGraph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class NodeInfo:
    """Description of a node *type* (template, not an instance)."""
    category: str
    name: str            # NODE_NAME (human label, may contain spaces)
    type_id: str         # NodeGraphQt's type_ identifier
    properties: list[str]
    input_ports: list[str]
    output_ports: list[str]
    summary: str         # one-line description for catalog


@dataclass
class NodeRecord:
    """State of an active node *instance* in the current graph."""
    id: str
    type_id: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    status: str = 'pending'   # 'pending' | 'running' | 'clean' | 'error'
    last_message: str | None = None


class GraphController(Protocol):
    """Operations the MCP tools call.  All methods run on the Qt main thread."""

    def list_registered(self) -> list[NodeInfo]: ...
    def describe_registered(self, type_id: str) -> NodeInfo: ...
    def list_active(self) -> list[NodeRecord]: ...
    def list_connections(self) -> list[tuple[str, str, str, str]]: ...
    def get_node(self, node_id: str) -> NodeRecord: ...
    def add_node(self, type_id: str, properties: dict | None = None,
                 position: tuple[float, float] | None = None) -> str: ...
    def delete_node(self, node_id: str) -> None: ...
    def set_property(self, node_id: str, prop: str, value: Any) -> None: ...
    def connect(self, src_id: str, src_port: str,
                dst_id: str, dst_port: str) -> None: ...
    def disconnect(self, src_id: str, src_port: str,
                   dst_id: str, dst_port: str) -> None: ...
    def run_node(self, node_id: str) -> dict: ...


# ── Fake implementation for unit tests ──────────────────────────────────────

class FakeGraphController:
    """In-memory GraphController for unit-testing tools without Qt."""

    def __init__(self, registered: list[NodeInfo] | None = None) -> None:
        self._registered = {n.type_id: n for n in (registered or [])}
        self._active: dict[str, NodeRecord] = {}
        self._connections: list[tuple[str, str, str, str]] = []
        self._counter = 0
        self._run_results: dict[str, dict] = {}

    # ── inspection / query ──────────────────────────────────────────────
    def list_registered(self) -> list[NodeInfo]:
        return list(self._registered.values())

    def describe_registered(self, type_id: str) -> NodeInfo:
        if type_id not in self._registered:
            raise KeyError(f"unknown node type: {type_id}")
        return self._registered[type_id]

    def list_active(self) -> list[NodeRecord]:
        return list(self._active.values())

    def list_connections(self) -> list[tuple[str, str, str, str]]:
        return list(self._connections)

    def get_node(self, node_id: str) -> NodeRecord:
        if node_id not in self._active:
            raise KeyError(f"unknown node id: {node_id}")
        return self._active[node_id]

    # ── mutation ────────────────────────────────────────────────────────
    def add_node(self, type_id: str, properties: dict | None = None,
                 position: tuple[float, float] | None = None) -> str:
        if type_id not in self._registered:
            raise KeyError(f"unknown node type: {type_id}")
        self._counter += 1
        nid = f'n{self._counter}'
        info = self._registered[type_id]
        self._active[nid] = NodeRecord(
            id=nid, type_id=type_id, name=info.name,
            properties=dict(properties or {}),
        )
        return nid

    def set_property(self, node_id: str, prop: str, value: Any) -> None:
        rec = self.get_node(node_id)
        rec.properties[prop] = value

    def connect(self, src_id: str, src_port: str,
                dst_id: str, dst_port: str) -> None:
        self.get_node(src_id)  # raises if missing
        self.get_node(dst_id)
        self._connections.append((src_id, src_port, dst_id, dst_port))

    def disconnect(self, src_id: str, src_port: str,
                   dst_id: str, dst_port: str) -> None:
        edge = (src_id, src_port, dst_id, dst_port)
        if edge not in self._connections:
            raise KeyError(f"no such connection: {edge}")
        self._connections.remove(edge)

    def delete_node(self, node_id: str) -> None:
        if node_id not in self._active:
            raise KeyError(f"unknown node id: {node_id}")
        del self._active[node_id]
        # Drop any edges touching this node.
        self._connections = [
            (s, sp, d, dp) for (s, sp, d, dp) in self._connections
            if s != node_id and d != node_id
        ]

    # ── execution ───────────────────────────────────────────────────────
    def run_node(self, node_id: str) -> dict:
        self.get_node(node_id)
        return self._run_results.get(node_id,
            {'success': True, 'message': None, 'duration_ms': 0.0})

    def set_run_result(self, node_id: str, *, success: bool,
                       message: str | None, duration_ms: float) -> None:
        """Test helper: pre-program what run_node returns for a node."""
        self._run_results[node_id] = {
            'success': success, 'message': message,
            'duration_ms': duration_ms,
        }
