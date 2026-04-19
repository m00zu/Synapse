"""Server-side session state.

Wraps NodeGraphQt's NodeGraph so the server controls the graph through a
narrow API (add/remove/set_prop/connect/disconnect/export/import_).
One SessionState per `synapse serve` process in Phase 1b (single-user).
The graph is headless: no view window, no show() calls, but each
BaseExecutionNode still builds its Qt widget tree during __init__ because
the spec builder side-effect depends on it.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Optional

from NodeGraphQt import NodeGraph

from synapse.server.event_bus import EventBus


class NodeGraphHeadless:
    """Thin façade over NodeGraphQt.NodeGraph exposing only server operations.

    The underlying ``NodeGraph`` instance is reachable via ``.node_graph``
    for callers that need the full NodeGraphQt surface — specifically the
    chat tool-dispatcher, whose handlers (``generate_workflow``,
    ``modify_workflow``) call ``create_node`` / ``registered_nodes`` /
    ``remove_node(node_obj)`` directly.
    """

    def __init__(self) -> None:
        self._g = NodeGraph()
        # Register every known node class so add_node() can find them.
        from synapse.widgets.catalog import (
            _install_legacy_shims, _import_all_plugins, _iter_subclasses,
        )
        _install_legacy_shims()
        _import_all_plugins()
        from synapse.nodes.base import BaseExecutionNode
        self._types: dict[str, type] = {}
        for cls in _iter_subclasses(BaseExecutionNode):
            self._types[cls.__name__] = cls
            try:
                self._g.register_node(cls)
            except Exception:
                pass  # some classes may fail to register; server still boots

    @property
    def node_graph(self):
        """The underlying NodeGraphQt NodeGraph. Exposed for callers that
        need the full API — the chat tool-dispatcher is the main client."""
        return self._g

    # ---- CRUD ----
    def add_node(self, type_name: str, x: float = 0, y: float = 0) -> str:
        if type_name not in self._types:
            raise ValueError(f"unknown node type: {type_name}")
        cls = self._types[type_name]
        # NodeGraphQt registers nodes as __identifier__ + "." + __name__ (class name),
        # NOT __identifier__ + "." + NODE_NAME (display name).
        node = self._g.create_node(
            f"{cls.__identifier__}.{cls.__name__}",
            pos=[x, y],
            push_undo=False,
        )
        return node.id

    def remove_node(self, node_id: str) -> None:
        node = self._g.get_node_by_id(node_id)
        if node is None:
            raise KeyError(node_id)
        self._g.remove_node(node, push_undo=False)

    def all_nodes(self) -> list:
        return list(self._g.all_nodes())

    def get_node(self, node_id: str):
        node = self._g.get_node_by_id(node_id)
        if node is None:
            raise KeyError(node_id)
        return node

    def set_prop(self, node_id: str, prop: str, value) -> None:
        self.get_node(node_id).set_property(prop, value, push_undo=False)

    def set_pos(self, node_id: str, x: float, y: float) -> None:
        """Update a node's canvas position."""
        self.get_node(node_id).set_pos(float(x), float(y))

    def connect(self, src_id: str, dst_id: str,
                src_port: Optional[str] = None, dst_port: Optional[str] = None) -> None:
        src = self.get_node(src_id)
        dst = self.get_node(dst_id)
        if src_port is None:
            src_port = next(iter(src.outputs().keys()))
        if dst_port is None:
            dst_port = next(iter(dst.inputs().keys()))
        src.outputs()[src_port].connect_to(dst.inputs()[dst_port])

    def disconnect(self, src_id: str, dst_id: str,
                   src_port: str, dst_port: str) -> None:
        src = self.get_node(src_id)
        dst = self.get_node(dst_id)
        src.outputs()[src_port].disconnect_from(dst.inputs()[dst_port])

    def is_connected(self, src_id: str, dst_id: str) -> bool:
        src = self.get_node(src_id)
        dst = self.get_node(dst_id)
        for op in src.outputs().values():
            for peer in op.connected_ports():
                if peer.node() is dst:
                    return True
        return False

    def export(self) -> dict:
        """Return workflow JSON compatible with the desktop's save format."""
        return self._g.serialize_session()

    def import_(self, workflow: dict) -> None:
        self._g.deserialize_session(workflow)

    def clear(self) -> None:
        """Remove every node from the graph. Used by test fixtures to reset
        state between tests without tearing down the whole NodeGraph (which
        would re-walk every registered node class and Qt-instantiate them)."""
        try:
            self._g.clear_session()
        except Exception:
            pass


class SessionState:
    def __init__(self, allow_path: Optional[str] = None) -> None:
        self.graph = NodeGraphHeadless()
        self.allow_path = allow_path
        self.preview_dir = Path(tempfile.mkdtemp(prefix="synapse-serve-"))
        self.lock = asyncio.Lock()
        self.bus = EventBus()
        self.executor = None
        self._closed = False
        # Chat slot — lazily attached by routes_chat.start_turn. History persists
        # for the session's lifetime (not across server restarts — Phase 2+).
        self.chat_session = None  # type: Optional[Any]
        self.chat_history: list[dict] = []

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Clear the NodeGraph BEFORE tempdir removal. Otherwise each session
        # leaks its NodeGraph + all Qt widgets to interpreter shutdown, where
        # nondeterministic GC ordering crashes Qt with "Fatal Python error:
        # Aborted" on some platforms (observed on macOS + PySide6 6.10).
        try:
            self.graph.clear()
        except Exception:
            pass
        import shutil
        shutil.rmtree(self.preview_dir, ignore_errors=True)
