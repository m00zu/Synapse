"""Graph executor — topological run, cancel flag, event emission.

Reuses desktop BatchGraphWorker's node-order logic (synapse/app.py) so the
execution semantics match what the user sees in the desktop app.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator


class Executor:
    def __init__(self, session) -> None:
        self._s = session
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    async def run(self) -> AsyncIterator[dict]:
        graph = self._s.graph
        sorted_nodes = _topo_sort(graph.all_nodes())
        for node in sorted_nodes:
            if self._stop:
                break
            yield {"kind": "node_started", "node_id": node.id}
            success, err = await asyncio.to_thread(_evaluate_one, node)
            yield {"kind": "node_finished", "node_id": node.id,
                   "success": bool(success),
                   "error": None if success else (err or "evaluate failed")}
            if success:
                try:
                    from synapse.server.previews import write_previews
                    output_values = getattr(node, "output_values", {}) or {}
                    written = await asyncio.to_thread(
                        write_previews, node.id, output_values, self._s.preview_dir,
                    )
                except Exception as exc:  # defensive — preview failures must not kill the run
                    written = []
                    import logging; logging.getLogger(__name__).warning(
                        "preview write failed for %s: %s", node.id, exc,
                    )
                for w in written:
                    yield {"kind": "preview_available",
                           "node_id": node.id, "port": w["port"], "preview_kind": w["kind"]}


async def run_graph(session) -> AsyncIterator[dict]:
    """Convenience wrapper used by tests and routes."""
    exe = Executor(session)
    async for ev in exe.run():
        yield ev


def _topo_sort(nodes: list) -> list:
    """Return nodes in dependency order (leaves last).
    Mirrors synapse/app.py::BatchGraphWorker topo sort."""
    visited: set = set()
    order: list = []

    def _walk(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for in_p in node.inputs().values():
            for cp in in_p.connected_ports():
                _walk(cp.node())
        order.append(node)

    for n in nodes:
        _walk(n)
    return order


def _evaluate_one(node) -> tuple[bool, str | None]:
    """Invoke node.evaluate(); translate any exception into (False, reason)."""
    if not hasattr(node, "evaluate"):
        return True, None
    try:
        success, err = node.evaluate()
        return bool(success), err
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
