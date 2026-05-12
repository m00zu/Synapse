"""Execution tools: run_node, get_node_status."""
from __future__ import annotations

from typing import Any

from ..controller import GraphController


def run_node(controller: GraphController, node_id: str) -> dict[str, Any]:
    """Evaluate a node, re-evaluating any dirty upstream first.

    Mirrors what the user gets when they click "Run" on the node in the
    Synapse UI.  Returns ``{success, message, duration_ms}``.  A node
    that succeeded but produced a status message (warnings, counts, etc.)
    still has ``success=True``; ``message`` carries the text.
    """
    try:
        controller.get_node(node_id)
    except KeyError:
        raise ValueError(
            f"Unknown node id: {node_id!r}. "
            f"Call describe_graph() to see current node ids.")
    return controller.run_node(node_id)


def get_node_status(controller: GraphController,
                    node_id: str) -> dict[str, Any]:
    """Return the last known status of a node without re-running it.

    Status values: ``'pending'`` (never evaluated since last dirty),
    ``'running'``, ``'clean'`` (last evaluate succeeded), ``'error'``
    (last evaluate raised or returned False).
    """
    try:
        rec = controller.get_node(node_id)
    except KeyError:
        raise ValueError(
            f"Unknown node id: {node_id!r}. "
            f"Call describe_graph() to see current node ids.")
    return {'node_id': rec.id, 'status': rec.status,
            'last_message': rec.last_message}
