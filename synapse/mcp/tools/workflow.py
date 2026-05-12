"""Workflow-I/O tools: new_workflow, save_workflow, load_workflow."""
from __future__ import annotations

from typing import Any

from ..controller import GraphController


def new_workflow(controller: GraphController) -> dict[str, Any]:
    """Clear the current graph.

    Removes every node (and every connection) from the active workflow.
    Use this before composing a fresh pipeline if the canvas already has
    unrelated work on it.
    """
    controller.clear_graph()
    return {'cleared': True}


def save_workflow(controller: GraphController, path: str) -> dict[str, Any]:
    """Serialize the current workflow to a JSON file at ``path``.

    Parent directories are created if missing.  Overwrites any existing
    file at the destination — no prompt.  The file can later be
    restored via ``load_workflow``.
    """
    controller.save_graph(path)
    return {'saved_to': path}


def load_workflow(controller: GraphController, path: str) -> dict[str, Any]:
    """Load a workflow from a JSON file at ``path``.

    Replaces the current graph entirely (the old state is discarded).
    Raises if the file doesn't exist or contains unknown node types
    (e.g., a plugin you don't have installed).
    """
    try:
        controller.load_graph(path)
    except FileNotFoundError:
        raise ValueError(f"No such file: {path!r}")
    except KeyError as e:
        raise ValueError(
            f"{e.args[0]}. The workflow was saved on a system that has "
            f"plugins you don't. Call list_nodes() to see what's available.")
    return {'loaded_from': path}
