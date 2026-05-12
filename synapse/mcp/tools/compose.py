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


def create_workflow(controller: GraphController,
                    definition: dict,
                    run: bool = False) -> dict[str, Any]:
    """Build a whole workflow in one shot, atomically.

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

    result: dict[str, Any] = {'created_ids': dict(created)}

    if run:
        run_results: dict[str, dict] = {}
        for alias in _terminal_aliases(nodes, connections):
            run_results[alias] = controller.run_node(created[alias])
        result['run_results'] = run_results

    return result
