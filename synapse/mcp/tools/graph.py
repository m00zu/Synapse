"""Graph snapshot + manipulation tools."""
from __future__ import annotations

from typing import Any

from ..controller import GraphController


def describe_graph(controller: GraphController) -> dict[str, Any]:
    """Snapshot of the current workflow: every node + every connection.

    Returns ``{nodes: [...], connections: [...]}``.  Each node has
    ``{id, type, name, properties, status}``.  Each connection has
    ``{src_node_id, src_port, dst_node_id, dst_port}``.  Use this any
    time you've lost track of the graph state.
    """
    nodes = [
        {'id': n.id, 'type': n.type_id, 'name': n.name,
         'properties': dict(n.properties),
         'status': n.status, 'last_message': n.last_message}
        for n in controller.list_active()
    ]
    conns = [
        {'src_node_id': s, 'src_port': sp,
         'dst_node_id': d, 'dst_port': dp}
        for (s, sp, d, dp) in controller.list_connections()
    ]
    return {'nodes': nodes, 'connections': conns}


def add_node(controller: GraphController, node_type: str,
             properties: dict | None = None,
             position: tuple[float, float] | None = None
             ) -> dict[str, Any]:
    """Create a single new node of ``node_type`` in the current graph.

    **For building a workflow from scratch with 2+ nodes, prefer
    ``create_workflow`` instead** — it batches add+connect into one atomic
    call and auto-positions nodes so they don't overlap.

    Use ``add_node`` (this tool) only for surgical single-node insertions
    into an existing graph.

    Returns ``{node_id, inputs, outputs}``.  Pre-set properties may be
    provided as ``{prop_name: value}``.  Use ``describe_node`` first to
    learn the available port + property names.
    """
    try:
        nid = controller.add_node(node_type, properties=properties,
                                   position=position)
    except KeyError:
        raise ValueError(
            f"Unknown node type: {node_type!r}. "
            f"Call list_nodes() to see all registered types.")
    info = controller.describe_registered(node_type)
    return {'node_id': nid,
            'inputs': list(info.input_ports),
            'outputs': list(info.output_ports)}


def set_property(controller: GraphController, node_id: str,
                 prop: str, value: Any) -> dict[str, Any]:
    """Set a property on an existing node.

    Returns ``{node_id, prop, value}`` confirming the write.  Property
    names come from ``describe_node`` (or ``describe_graph`` for an
    instance's current values).
    """
    try:
        controller.set_property(node_id, prop, value)
    except KeyError as e:
        raise ValueError(
            f"{e.args[0]}. Call describe_graph() to see current node ids.")
    return {'node_id': node_id, 'prop': prop, 'value': value}


def connect(controller: GraphController,
            src_node_id: str, src_port: str,
            dst_node_id: str, dst_port: str) -> dict[str, Any]:
    """Wire ``src_node_id.src_port`` -> ``dst_node_id.dst_port``.

    Use ``describe_node`` to discover port names per type, and
    ``describe_graph`` to see current node ids.
    """
    try:
        controller.connect(src_node_id, src_port, dst_node_id, dst_port)
    except KeyError as e:
        raise ValueError(
            f"{e.args[0]}. Call describe_graph() to see current node ids.")
    return {'src_node_id': src_node_id, 'src_port': src_port,
            'dst_node_id': dst_node_id, 'dst_port': dst_port}


def disconnect(controller: GraphController,
               src_node_id: str, src_port: str,
               dst_node_id: str, dst_port: str) -> dict[str, Any]:
    """Remove a wire previously created with ``connect``.

    Errors if the edge does not exist; use ``describe_graph()`` to
    enumerate current connections.
    """
    try:
        controller.disconnect(src_node_id, src_port, dst_node_id, dst_port)
    except KeyError as e:
        raise ValueError(
            f"{e.args[0]}. Call describe_graph() to see current connections.")
    return {'src_node_id': src_node_id, 'src_port': src_port,
            'dst_node_id': dst_node_id, 'dst_port': dst_port}


def delete_node(controller: GraphController,
                node_id: str) -> dict[str, Any]:
    """Remove a node and any edges that touch it.

    Returns ``{deleted: node_id}``.  Use ``describe_graph()`` first if
    the LLM is uncertain about node ids.
    """
    try:
        controller.delete_node(node_id)
    except KeyError as e:
        raise ValueError(
            f"{e.args[0]}. Call describe_graph() to see current node ids.")
    return {'deleted': node_id}
