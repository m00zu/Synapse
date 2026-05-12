"""FastMCP server bootstrap for Synapse.

Runs an MCP streamable-HTTP server on 127.0.0.1:<port> in a background
thread.  Tools dispatch through a ``ThreadHop`` so each call runs on
the Qt main thread (where NodeGraph lives).  Port is written to
``~/.synapse/mcp-port`` for clients to discover.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .bridge import ThreadHop
from .controller import GraphController, NodeGraphController
from .tools.discovery import list_nodes, describe_node, search_nodes
from .tools.graph import (
    describe_graph, add_node, delete_node, set_property, connect, disconnect,
)
from .tools.compose import create_workflow
from .tools.execution import run_node, get_node_status

_PORT_FILE = Path.home() / '.synapse' / 'mcp-port'

_server_thread: Optional[threading.Thread] = None
_fastmcp: Optional[FastMCP] = None
_tool_names: list[str] = []


def _wrap(hop: ThreadHop, controller: GraphController, fn):
    """Wrap a tool fn so it dispatches to the Qt main thread via ThreadHop."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return hop.call(fn, controller, *args, **kwargs)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _register_tools(mcp: FastMCP, hop: ThreadHop,
                    controller: GraphController) -> list[str]:
    pairs = [
        ('list_nodes', list_nodes),
        ('describe_node', describe_node),
        ('search_nodes', search_nodes),
        ('describe_graph', describe_graph),
        ('add_node', add_node),
        ('delete_node', delete_node),
        ('set_property', set_property),
        ('connect', connect),
        ('disconnect', disconnect),
        ('create_workflow', create_workflow),
        ('run_node', run_node),
        ('get_node_status', get_node_status),
    ]
    names = []
    for name, fn in pairs:
        wrapped = _wrap(hop, controller, fn)
        mcp.tool(name=name)(wrapped)
        names.append(name)
    return names


def list_registered_tool_names() -> list[str]:
    """For tests/diagnostics: returns the names of currently-registered tools."""
    return list(_tool_names)


def start_server_with_controller(controller: GraphController,
                                  port: int = 0) -> dict:
    """Start the MCP server bound to a provided controller (for tests).

    Returns ``{port}``.  Production code should call ``start_server(window)``
    which constructs a NodeGraphController automatically.
    """
    global _server_thread, _fastmcp, _tool_names
    if _server_thread is not None:
        raise RuntimeError('MCP server already running.')

    hop = ThreadHop()
    mcp = FastMCP('synapse')
    _tool_names = _register_tools(mcp, hop, controller)
    _fastmcp = mcp

    # Pick an actual port if 0 was passed.
    import socket
    if port == 0:
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]

    # Write port discovery file before starting the thread so callers can
    # read the port immediately after this function returns.
    _PORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PORT_FILE.write_text(json.dumps({'port': port}))

    def _serve():
        # FastMCP exposes run() for stdio; for HTTP we use the streamable HTTP
        # transport via uvicorn against the underlying ASGI app.
        import uvicorn
        config = uvicorn.Config(
            mcp.streamable_http_app(),
            host='127.0.0.1', port=port,
            log_level='warning',
        )
        server = uvicorn.Server(config)
        server.run()

    _server_thread = threading.Thread(target=_serve, daemon=True,
                                       name='synapse-mcp')
    _server_thread.start()
    return {'port': port}


def start_server(window) -> int:
    """Start the MCP server against the running Synapse window."""
    controller = NodeGraphController(window.graph)
    return start_server_with_controller(controller)['port']


def stop_server() -> None:
    """Best-effort stop.  v0 relies on daemon thread + process exit."""
    global _server_thread, _fastmcp, _tool_names
    # uvicorn doesn't expose a clean shutdown without keeping a handle;
    # daemon thread dies with the process.  v1 will use a proper handle.
    _server_thread = None
    _fastmcp = None
    _tool_names = []
    try:
        _PORT_FILE.unlink()
    except FileNotFoundError:
        pass
