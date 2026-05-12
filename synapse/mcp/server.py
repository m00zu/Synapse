"""FastMCP server bootstrap for Synapse.

Runs an MCP streamable-HTTP server on 127.0.0.1:<port> in a background
thread.  Tools dispatch through a ``ThreadHop`` so each call runs on
the Qt main thread (where NodeGraph lives).  Port is written to
``~/.synapse/mcp-port`` for clients to discover.
"""
from __future__ import annotations

import functools
import inspect
import json
import threading
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .bridge import ThreadHop
from .controller import GraphController, NodeGraphController
from .tools.discovery import list_nodes, describe_node, search_nodes
from .tools.graph import (
    describe_graph, add_node, delete_node, set_property,
    connect, disconnect, replace_node,
)
from .tools.compose import create_workflow
from .tools.execution import run_node, get_node_status, get_node_output
from .tools.workflow import new_workflow, save_workflow, load_workflow

_PORT_FILE = Path.home() / '.synapse' / 'mcp-port'

# Stable port that the MCP server tries to bind first, so users only need
# to run ``claude mcp add`` once per machine.  Picked from the ephemeral
# range (49152–65535) at a memorable position; falls back to a random
# port if this one is busy.
_DEFAULT_PORT = 51780

_server_thread: Optional[threading.Thread] = None
_fastmcp: Optional[FastMCP] = None
_tool_names: list[str] = []
_uvicorn_server: Optional[Any] = None


def _wrap(hop: ThreadHop, controller: GraphController, fn):
    """Wrap a tool fn so it dispatches to the Qt main thread via ThreadHop.

    Preserves the wrapped function's typed signature (minus the first
    ``controller`` arg, which is closed over) so FastMCP can build a
    proper JSON input schema for the LLM.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return hop.call(fn, controller, *args, **kwargs)
    # Drop the leading 'controller' parameter from the LLM-visible signature.
    orig_sig = inspect.signature(fn)
    params = list(orig_sig.parameters.values())
    wrapper.__signature__ = orig_sig.replace(parameters=params[1:])
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
        ('replace_node', replace_node),
        ('set_property', set_property),
        ('connect', connect),
        ('disconnect', disconnect),
        ('create_workflow', create_workflow),
        ('run_node', run_node),
        ('get_node_status', get_node_status),
        ('get_node_output', get_node_output),
        ('new_workflow', new_workflow),
        ('save_workflow', save_workflow),
        ('load_workflow', load_workflow),
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

    # Pick an actual port if 0 was passed.  Try the stable default first
    # (so users only have to run `claude mcp add` once); fall back to a
    # random port if the default is already taken (e.g. a second Synapse
    # instance is running).
    import socket
    if port == 0:
        try:
            with socket.socket() as s:
                s.bind(('127.0.0.1', _DEFAULT_PORT))
            port = _DEFAULT_PORT
        except OSError:
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0))
                port = s.getsockname()[1]
            print(f"[mcp] default port {_DEFAULT_PORT} busy; "
                  f"using random {port}. Re-run `claude mcp add` "
                  f"to pick up the new URL.")

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
        global _uvicorn_server
        _uvicorn_server = uvicorn.Server(config)
        _uvicorn_server.run()

    _server_thread = threading.Thread(target=_serve, daemon=True,
                                       name='synapse-mcp')
    _server_thread.start()
    return {'port': port}


def start_server(window) -> int:
    """Start the MCP server against the running Synapse window."""
    controller = NodeGraphController(window.graph)
    return start_server_with_controller(controller)['port']


def stop_server(timeout: float = 3.0) -> None:
    """Signal uvicorn shutdown and wait for the serving thread to exit.

    Best-effort: if uvicorn doesn't honour ``should_exit`` within
    ``timeout`` seconds, we move on and the daemon thread dies at
    process exit instead.
    """
    global _server_thread, _fastmcp, _tool_names, _uvicorn_server

    thread = _server_thread
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True

    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
        # If still alive after the join window, the thread keeps running
        # in the background but we surrender ownership — process exit
        # will reap it (daemon=True).

    _server_thread = None
    _fastmcp = None
    _uvicorn_server = None
    _tool_names = []
    try:
        _PORT_FILE.unlink()
    except FileNotFoundError:
        pass
