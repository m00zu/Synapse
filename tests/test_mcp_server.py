"""Smoke test: server starts on a free port, port file is written, and
the registered tool list contains the 18 v0 tools."""
import json
import time
import socket
from pathlib import Path

import pytest
from PySide6 import QtWidgets


@pytest.fixture(scope='module')
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def test_server_starts_and_writes_port_file(qapp, tmp_path,
                                              monkeypatch):
    from synapse.mcp.controller import FakeGraphController, NodeInfo
    from synapse.mcp import server as mcp_server

    monkeypatch.setattr(mcp_server, '_PORT_FILE',
                        tmp_path / 'mcp-port')

    ctl = FakeGraphController(registered=[
        NodeInfo('cat', 'X', 'cat.X', [], [], [], 'docs'),
    ])
    port = _free_port()
    handle = mcp_server.start_server_with_controller(ctl, port=port)
    try:
        # Wait for port file to appear
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if (tmp_path / 'mcp-port').exists():
                break
            qapp.processEvents()
            time.sleep(0.05)
        assert (tmp_path / 'mcp-port').exists(), \
            'port discovery file not written'
        data = json.loads((tmp_path / 'mcp-port').read_text())
        assert data['port'] == port
    finally:
        mcp_server.stop_server()


def test_server_registers_v0_tools(qapp, tmp_path, monkeypatch):
    from synapse.mcp.controller import FakeGraphController
    from synapse.mcp import server as mcp_server

    monkeypatch.setattr(mcp_server, '_PORT_FILE',
                        tmp_path / 'mcp-port')

    ctl = FakeGraphController(registered=[])
    port = _free_port()
    handle = mcp_server.start_server_with_controller(ctl, port=port)
    try:
        tools = mcp_server.list_registered_tool_names()
        assert set(tools) == {
            'list_nodes', 'describe_node', 'search_nodes',
            'describe_graph',
            'add_node', 'delete_node', 'replace_node', 'set_property',
            'connect', 'disconnect',
            'create_workflow',
            'run_node', 'get_node_status', 'get_node_output',
            'get_node_image',
            'new_workflow', 'save_workflow', 'load_workflow',
        }
    finally:
        mcp_server.stop_server()


def test_tool_schemas_omit_controller_arg(qapp, tmp_path, monkeypatch):
    """Regression: _wrap must preserve typed signatures so FastMCP exposes
    proper input schemas to the LLM (not the useless *args/**kwargs that
    a naive wrapper produces).
    """
    import asyncio
    from synapse.mcp.controller import FakeGraphController
    from synapse.mcp import server as mcp_server

    monkeypatch.setattr(mcp_server, '_PORT_FILE', tmp_path / 'mcp-port')

    ctl = FakeGraphController(registered=[])
    port = _free_port()
    mcp_server.start_server_with_controller(ctl, port=port)
    try:
        tools = asyncio.new_event_loop().run_until_complete(
            mcp_server._fastmcp.list_tools())
        # Every tool's input schema must (a) NOT contain 'controller' and
        # (b) NOT contain a bare 'args' or 'kwargs' (which would be the
        # broken naive *args/**kwargs schema).
        for t in tools:
            props = set((t.inputSchema or {}).get('properties', {}).keys())
            assert 'controller' not in props, \
                f"{t.name}: schema leaks controller arg"
            # A naive *args/**kwargs wrapper produces exactly {'args', 'kwargs'}.
            # A properly-wrapped tool either has real named params or is empty
            # (for no-user-arg tools like list_nodes).
            assert props != {'args', 'kwargs'}, \
                f"{t.name}: schema is broken naive *args/**kwargs " \
                f"(saw only {props})"
    finally:
        mcp_server.stop_server()


def test_stop_server_releases_port_within_timeout(qapp, tmp_path, monkeypatch):
    """stop_server should release the bound port quickly enough that
    a subsequent start_server can re-bind the same port."""
    import time
    from synapse.mcp.controller import FakeGraphController
    from synapse.mcp import server as mcp_server

    monkeypatch.setattr(mcp_server, '_PORT_FILE',
                        tmp_path / 'mcp-port')

    ctl = FakeGraphController(registered=[])
    port = _free_port()

    # Start, stop, then re-bind the same port to confirm it was released.
    mcp_server.start_server_with_controller(ctl, port=port)
    # Wait briefly for uvicorn to actually listen.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            with socket.socket() as s:
                s.connect(('127.0.0.1', port))
                break
        except OSError:
            qapp.processEvents()
            time.sleep(0.05)

    mcp_server.stop_server(timeout=3.0)

    # After stop_server returns, the port should be re-bindable
    # within a brief settling window.  Try for up to 2s.
    rebound = False
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            with socket.socket() as s:
                s.bind(('127.0.0.1', port))
                rebound = True
                break
        except OSError:
            qapp.processEvents()
            time.sleep(0.05)
    assert rebound, f"port {port} still bound 2s after stop_server()"
