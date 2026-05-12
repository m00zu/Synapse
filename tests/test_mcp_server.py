"""Smoke test: server starts on a free port, port file is written, and
the registered tool list contains the 12 v0 tools."""
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
            'add_node', 'delete_node', 'set_property',
            'connect', 'disconnect',
            'create_workflow',
            'run_node', 'get_node_status',
        }
    finally:
        mcp_server.stop_server()
