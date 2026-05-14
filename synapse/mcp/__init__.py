"""MCP (Model Context Protocol) server integration for Synapse.

Exposes the live NodeGraph to LLM chat clients (Claude Desktop, Claude Code)
over streamable HTTP transport.  Tools operate on whichever workflow is
currently open in the running Synapse instance.

Public API:
    start_server(window) -> int    # returns chosen port
    stop_server() -> None
"""
from __future__ import annotations

__all__ = ['start_server', 'stop_server']


def start_server(window):  # pragma: no cover -- wired in Task 10
    from .server import start_server as _impl
    return _impl(window)


def stop_server():  # pragma: no cover -- wired in Task 10
    from .server import stop_server as _impl
    return _impl()
