"""stdio <-> HTTP MCP bridge.

Claude Desktop only speaks stdio to MCP servers, but Synapse's MCP
server runs inside the live PySide6 process on HTTP.  This script is
the bridge: launched by Claude Desktop as a subprocess, it runs an
stdio MCP server locally that forwards every call to Synapse over HTTP
via the official ``mcp`` SDK client.

Setup (one-time):

    Edit ~/Library/Application Support/Claude/claude_desktop_config.json::

        {
          "mcpServers": {
            "synapse": {
              "command": "python",
              "args": ["-m", "synapse.mcp.bridge_stdio"]
            }
          }
        }

Then restart Claude Desktop.  Synapse must already be running when
Claude Desktop launches the bridge; the bridge reads the live port
from ``~/.synapse/mcp-port``.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# ── locate Synapse ──────────────────────────────────────────────────────────
PORT_FILE = Path.home() / ".synapse" / "mcp-port"


def _read_port() -> int | None:
    if not PORT_FILE.is_file():
        return None
    try:
        return int(json.loads(PORT_FILE.read_text())["port"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


async def _run() -> None:
    """Open an upstream HTTP MCP session and proxy every call to/from stdio."""
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession
    from mcp.server.lowlevel import Server
    import mcp.server.stdio as stdio_mod
    import mcp.types as types

    port = _read_port()
    if port is None:
        sys.stderr.write(
            "[synapse-bridge] No port file at "
            f"{PORT_FILE}. Is Synapse running?\n"
        )
        sys.exit(1)

    upstream_url = f"http://127.0.0.1:{port}/mcp"

    # streamablehttp_client returns a 3-tuple: (read_stream, write_stream, get_session_id)
    async with streamablehttp_client(upstream_url) as (read, write, _):
        async with ClientSession(read, write) as upstream:
            await upstream.initialize()

            # Build a local stdio server that forwards every tool call.
            server: Server = Server("synapse-stdio-bridge")

            @server.list_tools()
            async def _list_tools() -> list[types.Tool]:
                resp = await upstream.list_tools()
                return list(resp.tools)

            # call_tool decorator calls func(tool_name, arguments)
            @server.call_tool()
            async def _call_tool(
                name: str,
                arguments: dict[str, Any] | None,
            ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
                resp = await upstream.call_tool(name, arguments or {})
                # Pass through whatever the upstream returned unchanged.
                return list(resp.content)

            # stdio_server() is an async context manager yielding (read_stream, write_stream)
            async with stdio_mod.stdio_server() as (rstream, wstream):
                await server.run(
                    rstream,
                    wstream,
                    server.create_initialization_options(),
                )


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
