"""Pure-logic helpers for wiring AI chat clients to Synapse's MCP server.

UI-free so it's trivially unit-testable; the dialog in
``setup_dialog.py`` glues these to Qt buttons and clipboard.
"""
from __future__ import annotations

import json
import platform
import shlex
import sys
from pathlib import Path
from typing import Any


def claude_desktop_config_path() -> Path:
    """Return the OS-specific location of Claude Desktop's config file."""
    system = platform.system()
    if system == 'Darwin':
        return (Path.home() / 'Library' / 'Application Support'
                / 'Claude' / 'claude_desktop_config.json')
    if system == 'Windows':
        # %APPDATA% is the canonical location.
        import os
        appdata = os.environ.get('APPDATA')
        if appdata:
            return Path(appdata) / 'Claude' / 'claude_desktop_config.json'
        return Path.home() / 'AppData' / 'Roaming' / 'Claude' / 'claude_desktop_config.json'
    # Linux / other — best effort.
    return Path.home() / '.config' / 'Claude' / 'claude_desktop_config.json'


def claude_code_command(port: int,
                        server_name: str = 'synapse') -> str:
    """Return the one-line ``claude mcp add`` command for Claude Code.

    Includes proper shell-quoting of the URL so it's copy-paste-safe.
    """
    url = f"http://127.0.0.1:{port}/mcp"
    return (f"claude mcp add {shlex.quote(server_name)} "
            f"--transport http {shlex.quote(url)}")


def mcp_url(port: int) -> str:
    """The bare MCP HTTP URL for paste-it-anywhere clients."""
    return f"http://127.0.0.1:{port}/mcp"


def claude_desktop_entry(python_path: str | None = None,
                          server_name: str = 'synapse') -> dict:
    """Build the JSON snippet Claude Desktop expects.

    Uses ``sys.executable`` by default — the exact Python running
    Synapse — so the stdio bridge subprocess inherits the right env
    (where the ``synapse`` package is importable).

    Also pins ``cwd`` to the directory containing the ``synapse``
    package.  Required when running from source so
    ``python -m synapse.mcp.bridge_stdio`` resolves; harmless when
    Synapse is installed into site-packages.
    """
    try:
        import synapse as _synapse_pkg
        synapse_parent: str | None = str(
            Path(_synapse_pkg.__file__).parent.parent)
    except Exception:
        synapse_parent = None

    entry: dict = {
        'command': python_path or sys.executable,
        'args': ['-m', 'synapse.mcp.bridge_stdio'],
    }
    if synapse_parent:
        entry['cwd'] = synapse_parent
    return {server_name: entry}


def write_claude_desktop_config(
    config_path: Path,
    python_path: str | None = None,
    server_name: str = 'synapse',
) -> dict[str, Any]:
    """Merge Synapse into Claude Desktop's config (creating the file if missing).

    - Preserves every other ``mcpServers`` entry the user has set up.
    - Pretty-prints with 2-space indent so the file stays human-editable.
    - Returns ``{config_path, replaced: bool, other_servers: list[str]}``
      so the caller can show an accurate confirmation message.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text())
            if not isinstance(data, dict):
                data = {}
        except json.JSONDecodeError:
            # Malformed file — refuse to clobber.  Caller surfaces this.
            raise ValueError(
                f"Existing config at {config_path} isn't valid JSON. "
                "Fix or move it aside first.")
    else:
        data = {}

    servers = data.setdefault('mcpServers', {})
    if not isinstance(servers, dict):
        raise ValueError(
            f"'mcpServers' in {config_path} isn't an object — "
            "the file shape is unexpected.")

    replaced = server_name in servers
    other_servers = sorted(s for s in servers if s != server_name)
    servers.update(claude_desktop_entry(python_path, server_name))

    config_path.write_text(json.dumps(data, indent=2))
    return {
        'config_path': str(config_path),
        'replaced': replaced,
        'other_servers': other_servers,
    }


def get_running_port() -> int | None:
    """Read the live MCP server port from the discovery file, or None."""
    port_file = Path.home() / '.synapse' / 'mcp-port'
    if not port_file.is_file():
        return None
    try:
        data = json.loads(port_file.read_text())
        return int(data.get('port'))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
