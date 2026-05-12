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


def _is_frozen() -> bool:
    """Detect whether we're running inside a Nuitka or PyInstaller bundle."""
    # PyInstaller sets sys.frozen=True.  Nuitka sets __compiled__ on the
    # __main__ module (when built with --standalone or onefile).
    if getattr(sys, 'frozen', False):
        return True
    main_mod = sys.modules.get('__main__')
    if main_mod is not None and hasattr(main_mod, '__compiled__'):
        return True
    return False


def claude_desktop_entry(python_path: str | None = None,
                          server_name: str = 'synapse') -> dict:
    """Build the JSON snippet Claude Desktop expects.

    Two flavours, picked automatically based on whether Synapse is
    running as a frozen binary or from source:

    - **Frozen** (Nuitka / PyInstaller bundle): re-launch ``sys.executable``
      with the ``--mcp-bridge`` flag.  ``main.py`` short-circuits to the
      stdio proxy before any Qt imports happen, so the bundle binary
      acts as both the GUI and the bridge — no extra files to ship.

    - **Source** (running ``python main.py`` during dev): invoke
      ``python -m synapse.mcp.bridge_stdio`` with ``cwd`` pinned to the
      directory containing the ``synapse`` package, so the package
      resolves regardless of where Claude Desktop launches the
      subprocess from.

    The ``python_path`` override always wins for the dev case (lets the
    user point at a different env if they want).
    """
    if _is_frozen():
        # Bundle case: re-launch the Synapse binary itself.
        return {
            server_name: {
                'command': sys.executable,
                'args': ['--mcp-bridge'],
            },
        }

    # Source case: invoke via python -m.
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
