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


def antigravity_config_path() -> Path:
    """Return the location of Google Antigravity's MCP config file.

    Antigravity follows the Gemini-CLI dotfile convention
    (``~/.gemini/...``) on every platform — same path on macOS,
    Windows, and Linux.
    """
    return Path.home() / '.gemini' / 'antigravity' / 'mcp_config.json'


def antigravity_entry(port: int, server_name: str = 'synapse') -> dict:
    """Build the JSON snippet Antigravity expects.

    Antigravity uses an HTTP transport with a ``serverUrl`` field
    rather than the stdio ``{command, args}`` shape Claude Desktop uses
    — no bridge subprocess needed, the client talks directly to
    Synapse's MCP HTTP server.
    """
    return {server_name: {'serverUrl': mcp_url(int(port))}}


def write_antigravity_config(
    config_path: Path,
    port: int,
    server_name: str = 'synapse',
) -> dict[str, Any]:
    """Merge Synapse into Antigravity's MCP config (HTTP shape).

    Same shape and merge semantics as ``write_claude_desktop_config``
    but writes the ``{serverUrl: URL}`` entry instead of the
    ``{command, args, cwd}`` stdio entry.  Preserves other servers
    and unrelated keys.
    """
    return _write_http_config(
        config_path, antigravity_entry(port, server_name), server_name)


# ── Gemini CLI ──────────────────────────────────────────────────────────────

def gemini_cli_config_path() -> Path:
    """Return the location of Gemini CLI's settings file.

    Same path on macOS, Windows, and Linux — Gemini CLI uses a single
    top-level settings file under ``~/.gemini/``.
    """
    return Path.home() / '.gemini' / 'settings.json'


def gemini_cli_entry(port: int, server_name: str = 'synapse') -> dict:
    """Build the JSON snippet Gemini CLI expects.

    Gemini CLI uses ``httpUrl`` for HTTP MCP transport (different
    field name from Antigravity's ``serverUrl`` or Claude Code's
    URL-as-positional-arg).
    """
    return {server_name: {'httpUrl': mcp_url(int(port))}}


def write_gemini_cli_config(
    config_path: Path,
    port: int,
    server_name: str = 'synapse',
) -> dict[str, Any]:
    """Merge Synapse into Gemini CLI's settings file.

    Preserves all other keys and other MCP servers in the file.
    Refuses to clobber malformed JSON.
    """
    return _write_http_config(
        config_path, gemini_cli_entry(port, server_name), server_name)


# ── Shared merge logic for any HTTP-style MCP config ────────────────────────

def _write_http_config(
    config_path: Path,
    entry: dict,
    server_name: str,
) -> dict[str, Any]:
    """Merge a single ``mcpServers`` entry into a JSON settings file.

    ``entry`` must have shape ``{server_name: {<client-specific keys>}}``.
    Used by both Antigravity (``serverUrl``) and Gemini CLI (``httpUrl``).
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text())
            if not isinstance(data, dict):
                data = {}
        except json.JSONDecodeError:
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
    servers.update(entry)

    config_path.write_text(json.dumps(data, indent=2))
    return {
        'config_path': str(config_path),
        'replaced': replaced,
        'other_servers': other_servers,
    }


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
        # Nuitka's ``--onefile`` mode may make ``sys.executable`` point
        # at a temp-extracted bootstrap that gets cleaned up.  Nuitka
        # exposes the durable path to the *original* .exe / .app via
        # the ``NUITKA_ONEFILE_BINARY`` env var — prefer that.
        import os as _os
        exe = _os.environ.get('NUITKA_ONEFILE_BINARY') or sys.executable
        return {
            server_name: {
                'command': exe,
                'args': ['--mcp-bridge'],
            },
        }

    # Source case: launch the bridge script by absolute path.  The
    # script itself fixes up sys.path to find the ``synapse`` package,
    # so this works regardless of cwd, PYTHONPATH, or whether site-
    # packages happens to have an unrelated ``synapse`` package on it.
    try:
        import synapse as _synapse_pkg
        bridge_path = (Path(_synapse_pkg.__file__).parent
                       / 'mcp' / 'bridge_stdio.py')
    except Exception:
        # Fallback shouldn't happen — synapse is importable since
        # we're running inside it — but be defensive.
        bridge_path = Path('synapse/mcp/bridge_stdio.py')

    return {
        server_name: {
            'command': python_path or sys.executable,
            'args': [str(bridge_path)],
        },
    }


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


# ── Preferred-port preference (persists user's choice across launches) ──────

_PREF_FILE = Path.home() / '.synapse' / 'mcp-port-preference'


def get_preferred_port() -> int | None:
    """Return the user's saved preferred port, or None if never set.

    The MCP server reads this at startup and tries it before falling
    back to the built-in default (51780) and then to a random port.
    """
    if not _PREF_FILE.is_file():
        return None
    try:
        data = json.loads(_PREF_FILE.read_text())
        port = int(data.get('port'))
        if 1 <= port <= 65535:
            return port
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def set_preferred_port(port: int) -> Path:
    """Persist the user's preferred port for next-launch use.

    Validates the port is in the usable range (1024–65535 — privileged
    ports below 1024 require root on most systems).  Returns the
    written path so callers can show it in a confirmation message.
    """
    port = int(port)
    if not (1024 <= port <= 65535):
        raise ValueError(
            f"Port {port} out of range — pick a value between 1024 and 65535.")
    _PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PREF_FILE.write_text(json.dumps({'port': port}))
    return _PREF_FILE


def clear_preferred_port() -> bool:
    """Forget the saved preference, falling back to the default port.

    Returns True if a preference was removed, False if there was none.
    """
    if _PREF_FILE.is_file():
        _PREF_FILE.unlink()
        return True
    return False
