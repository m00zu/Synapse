"""`synapse-serve` CLI entry point."""
from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import webbrowser

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synapse-serve",
        description="Run the Synapse web UI locally.",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Interface to bind (default 127.0.0.1).")
    p.add_argument("--port", type=int, default=0,
                   help="Port to bind (default 0 = pick a free one).")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not auto-open a browser tab.")
    p.add_argument("--allow-path", type=str, default=None,
                   help="Widen server-browse scope beyond $HOME to this dir.")
    return p


def run(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    port = args.port or _pick_free_port(args.host)
    url = f"http://{args.host}:{port}"
    if not args.no_browser:
        # Poll the port until uvicorn is accepting connections, THEN open
        # the browser. A fixed Timer races with the lifespan hook (imports
        # PySide6 + builds the widget catalog, ~3 s cold) and results in a
        # "can't connect" splash on first launch.
        threading.Thread(
            target=_open_when_ready, args=(args.host, port, url), daemon=True,
        ).start()
    # Stash CLI options on the app for the lifespan hook to consume.
    if args.allow_path:
        os.environ["SYNAPSE_ALLOW_PATH"] = args.allow_path
    uvicorn.run(
        "synapse.server.app:app",
        host=args.host,
        port=port,
        log_level="info",
        reload=False,
    )


def _pick_free_port(host: str) -> int:
    """Ask the OS for a free port on *host*. Closes the socket before
    uvicorn binds — small window but acceptable for single-user local."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _open_when_ready(host: str, port: int, url: str, timeout: float = 30.0) -> None:
    """Poll TCP connect to (host, port) until it succeeds, then open the
    browser. Gives up after *timeout* seconds to avoid hanging a background
    thread on a stuck startup."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.1)


def main() -> None:
    """Entry point registered under [project.scripts] in pyproject.toml."""
    run()
