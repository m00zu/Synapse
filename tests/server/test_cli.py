"""CLI argparse shape + boot-config plumbing tests (no uvicorn actually run)."""
import pytest
from unittest.mock import patch

from synapse.server.cli import build_parser, run


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 0            # 0 = pick a free port
    assert args.no_browser is False
    assert args.allow_path is None


def test_parser_overrides():
    parser = build_parser()
    args = parser.parse_args([
        "--host", "0.0.0.0", "--port", "8765",
        "--no-browser", "--allow-path", "/data",
    ])
    assert args.host == "0.0.0.0"
    assert args.port == 8765
    assert args.no_browser is True
    assert args.allow_path == "/data"


def test_run_calls_uvicorn_with_app_and_args():
    with patch("synapse.server.cli.uvicorn.run") as mock_run, \
         patch("synapse.server.cli.webbrowser.open") as mock_open:
        run(["--port", "9999", "--no-browser"])
    mock_run.assert_called_once()
    call = mock_run.call_args
    assert call.kwargs["host"] == "127.0.0.1"
    assert call.kwargs["port"] == 9999
    mock_open.assert_not_called()


def test_run_opens_browser_by_default():
    with patch("synapse.server.cli.uvicorn.run") as mock_run, \
         patch("synapse.server.cli.webbrowser.open") as mock_open, \
         patch("synapse.server.cli.threading.Thread") as mock_thread:
        # Make threading.Thread synchronous so the poll-and-open runs inline.
        class _ImmediateThread:
            def __init__(self, target, args=(), kwargs=None, daemon=None):
                self._fn = target
                self._args = args
                self._kwargs = kwargs or {}
            def start(self):
                self._fn(*self._args, **self._kwargs)
        mock_thread.side_effect = _ImmediateThread
        # Short-circuit the port-poll: pretend the port is instantly reachable.
        with patch("synapse.server.cli.socket.create_connection") as mock_conn:
            mock_conn.return_value.__enter__.return_value = object()
            run(["--port", "9999"])
    mock_open.assert_called_once()
    url = mock_open.call_args.args[0]
    assert url.startswith("http://127.0.0.1:9999")
