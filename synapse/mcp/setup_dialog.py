"""Qt dialog presenting MCP setup options to end users.

Opened from ``Help → AI Connection (MCP)…``.  Uses the pure-logic
helpers in ``setup_helper.py`` for everything stateful so the UI
itself is just glue.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from . import setup_helper as _helper


class MCPSetupDialog(QtWidgets.QDialog):
    """Standalone help dialog — non-modal, can stay open while user works."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('AI Connection (MCP)')
        self.setWindowFlag(QtCore.Qt.WindowType.Dialog)
        self.resize(620, 520)

        self._port = _helper.get_running_port()
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Status banner
        layout.addWidget(self._status_banner())

        # Intro text
        intro = QtWidgets.QLabel(
            "Connect Claude Code or Claude Desktop to this Synapse "
            "instance so you can build and run workflows by chatting. "
            "No API key needed — your chat subscription handles auth.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Claude Code section
        layout.addWidget(self._claude_code_section())

        # Claude Desktop section
        layout.addWidget(self._claude_desktop_section())

        # Other clients
        layout.addWidget(self._other_clients_section())

        layout.addStretch(1)

        # Bottom: Close button
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    def _status_banner(self) -> QtWidgets.QWidget:
        w = QtWidgets.QFrame()
        w.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(10, 8, 10, 8)
        dot = QtWidgets.QLabel()
        if self._port is not None:
            dot.setText('●')
            dot.setStyleSheet('color: #3ec73e; font-size: 18px;')
            msg = QtWidgets.QLabel(
                f"MCP server listening on "
                f"<b>127.0.0.1:{self._port}</b>")
        else:
            dot.setText('●')
            dot.setStyleSheet('color: #d05050; font-size: 18px;')
            msg = QtWidgets.QLabel(
                "<b>MCP server not running.</b>  "
                "Restart Synapse — the server starts automatically.")
        h.addWidget(dot)
        h.addWidget(msg, 1)
        return w

    def _claude_code_section(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox('Claude Code (CLI)')
        v = QtWidgets.QVBoxLayout(group)
        v.addWidget(QtWidgets.QLabel(
            "Run this once in a terminal where the <code>claude</code> "
            "CLI is installed:"))

        cmd_view = QtWidgets.QLineEdit(
            _helper.claude_code_command(self._port) if self._port
            else '— MCP server not running —')
        cmd_view.setReadOnly(True)
        cmd_view.setFont(QtGui.QFont('Menlo, Consolas, monospace'))
        v.addWidget(cmd_view)

        btn = QtWidgets.QPushButton('Copy Command')
        btn.clicked.connect(
            lambda: self._copy(cmd_view.text(),
                                'Command copied to clipboard.'))
        btn.setEnabled(self._port is not None)
        v.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        return group

    def _claude_desktop_section(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox('Claude Desktop (macOS / Windows app)')
        v = QtWidgets.QVBoxLayout(group)
        v.addWidget(QtWidgets.QLabel(
            "Auto-configure Claude Desktop to launch Synapse's stdio "
            "bridge.  Existing MCP servers in your config are preserved."))

        btn = QtWidgets.QPushButton('Auto-configure Claude Desktop')
        btn.clicked.connect(self._on_setup_claude_desktop)
        v.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        path_info = QtWidgets.QLabel(
            f"<i>Config file: "
            f"<code>{_helper.claude_desktop_config_path()}</code></i>")
        path_info.setStyleSheet('color: #888;')
        path_info.setWordWrap(True)
        v.addWidget(path_info)

        return group

    def _other_clients_section(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox('Other clients (Cursor, ChatGPT desktop, etc.)')
        v = QtWidgets.QVBoxLayout(group)
        v.addWidget(QtWidgets.QLabel(
            "Add an MCP server with this URL in your client's settings:"))

        url_view = QtWidgets.QLineEdit(
            _helper.mcp_url(self._port) if self._port
            else '— MCP server not running —')
        url_view.setReadOnly(True)
        url_view.setFont(QtGui.QFont('Menlo, Consolas, monospace'))
        v.addWidget(url_view)

        btn = QtWidgets.QPushButton('Copy URL')
        btn.clicked.connect(
            lambda: self._copy(url_view.text(),
                                'URL copied to clipboard.'))
        btn.setEnabled(self._port is not None)
        v.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        return group

    # ── Actions ──────────────────────────────────────────────────────
    def _copy(self, text: str, toast: str) -> None:
        QtWidgets.QApplication.clipboard().setText(text)
        QtWidgets.QMessageBox.information(
            self, 'Copied', toast)

    def _on_setup_claude_desktop(self) -> None:
        if self._port is None:
            QtWidgets.QMessageBox.warning(
                self, 'MCP not running',
                'The MCP server is not running.  Restart Synapse and '
                'try again.')
            return
        path = _helper.claude_desktop_config_path()
        try:
            result = _helper.write_claude_desktop_config(path)
        except ValueError as e:
            QtWidgets.QMessageBox.critical(
                self, 'Could not update Claude Desktop config', str(e))
            return

        extras = ''
        if result['other_servers']:
            extras = (
                f"\n\nPreserved other MCP servers: "
                f"{', '.join(result['other_servers'])}.")
        msg = (
            f"{'Updated' if result['replaced'] else 'Added'} "
            f"Synapse in:\n{result['config_path']}"
            f"{extras}\n\n"
            "Restart Claude Desktop to pick up the new config.  "
            "Synapse must be running for the bridge to connect.")
        QtWidgets.QMessageBox.information(
            self, 'Claude Desktop configured', msg)


def open_setup_dialog(parent=None) -> None:
    """Open (or re-focus) the MCP setup dialog."""
    dlg = MCPSetupDialog(parent)
    dlg.exec()
