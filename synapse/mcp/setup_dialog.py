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
        self.resize(640, 460)

        # Active port for everything the user copies / writes here.
        # Defaults to the live running port, falling back to the saved
        # preference, falling back to the built-in 51780.
        running = _helper.get_running_port()
        preferred = _helper.get_preferred_port()
        self._running_port = running                       # may be None
        self._port = running or preferred or 51780
        # Widgets that need to refresh when the port spinbox changes:
        self._port_spin: QtWidgets.QSpinBox | None = None
        self._cmd_view: QtWidgets.QLineEdit | None = None
        self._url_view: QtWidgets.QLineEdit | None = None
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Everything except the Close button lives in a scroll area so
        # the dialog stays compact even with all the auto-configure
        # sections.  Setting widgetResizable so the inner content
        # expands to the scroll viewport's width.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
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

        # Port selector — affects every command/URL shown below.
        layout.addWidget(self._port_section())

        # Claude Code section
        layout.addWidget(self._claude_code_section())

        # Claude Desktop section
        layout.addWidget(self._claude_desktop_section())

        # Antigravity section
        layout.addWidget(self._antigravity_section())

        # Gemini CLI section
        layout.addWidget(self._gemini_cli_section())

        # Other clients
        layout.addWidget(self._other_clients_section())

        layout.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        # Bottom: Close button (stays pinned outside the scroll area).
        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(16, 8, 16, 12)
        button_row.addStretch(1)
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        button_row.addWidget(close_btn)
        outer.addLayout(button_row)

    def _status_banner(self) -> QtWidgets.QWidget:
        w = QtWidgets.QFrame()
        w.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(10, 8, 10, 8)
        dot = QtWidgets.QLabel()
        if self._running_port is not None:
            dot.setText('●')
            dot.setStyleSheet('color: #3ec73e; font-size: 18px;')
            msg = QtWidgets.QLabel(
                f"MCP server listening on "
                f"<b>127.0.0.1:{self._running_port}</b>")
        else:
            dot.setText('●')
            dot.setStyleSheet('color: #d05050; font-size: 18px;')
            msg = QtWidgets.QLabel(
                "<b>MCP server not running.</b>  "
                "Restart Synapse — the server starts automatically.")
        h.addWidget(dot)
        h.addWidget(msg, 1)
        return w

    def _port_section(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox('Port')
        v = QtWidgets.QVBoxLayout(group)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Use port:'))
        spin = QtWidgets.QSpinBox()
        spin.setRange(1024, 65535)
        spin.setValue(int(self._port))
        spin.valueChanged.connect(self._on_port_changed)
        self._port_spin = spin
        row.addWidget(spin)

        apply_btn = QtWidgets.QPushButton('Save for next launch')
        apply_btn.clicked.connect(self._on_save_preferred_port)
        row.addWidget(apply_btn)

        reset_btn = QtWidgets.QPushButton('Reset to default')
        reset_btn.clicked.connect(self._on_reset_preferred_port)
        row.addWidget(reset_btn)
        row.addStretch(1)
        v.addLayout(row)

        note = QtWidgets.QLabel(
            "<i>Defaults to the currently-running port.  Changing it "
            "here updates the commands below immediately.  "
            "<b>'Save for next launch'</b> persists the choice so Synapse "
            "binds the same port on every future start — restart "
            "Synapse after saving for it to take effect.</i>")
        note.setWordWrap(True)
        note.setStyleSheet('color: #888;')
        v.addWidget(note)
        return group

    def _claude_code_section(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox('Claude Code (CLI)')
        v = QtWidgets.QVBoxLayout(group)
        v.addWidget(QtWidgets.QLabel(
            "Run this once in a terminal where the <code>claude</code> "
            "CLI is installed:"))

        cmd_view = QtWidgets.QLineEdit(_helper.claude_code_command(self._port))
        cmd_view.setReadOnly(True)
        cmd_view.setFont(QtGui.QFont('Menlo, Consolas, monospace'))
        self._cmd_view = cmd_view
        v.addWidget(cmd_view)

        btn = QtWidgets.QPushButton('Copy Command')
        btn.clicked.connect(
            lambda: self._copy(cmd_view.text(),
                                'Command copied to clipboard.'))
        v.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        return group

    def _claude_desktop_section(self) -> QtWidgets.QWidget:
        return self._stdio_client_section(
            title='Claude Desktop (macOS / Windows app)',
            blurb=("Auto-configure Claude Desktop to launch Synapse's "
                   "stdio bridge.  Existing MCP servers in your config "
                   "are preserved."),
            button_label='Auto-configure Claude Desktop',
            config_path=_helper.claude_desktop_config_path(),
            display_name='Claude Desktop',
        )

    def _antigravity_section(self) -> QtWidgets.QWidget:
        """Antigravity uses an HTTP transport (``serverUrl`` field) —
        no stdio bridge needed.  Writes directly to its config file.
        """
        return self._http_client_section(
            title='Google Antigravity',
            blurb=("Auto-configure Antigravity to talk to Synapse over "
                   "HTTP (no stdio bridge).  Existing MCP servers in "
                   "the file are preserved."),
            button_label='Auto-configure Antigravity',
            config_path=_helper.antigravity_config_path(),
            display_name='Antigravity',
            writer=_helper.write_antigravity_config,
        )

    def _gemini_cli_section(self) -> QtWidgets.QWidget:
        """Gemini CLI uses an HTTP transport (``httpUrl`` field) — same
        pattern as Antigravity but a different config file + key name.
        """
        return self._http_client_section(
            title='Gemini CLI',
            blurb=("Auto-configure Gemini CLI to talk to Synapse over "
                   "HTTP.  Merges into ``~/.gemini/settings.json``; "
                   "other settings + MCP servers are preserved."),
            button_label='Auto-configure Gemini CLI',
            config_path=_helper.gemini_cli_config_path(),
            display_name='Gemini CLI',
            writer=_helper.write_gemini_cli_config,
        )

    def _http_client_section(
        self, *,
        title: str,
        blurb: str,
        button_label: str,
        config_path,
        display_name: str,
        writer,
    ) -> QtWidgets.QWidget:
        """Shared builder for any HTTP-MCP client (no bridge).  ``writer``
        is the function from ``setup_helper`` that takes
        ``(config_path, port)`` and merges the URL entry into the file.
        """
        group = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(group)
        v.addWidget(QtWidgets.QLabel(blurb))

        btn = QtWidgets.QPushButton(button_label)
        btn.clicked.connect(
            lambda: self._write_http_config(
                config_path, display_name, writer))
        v.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        path_info = QtWidgets.QLabel(
            f"<i>Config file: <code>{config_path}</code></i>")
        path_info.setStyleSheet('color: #888;')
        path_info.setWordWrap(True)
        v.addWidget(path_info)

        return group

    def _stdio_client_section(
        self, *,
        title: str,
        blurb: str,
        button_label: str,
        config_path,
        display_name: str,
    ) -> QtWidgets.QWidget:
        """Shared builder for any stdio-MCP client that uses the
        ``{command, args, cwd?}`` config shape (Claude Desktop, Antigravity,
        Cursor, Cline, etc.).
        """
        group = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(group)
        v.addWidget(QtWidgets.QLabel(blurb))

        btn = QtWidgets.QPushButton(button_label)
        btn.clicked.connect(
            lambda: self._write_stdio_config(config_path, display_name))
        v.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        path_info = QtWidgets.QLabel(
            f"<i>Config file: <code>{config_path}</code></i>")
        path_info.setStyleSheet('color: #888;')
        path_info.setWordWrap(True)
        v.addWidget(path_info)

        return group

    def _other_clients_section(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox('Other clients (Cursor, ChatGPT desktop, etc.)')
        v = QtWidgets.QVBoxLayout(group)
        v.addWidget(QtWidgets.QLabel(
            "Add an MCP server with this URL in your client's settings:"))

        url_view = QtWidgets.QLineEdit(_helper.mcp_url(self._port))
        url_view.setReadOnly(True)
        url_view.setFont(QtGui.QFont('Menlo, Consolas, monospace'))
        self._url_view = url_view
        v.addWidget(url_view)

        btn = QtWidgets.QPushButton('Copy URL')
        btn.clicked.connect(
            lambda: self._copy(url_view.text(),
                                'URL copied to clipboard.'))
        v.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        return group

    # ── Actions ──────────────────────────────────────────────────────
    def _copy(self, text: str, toast: str) -> None:
        QtWidgets.QApplication.clipboard().setText(text)
        QtWidgets.QMessageBox.information(
            self, 'Copied', toast)

    def _write_http_config(self, config_path, display_name: str,
                            writer) -> None:
        """Shared HTTP-MCP-client writer.  ``writer(config_path, port)``
        is the helper that produces the right entry shape for this
        client (``write_antigravity_config``, ``write_gemini_cli_config``,
        etc.).
        """
        try:
            result = writer(config_path, self._port)
        except ValueError as e:
            QtWidgets.QMessageBox.critical(
                self, f'Could not update {display_name} config', str(e))
            return

        extras = ''
        if result['other_servers']:
            extras = (
                f"\n\nPreserved other MCP servers: "
                f"{', '.join(result['other_servers'])}.")
        port_warn = ''
        if (self._running_port is not None
                and self._running_port != self._port):
            port_warn = (
                f"\n\n⚠ Wrote URL using port {self._port}, but Synapse "
                f"is currently listening on {self._running_port}.  "
                f"Update the port spinbox above or restart Synapse so "
                f"the live port matches before connecting from "
                f"{display_name}.")
        msg = (
            f"{'Updated' if result['replaced'] else 'Added'} "
            f"Synapse in:\n{result['config_path']}"
            f"{extras}"
            f"{port_warn}\n\n"
            f"Restart {display_name} to pick up the new config.")
        QtWidgets.QMessageBox.information(
            self, f'{display_name} configured', msg)

    def _write_stdio_config(self, config_path, display_name: str) -> None:
        """Shared handler — write the synapse stdio entry into any
        client's MCP config file in the standard ``{command, args, cwd?}``
        shape.  Used by the Claude Desktop and Antigravity buttons.
        """
        try:
            result = _helper.write_claude_desktop_config(config_path)
        except ValueError as e:
            QtWidgets.QMessageBox.critical(
                self, f'Could not update {display_name} config', str(e))
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
            f"Restart {display_name} to pick up the new config.  "
            f"Synapse must be running for the bridge to connect.  "
            f"The bridge auto-discovers the live port from "
            f"~/.synapse/mcp-port — port changes here don't affect "
            f"{display_name}.")
        QtWidgets.QMessageBox.information(
            self, f'{display_name} configured', msg)

    def _on_port_changed(self, value: int) -> None:
        """Refresh the displayed command + URL when the port spinbox changes."""
        self._port = int(value)
        if self._cmd_view is not None:
            self._cmd_view.setText(_helper.claude_code_command(self._port))
        if self._url_view is not None:
            self._url_view.setText(_helper.mcp_url(self._port))

    def _on_save_preferred_port(self) -> None:
        """Persist the spinbox value as the next-launch port."""
        try:
            path = _helper.set_preferred_port(self._port)
        except ValueError as e:
            QtWidgets.QMessageBox.critical(
                self, 'Invalid port', str(e))
            return
        running = self._running_port
        running_msg = ''
        if running is not None and running != self._port:
            running_msg = (
                f"\n\nSynapse is still listening on {running}.  "
                f"Restart Synapse to switch to port {self._port}.")
        QtWidgets.QMessageBox.information(
            self, 'Preferred port saved',
            f"Saved port {self._port} as the preference at:\n{path}"
            f"{running_msg}")

    def _on_reset_preferred_port(self) -> None:
        """Forget any saved preference and fall back to the built-in default."""
        had = _helper.clear_preferred_port()
        if had:
            QtWidgets.QMessageBox.information(
                self, 'Preference cleared',
                "Saved port preference removed.  Synapse will use the "
                "built-in default (51780) on next launch.")
        else:
            QtWidgets.QMessageBox.information(
                self, 'No preference set',
                "There was no saved preference — already using the "
                "built-in default (51780).")


def open_setup_dialog(parent=None) -> None:
    """Open (or re-focus) the MCP setup dialog."""
    dlg = MCPSetupDialog(parent)
    dlg.exec()
