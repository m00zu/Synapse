"""Live MCP call log dialog.

Non-modal QDialog that shows every MCP tool call (timestamp, tool,
duration, success/error, args, result preview).  Subscribes to
``MCPLogger.new_entry`` for live updates without blocking the main
thread.  Includes Clear / Copy / Pause buttons.
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .logger import MCPLogger, MCPLogEntry


_COLUMNS = ['Time', 'Tool', 'ms', 'Status', 'Args', 'Result / Error']


class MCPLogDialog(QtWidgets.QDialog):
    """Non-modal viewer for the MCP call log."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('MCP Call Log')
        self.setWindowFlag(QtCore.Qt.WindowType.Dialog)
        self.setModal(False)
        self.resize(820, 480)

        self._logger = MCPLogger.instance()
        self._paused = False

        self._build_ui()
        self._populate_initial()
        # AutoConnection — emitter is on main thread (logger moves itself
        # there), so this is a direct connection.  Calls into log() from
        # the asyncio thread queue automatically through the singleton's
        # owning thread.
        self._logger.new_entry.connect(
            self._on_new_entry, QtCore.Qt.ConnectionType.AutoConnection)

    # ── UI ──────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Toolbar row
        toolbar = QtWidgets.QHBoxLayout()
        self._pause_btn = QtWidgets.QPushButton('Pause')
        self._pause_btn.setCheckable(True)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        toolbar.addWidget(self._pause_btn)

        clear_btn = QtWidgets.QPushButton('Clear')
        clear_btn.clicked.connect(self._on_clear)
        toolbar.addWidget(clear_btn)

        copy_btn = QtWidgets.QPushButton('Copy All')
        copy_btn.clicked.connect(self._on_copy_all)
        toolbar.addWidget(copy_btn)

        toolbar.addStretch(1)
        self._count_label = QtWidgets.QLabel('0 entries')
        self._count_label.setStyleSheet('color: #888;')
        toolbar.addWidget(self._count_label)
        layout.addLayout(toolbar)

        # Table
        self._table = QtWidgets.QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            5, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, 1)

    # ── data plumbing ──────────────────────────────────────────────
    def _populate_initial(self) -> None:
        for entry in self._logger.snapshot():
            self._append_row(entry)
        self._update_count()

    def _on_new_entry(self, entry: MCPLogEntry) -> None:
        if self._paused:
            return
        self._append_row(entry)
        self._update_count()
        # Scroll to bottom so newest is visible.
        self._table.scrollToBottom()

    def _append_row(self, entry: MCPLogEntry) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        ts = _dt.datetime.fromtimestamp(entry.timestamp).strftime(
            '%H:%M:%S.%f')[:-3]
        items = [
            ts,
            entry.tool,
            f'{entry.duration_ms:.1f}',
            'ok' if entry.success else 'ERROR',
            self._compact_args(entry.args),
            entry.error if entry.error else entry.result_summary,
        ]
        for col, text in enumerate(items):
            item = QtWidgets.QTableWidgetItem(text)
            # Red the row on error.
            if not entry.success:
                item.setForeground(QtGui.QColor('#d05050'))
            # Monospace for tool / args / result.
            if col in (1, 4, 5):
                item.setFont(QtGui.QFont('Menlo, Consolas, monospace'))
            # Tooltip — full text for long values.
            item.setToolTip(text)
            self._table.setItem(row, col, item)

    def _compact_args(self, args: dict) -> str:
        if not args:
            return ''
        try:
            s = json.dumps(args, default=str, ensure_ascii=False)
        except Exception:
            s = repr(args)
        return s if len(s) <= 150 else s[:147] + '…'

    def _update_count(self) -> None:
        n = self._table.rowCount()
        self._count_label.setText(f"{n} entr{'y' if n == 1 else 'ies'}")

    # ── slot handlers ──────────────────────────────────────────────
    def _on_pause_toggled(self, on: bool) -> None:
        self._paused = on
        self._pause_btn.setText('Resume' if on else 'Pause')

    def _on_clear(self) -> None:
        self._logger.clear()
        self._table.setRowCount(0)
        self._update_count()

    def _on_copy_all(self) -> None:
        # Plain-text TSV of all rows so the user can paste into a spreadsheet
        # or shell.
        lines = ['\t'.join(_COLUMNS)]
        for row in range(self._table.rowCount()):
            cells = []
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                cells.append(item.text() if item else '')
            lines.append('\t'.join(cells))
        QtWidgets.QApplication.clipboard().setText('\n'.join(lines))


def open_log_dialog(parent=None) -> None:
    """Open (or focus) the MCP call log dialog."""
    dlg = MCPLogDialog(parent)
    dlg.show()
    # Keep a reference on the parent so GC doesn't kill it.
    if parent is not None:
        if not hasattr(parent, '_mcp_log_dialog'):
            parent._mcp_log_dialog = None
        # Close any previous instance before replacing.
        prev = getattr(parent, '_mcp_log_dialog', None)
        if prev is not None and prev is not dlg:
            try:
                prev.close()
            except Exception:
                pass
        parent._mcp_log_dialog = dlg
