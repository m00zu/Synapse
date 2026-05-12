"""Thread-safe MCP call log + Qt signal for live observation.

``MCPLogger`` is a singleton living on the Qt main thread.  The
``_wrap`` function in ``server.py`` calls ``MCPLogger.instance().log(...)``
from the asyncio bridge thread; the QObject signal ``new_entry`` is
auto-queued via Qt's connection routing so dialog subscribers on the
main thread receive updates safely.

Bounded ring buffer (1000 entries) prevents unbounded memory growth.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from PySide6 import QtCore, QtWidgets


@dataclass
class MCPLogEntry:
    """One row of the MCP call log."""
    timestamp: float                       # time.time() at call start
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    duration_ms: float = 0.0
    result_summary: str = ''                # short repr of return value
    error: str | None = None                # exception text if it raised


class MCPLogger(QtCore.QObject):
    """Singleton call-log accumulator with a Qt signal for live updates."""

    new_entry = QtCore.Signal(object)       # carries MCPLogEntry

    _instance: Optional['MCPLogger'] = None
    _instance_lock = threading.Lock()

    MAX_ENTRIES = 1000

    def __init__(self) -> None:
        super().__init__()
        # Move ourselves onto the main thread so any emit is routed
        # via QueuedConnection to receivers (the dialog) regardless of
        # which thread called log().
        app = QtWidgets.QApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        self._entries: deque[MCPLogEntry] = deque(maxlen=self.MAX_ENTRIES)
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> 'MCPLogger':
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def log(self, entry: MCPLogEntry) -> None:
        """Append an entry; safe to call from any thread."""
        with self._lock:
            self._entries.append(entry)
        # Cross-thread emit is fine: AutoConnection picks QueuedConnection
        # when the receiver's thread != the emitting thread.
        self.new_entry.emit(entry)

    def snapshot(self) -> list[MCPLogEntry]:
        """Atomic copy of current entries (newest last)."""
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
