"""ThreadHop: marshal callables onto the Qt main thread from any thread.

The NodeGraph lives on the Qt main thread.  MCP tool calls arrive on a
background asyncio/HTTP thread.  ``ThreadHop.call(fn, *args, **kwargs)``
schedules ``fn`` on the main thread, blocks the caller, and returns the
result (or re-raises the exception).

If called from the main thread, ``fn`` runs inline — no event loop needed.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

from PySide6 import QtCore, QtWidgets


class _Hopper(QtCore.QObject):
    """Lives on the main thread; receives queued requests via signal."""
    _request = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        app = QtWidgets.QApplication.instance()
        if app is None:
            raise RuntimeError(
                'ThreadHop requires a running QApplication.')
        self.moveToThread(app.thread())
        # QueuedConnection forces the slot to run in the receiver thread.
        self._request.connect(self._run, QtCore.Qt.QueuedConnection)

    @QtCore.Slot(object)
    def _run(self, payload: dict) -> None:
        fn = payload['fn']
        args = payload['args']
        kwargs = payload['kwargs']
        done: threading.Event = payload['done']
        try:
            payload['result'] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — re-raised in caller
            payload['error'] = exc
        finally:
            done.set()


class ThreadHop:
    """Public facade for cross-thread invocation."""

    def __init__(self) -> None:
        self._hopper = _Hopper()
        self._main_thread = QtWidgets.QApplication.instance().thread()

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` on the main thread; block until done; return result."""
        if QtCore.QThread.currentThread() is self._main_thread:
            return fn(*args, **kwargs)

        done = threading.Event()
        payload: dict = {
            'fn': fn, 'args': args, 'kwargs': kwargs,
            'done': done, 'result': None, 'error': None,
        }
        self._hopper._request.emit(payload)
        done.wait()
        if payload['error'] is not None:
            raise payload['error']
        return payload['result']
