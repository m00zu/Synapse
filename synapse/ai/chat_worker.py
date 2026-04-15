"""Qt QThread wrapper for the ChatOrchestrator. Emits signals for each
orchestrator event so the chat panel can update the UI on the main thread.

Tool dispatches that mutate the NodeGraphQt canvas (modify_workflow,
write_python_script, generate_workflow's Apply path) must run on the Qt
main thread — creating NodeGraphQt widgets from a worker thread crashes
on macOS with ``NSWindow should only be instantiated on the main thread!``.

To handle that safely we wrap the provided ``ToolDispatcher`` in a
``_MainThreadDispatchProxy`` that stays on the main thread and uses a
BlockingQueuedConnection signal to marshal each dispatch call over.
"""
from __future__ import annotations

from PySide6 import QtCore

from synapse.ai.orchestrator import ChatOrchestrator


class _MainThreadDispatchProxy(QtCore.QObject):
    """Thin proxy around a ToolDispatcher that runs each ``dispatch`` call
    on the main Qt thread. Mimics ToolDispatcher's public interface so the
    orchestrator is none the wiser."""

    # Internal signal for blocking-queued cross-thread dispatch.
    # Payload: (name: str, tool_input: dict, result_box: dict)
    _dispatch_request = QtCore.Signal(str, object, object)

    def __init__(self, real_dispatcher):
        # parent=None — this proxy lives on whatever thread constructs it
        # (ChatStreamWorker is constructed on the main thread BEFORE the
        # worker is moved to its QThread, so this proxy stays on main).
        super().__init__()
        self._real = real_dispatcher
        self._dispatch_request.connect(
            self._on_dispatch_request,
            QtCore.Qt.BlockingQueuedConnection,
        )

    @QtCore.Slot(str, object, object)
    def _on_dispatch_request(self, name: str, tool_input: dict, result_box: dict) -> None:
        try:
            result_box["result"] = self._real.dispatch(name, tool_input)
        except Exception as e:
            result_box["result"] = {"error": f"{type(e).__name__}: {e}"}

    def dispatch(self, name: str, tool_input: dict):
        """Dispatch through the main thread. If the caller already IS on
        the main thread (tests, direct calls), bypass the signal entirely."""
        if QtCore.QThread.currentThread() is self.thread():
            return self._real.dispatch(name, tool_input)
        box: dict = {}
        self._dispatch_request.emit(name, tool_input, box)
        return box.get("result", {"error": "dispatch returned no result"})

    # Pass-through for anything else the orchestrator might need on a
    # dispatcher (currently just ``registered_names``).
    def __getattr__(self, item):
        return getattr(self._real, item)


class ChatStreamWorker(QtCore.QObject):
    # Streaming text chunks (prose turn).
    token_received = QtCore.Signal(str)
    # Tool lifecycle.
    tool_call_started = QtCore.Signal(str, dict)      # name, input
    tool_call_finished = QtCore.Signal(str, dict)     # name, result
    cap_exceeded = QtCore.Signal(str)                 # tool name at which cap hit
    # Workflow preview — emitted after generate_workflow tool_call_finished.
    workflow_preview = QtCore.Signal(dict)
    # Terminal.
    turn_finished = QtCore.Signal()
    error = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, graph, client, dispatcher, history, user_text: str, parent=None):
        super().__init__(parent)
        # Wrap the dispatcher with a main-thread proxy. Construction happens
        # on the main thread (ChatStreamWorker is built there; moveToThread
        # is applied AFTERWARDS to self). The proxy's QObject thread affinity
        # remains main, so BlockingQueuedConnection works correctly.
        self._dispatch_proxy = _MainThreadDispatchProxy(dispatcher)
        self._orch = ChatOrchestrator(
            graph=graph, client=client, dispatcher=self._dispatch_proxy,
            history=history,
        )
        self._user_text = user_text

    @QtCore.Slot()
    def run(self) -> None:
        """Thread entrypoint — connect QThread.started to this slot."""
        self._run_once()

    def request_cancel(self) -> None:
        self._orch.cancel()

    # Split out so tests can call it synchronously.
    def _run_once(self) -> None:
        try:
            for ev in self._orch.run_turn(self._user_text):
                if ev.kind == "text":
                    self.token_received.emit(ev.text or "")
                elif ev.kind == "tool_call_started":
                    self.tool_call_started.emit(ev.tool_name or "", ev.tool_input or {})
                elif ev.kind == "tool_call_finished":
                    self.tool_call_finished.emit(ev.tool_name or "", ev.tool_result or {})
                    if ev.tool_name == "generate_workflow" and ev.tool_result:
                        self.workflow_preview.emit(ev.tool_result)
                elif ev.kind == "cap_exceeded":
                    self.cap_exceeded.emit(ev.tool_name or "")
                elif ev.kind == "error":
                    self.error.emit(ev.error or "unknown error")
                elif ev.kind == "cancelled":
                    self.cancelled.emit()
                    break
                elif ev.kind == "turn_done":
                    break
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.turn_finished.emit()
