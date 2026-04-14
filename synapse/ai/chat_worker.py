"""Qt QThread wrapper for the ChatOrchestrator. Emits signals for each
orchestrator event so the chat panel can update the UI on the main thread."""
from __future__ import annotations

from PySide6 import QtCore

from synapse.ai.orchestrator import ChatOrchestrator


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
        self._orch = ChatOrchestrator(
            graph=graph, client=client, dispatcher=dispatcher, history=history,
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
