"""WebChatSession — drive the ChatOrchestrator on a background thread and
emit its OrchestratorEvent stream as WS events via the session's EventBus.

This is the web-side analogue of desktop's ChatStreamWorker
(synapse/ai/chat_worker.py) minus the Qt bits. No _MainThreadDispatchProxy
is needed: tool dispatches run directly on the orchestrator's background
thread, because there's no Qt main thread to marshal to.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional


class WebChatSession:
    def __init__(self, session) -> None:
        self._session = session
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start_turn(self, user_text: str, client, dispatcher, history: list) -> str:
        """Run one turn on a daemon thread; publish events via session.bus.
        Returns a turn_id that the stop route can target (Phase 1e has at
        most one in-flight turn per session — the stop route just flips
        self._cancel)."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("A turn is already in flight. Call stop() first.")
        self._cancel.clear()
        turn_id = f"turn-{threading.get_ident()}"
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        self._thread = threading.Thread(
            target=self._run_turn,
            args=(user_text, client, dispatcher, history, turn_id),
            daemon=True,
        )
        self._thread.start()
        return turn_id

    def stop(self) -> None:
        self._cancel.set()

    def _run_turn(self, user_text, client, dispatcher, history, turn_id):
        from synapse.ai.orchestrator import ChatOrchestrator

        orch = ChatOrchestrator(
            graph=self._session.graph,
            client=client,
            dispatcher=dispatcher,
            history=history,
        )
        # Open bubble up front so the frontend has an id to stream into.
        bubble_id = f"b-{turn_id}"
        self._emit({"kind": "chat_turn_started", "bubble_id": bubble_id,
                    "turn_id": turn_id, "user_text": user_text})
        try:
            for ev in orch.run_turn(user_text):
                if self._cancel.is_set():
                    self._emit({"kind": "chat_turn_cancelled", "bubble_id": bubble_id})
                    return
                ws_ev = self._to_ws(ev, bubble_id)
                if ws_ev is not None:
                    self._emit(ws_ev)
        except Exception as exc:  # noqa: BLE001 — surface any crash as chat_error
            self._emit({"kind": "chat_error", "bubble_id": bubble_id, "error": str(exc)})
        finally:
            self._emit({"kind": "chat_turn_done", "bubble_id": bubble_id})

    def _to_ws(self, ev, bubble_id: str) -> Optional[dict]:
        """Translate an OrchestratorEvent to a WS-shaped chat_* event.

        Returns None for events that are handled internally (e.g. turn_done is
        emitted from the finally block instead) or that require no WS message.
        """
        if ev.kind == "text":
            return {"kind": "chat_token", "bubble_id": bubble_id, "text": ev.text or ""}
        if ev.kind == "tool_call_started":
            return {"kind": "chat_tool_start",
                    "bubble_id": bubble_id,
                    "chip_id": ev.tool_call_id or f"c-{id(ev)}",
                    "name": ev.tool_name or "",
                    "input": ev.tool_input or {}}
        if ev.kind == "tool_call_finished":
            result = ev.tool_result or {}
            # Emit a workflow preview event when generate_workflow completes.
            if ev.tool_name == "generate_workflow" and result:
                self._emit({"kind": "chat_workflow_preview",
                            "bubble_id": bubble_id,
                            "result": result})
            status = "error" if (isinstance(result, dict) and "error" in result) else "ok"
            return {"kind": "chat_tool_finish",
                    "bubble_id": bubble_id,
                    "chip_id": ev.tool_call_id or f"c-{id(ev)}",
                    "status": status,
                    "result": result}
        if ev.kind == "cap_exceeded":
            return {"kind": "chat_cap_hit",
                    "bubble_id": bubble_id,
                    "tool_name": ev.tool_name or ""}
        if ev.kind == "error":
            return {"kind": "chat_error", "bubble_id": bubble_id,
                    "error": ev.error or ""}
        if ev.kind in ("turn_done", "cancelled"):
            # turn_done is emitted from the finally block; cancelled is handled
            # by the cancel check above. Either way, no extra WS event needed here.
            return None
        return {"kind": "chat_unknown", "bubble_id": bubble_id}

    def _emit(self, event: dict) -> None:
        """Publish on the session's EventBus.
        EventBus.publish is async; dispatch it onto the loop from our thread."""
        bus = self._session.bus
        if self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(bus.publish(event), self._loop)
            try:
                fut.result(timeout=1.0)
            except Exception:
                pass
        # No loop available — best-effort dropped. Tests use inline runs.
