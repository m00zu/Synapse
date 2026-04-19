"""Mock the LLM client + dispatcher; assert WS event sequence."""
import asyncio
import types

import pytest

pytest.importorskip("PySide6")


class _FakeClient:
    """Reusable fake LLM client driven by a list of scripted streams.

    Each ``chat_with_tools_stream`` call pops the next stream from ``_scripts``
    and yields its events. This lets a single test drive a multi-round
    orchestrator loop (tool_call → tool_result → follow-up text) without
    hitting a real provider.
    """
    model = "fake-model"
    supports_vision = False

    def __init__(self, scripts):
        self._scripts = list(scripts)

    def chat_with_tools_stream(self, *a, **k):
        events = self._scripts.pop(0) if self._scripts else [
            types.SimpleNamespace(kind="done"),
        ]
        for ev in events:
            yield ev


async def _pump_until_done(ws_q, kinds=None, timeout=2.0, max_events=40):
    kinds = kinds if kinds is not None else []
    for _ in range(max_events):
        ev = await asyncio.wait_for(ws_q.get(), timeout=timeout)
        kinds.append(ev["kind"])
        if ev["kind"] == "chat_turn_done":
            break
    return kinds


@pytest.mark.asyncio
async def test_web_chat_session_emits_token_then_done(client):
    """A client that streams 'hello' + done yields chat_token then chat_turn_done."""
    from synapse.server.app import app
    from synapse.server.chat_session import WebChatSession

    session = app.state.session
    ws_q = session.bus.subscribe()

    fake = _FakeClient([[
        types.SimpleNamespace(kind="text", text="hel"),
        types.SimpleNamespace(kind="text", text="lo"),
        types.SimpleNamespace(kind="done"),
    ]])

    from synapse.ai.tools import ToolDispatcher
    dispatcher = ToolDispatcher()  # empty — no tools invoked in this test

    chat = WebChatSession(session)
    chat.start_turn(
        user_text="hi",
        client=fake,
        dispatcher=dispatcher,
        history=[],
    )

    try:
        kinds = await _pump_until_done(ws_q)
    finally:
        session.bus.unsubscribe(ws_q)
    assert "chat_token" in kinds
    assert kinds[-1] == "chat_turn_done"


@pytest.mark.asyncio
async def test_web_chat_session_modify_workflow_end_to_end(client):
    """C1 regression: modify_workflow add_node mutates session.graph.

    Guards against passing the NodeGraphHeadless façade (no ``create_node``)
    to ``_build_dispatcher_for_graph``. The full NodeGraphQt NodeGraph must
    be reachable via ``session.graph.node_graph``.
    """
    from synapse.server.app import app
    from synapse.server.chat_session import WebChatSession
    from synapse.llm_assistant import _build_dispatcher_for_graph

    session = app.state.session
    ws_q = session.bus.subscribe()

    # Stream 1: emit a modify_workflow tool_call adding one GaussianBlurNode.
    # Stream 2: emit final assistant text + done.
    fake = _FakeClient([
        [
            types.SimpleNamespace(
                kind="tool_call",
                tool_call={
                    "id": "call-1",
                    "name": "modify_workflow",
                    "input": {"operations": [
                        {"op": "add_node", "type": "GaussianBlurNode", "id": "n1"},
                    ]},
                },
            ),
        ],
        [
            types.SimpleNamespace(kind="text", text="done."),
            types.SimpleNamespace(kind="done"),
        ],
    ])

    dispatcher = _build_dispatcher_for_graph(
        session.graph.node_graph, client=fake,
    )

    chat = WebChatSession(session)
    chat.start_turn(
        user_text="add a blur",
        client=fake, dispatcher=dispatcher, history=[],
    )

    kinds: list[str] = []
    finishes: list[dict] = []
    try:
        for _ in range(40):
            ev = await asyncio.wait_for(ws_q.get(), timeout=3.0)
            kinds.append(ev["kind"])
            if ev["kind"] == "chat_tool_finish":
                finishes.append(ev)
            if ev["kind"] == "chat_turn_done":
                break
    finally:
        session.bus.unsubscribe(ws_q)

    assert "chat_tool_start" in kinds
    assert any(ev.get("status") == "ok" for ev in finishes), (
        f"modify_workflow dispatch failed; finishes={finishes}"
    )
    names = [n.name() for n in session.graph.all_nodes()]
    assert any("Gaussian" in n for n in names), (
        f"expected GaussianBlurNode on canvas, got {names}"
    )


@pytest.mark.asyncio
async def test_web_chat_session_persists_assistant_text_across_turns(client):
    """C3 regression: assistant prose is appended to session.chat_history
    after each turn, and history is not corrupted by in-place orchestrator
    mutations between turns."""
    from synapse.server.app import app
    from synapse.server.chat_session import WebChatSession
    from synapse.ai.tools import ToolDispatcher

    session = app.state.session
    # Reset history so we can assert exact contents.
    session.chat_history.clear()
    ws_q = session.bus.subscribe()

    dispatcher = ToolDispatcher()

    async def run_turn(text_out: str, user_text: str):
        # Route-side append of the user message (mirrors routes_chat.start_turn).
        session.chat_history.append({"role": "user", "content": user_text})
        fake = _FakeClient([[
            types.SimpleNamespace(kind="text", text=text_out),
            types.SimpleNamespace(kind="done"),
        ]])
        chat = WebChatSession(session)
        chat.start_turn(
            user_text=user_text, client=fake,
            dispatcher=dispatcher,
            history=list(session.chat_history),  # COPY, matches route
        )
        await _pump_until_done(ws_q)

    try:
        await run_turn("first answer", "hi")
        await run_turn("second answer", "again")
    finally:
        session.bus.unsubscribe(ws_q)

    roles = [m["role"] for m in session.chat_history]
    contents = [m["content"] for m in session.chat_history]
    assert roles == ["user", "assistant", "user", "assistant"], (
        f"history roles corrupted: {roles}"
    )
    assert contents == ["hi", "first answer", "again", "second answer"], (
        f"history contents corrupted: {contents}"
    )
    # No tool_use / tool_result blocks should have leaked through — every
    # message must be a simple string (the orchestrator only injects those
    # when it mutates the *working* history, which we now pass as a copy).
    assert all(isinstance(m["content"], str) for m in session.chat_history)
