"""Mock the LLM client + dispatcher; assert WS event sequence."""
import asyncio
import pytest

pytest.importorskip("PySide6")


@pytest.mark.asyncio
async def test_web_chat_session_emits_token_then_done(client):
    """A client that streams 'hello' + done yields chat_token then chat_turn_done."""
    import types
    from synapse.server.app import app
    from synapse.server.chat_session import WebChatSession

    session = app.state.session
    ws_q = session.bus.subscribe()

    # Fake client: yields 2 text events then a done (orchestrator-internal kind).
    def fake_chat(system, messages, tools=None):
        yield types.SimpleNamespace(kind="text", text="hel")
        yield types.SimpleNamespace(kind="text", text="lo")
        yield types.SimpleNamespace(kind="done")

    class FakeClient:
        model = "fake-model"
        supports_vision = False

        def chat_with_tools_stream(self, *a, **k):
            return fake_chat(*a, **k)

    from synapse.ai.tools import ToolDispatcher
    dispatcher = ToolDispatcher()  # empty — no tools invoked in this test

    chat = WebChatSession(session)
    turn_id = chat.start_turn(
        user_text="hi",
        client=FakeClient(),
        dispatcher=dispatcher,
        history=[],
    )

    # Pump a few events.
    kinds: list[str] = []
    try:
        for _ in range(20):
            ev = await asyncio.wait_for(ws_q.get(), timeout=2.0)
            kinds.append(ev["kind"])
            if ev["kind"] == "chat_turn_done":
                break
    finally:
        session.bus.unsubscribe(ws_q)
    assert "chat_token" in kinds
    assert kinds[-1] == "chat_turn_done"
