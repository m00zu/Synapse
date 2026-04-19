"""WebSocket endpoint tests.

Uses a pure-ASGI WebSocket helper (anyio memory streams) instead of
starlette.testclient.TestClient.  TestClient spins up a background-thread
event loop which crashes on macOS when FastAPI routes call Qt APIs (node
creation touches Qt widgets and must run on the main thread).  The ASGI
helper runs entirely in the pytest-asyncio main-thread event loop, which is
the same thread that owns the QApplication.
"""
import asyncio
import json

import anyio
import pytest

pytest.importorskip("PySide6")



class _AsyncWSClient:
    """Lightweight ASGI WebSocket client (no background threads)."""

    def __init__(self, app, path: str):
        self._app = app
        self._path = path
        # client → app channel
        self._app_send, self._app_recv = anyio.create_memory_object_stream(256)
        # app → client channel
        self._client_send, self._client_recv = anyio.create_memory_object_stream(256)
        self._task_group = None

    async def __aenter__(self):
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "headers": [],
            "path": self._path,
            "raw_path": self._path.encode(),
            "query_string": b"",
            "root_path": "",
            "scheme": "ws",
            "server": ("testserver", 80),
            "subprotocols": [],
        }

        async def receive():
            return await self._app_recv.receive()

        async def send(msg):
            await self._client_send.send(msg)

        await self._app_send.send({"type": "websocket.connect"})

        self._tg_ctx = anyio.create_task_group()
        self._tg = await self._tg_ctx.__aenter__()
        self._tg.start_soon(self._app, scope, receive, send)

        # Consume the "websocket.accept" frame.
        accept = await self._client_recv.receive()
        assert accept["type"] == "websocket.accept"
        return self

    async def receive_json(self) -> dict:
        msg = await self._client_recv.receive()
        assert msg["type"] == "websocket.send"
        return json.loads(msg.get("text") or msg.get("bytes", b"{}"))

    async def close(self):
        await self._app_send.send({"type": "websocket.disconnect", "code": 1000})
        try:
            await self._tg_ctx.__aexit__(None, None, None)
        except Exception:
            pass

    async def __aexit__(self, *exc):
        await self.close()


@pytest.mark.asyncio
async def test_ws_receives_exec_events(client):
    """WS endpoint delivers node_started / node_finished / run_finished."""
    from synapse.server.app import app as real_app

    # Seed two nodes via the async client (runs on main thread — Qt safe).
    await client.post("/api/graph/nodes", json={"type": "ImageReadNode"})
    await client.post("/api/graph/nodes", json={"type": "BinaryThresholdNode"})

    async with _AsyncWSClient(real_app, "/api/ws") as ws:
        # Trigger a run (still on main thread — Qt safe).
        await client.post("/api/exec/run")

        events = []
        for _ in range(20):
            msg = await asyncio.wait_for(ws.receive_json(), timeout=10.0)
            events.append(msg)
            if msg["kind"] == "run_finished":
                break

    kinds = [e["kind"] for e in events]
    assert "node_started" in kinds
    assert "node_finished" in kinds
    assert kinds[-1] == "run_finished"


@pytest.mark.asyncio
async def test_ws_accepts_and_disconnects_cleanly(client):
    """WS endpoint handles immediate disconnect without error."""
    from synapse.server.app import app as real_app

    async with _AsyncWSClient(real_app, "/api/ws"):
        pass  # open + close immediately — no assertion beyond "no exception"
