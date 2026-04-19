import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/api/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = websocket.app.state.session
    q = session.bus.subscribe()
    try:
        async with anyio.create_task_group() as tg:

            async def _pump_events():
                """Forward bus events to the WebSocket."""
                while True:
                    ev = await q.get()
                    await websocket.send_json(ev)

            async def _watch_disconnect():
                """Wait for client disconnect, then cancel the event pump."""
                try:
                    await websocket.receive()
                except WebSocketDisconnect:
                    pass
                tg.cancel_scope.cancel()

            tg.start_soon(_pump_events)
            tg.start_soon(_watch_disconnect)
    except (WebSocketDisconnect, anyio.ClosedResourceError):
        pass
    finally:
        session.bus.unsubscribe(q)
