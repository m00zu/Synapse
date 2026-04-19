"""POST /api/exec/run, /api/exec/stop."""
import asyncio
import uuid
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/exec", tags=["exec"])


@router.post("/run", status_code=202)
async def run(request: Request) -> dict:
    session = request.app.state.session
    run_id = uuid.uuid4().hex[:8]
    asyncio.create_task(_drive_run(session, run_id))
    return {"run_id": run_id}


@router.post("/stop", status_code=204)
async def stop(request: Request) -> None:
    session = request.app.state.session
    if session.executor is not None:
        session.executor.request_stop()


async def _drive_run(session, run_id: str) -> None:
    from synapse.server.executor import Executor
    session.executor = Executor(session)
    async for ev in session.executor.run():
        await session.bus.publish(ev)
    await session.bus.publish({"kind": "run_finished", "run_id": run_id})
    session.executor = None
