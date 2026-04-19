"""CRUD routes for the session graph."""
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/graph", tags=["graph"])


class AddNodeReq(BaseModel):
    type: str
    x: float = 0
    y: float = 0


class ConnectReq(BaseModel):
    src: str
    dst: str
    src_port: Optional[str] = None
    dst_port: Optional[str] = None


@router.get("")
async def get_graph(request: Request) -> dict:
    return request.app.state.session.graph.export()


@router.post("/nodes", status_code=201)
async def add_node(request: Request, body: AddNodeReq) -> dict:
    session = request.app.state.session
    async with session.lock:
        try:
            nid = session.graph.add_node(body.type, body.x, body.y)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return {"id": nid}


@router.delete("/nodes/{node_id}", status_code=204)
async def delete_node(request: Request, node_id: str) -> None:
    session = request.app.state.session
    async with session.lock:
        try:
            session.graph.remove_node(node_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no such node: {node_id}")


@router.patch("/nodes/{node_id}/props")
async def patch_props(request: Request, node_id: str, props: dict[str, Any]) -> dict:
    session = request.app.state.session
    async with session.lock:
        try:
            for prop, value in props.items():
                session.graph.set_prop(node_id, prop, value)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no such node: {node_id}")
    return {"ok": True}


@router.post("/edges", status_code=201)
async def connect(request: Request, body: ConnectReq) -> dict:
    session = request.app.state.session
    async with session.lock:
        try:
            session.graph.connect(body.src, body.dst, body.src_port, body.dst_port)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such node: {exc}")
    return {"ok": True}


@router.delete("/edges")
async def disconnect(request: Request, body: ConnectReq) -> dict:
    session = request.app.state.session
    async with session.lock:
        try:
            session.graph.disconnect(body.src, body.dst, body.src_port, body.dst_port)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such node: {exc}")
    return {"ok": True}
