"""GET /api/nodes — return the widget catalog."""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["nodes"])


@router.get("/nodes")
async def get_nodes(request: Request) -> dict:
    """Return the widget catalog — {class_name: [spec_dict, ...]}."""
    return request.app.state.catalog
