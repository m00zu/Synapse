"""GET /api/nodes — return the widget catalog."""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["nodes"])


@router.get("/nodes")
async def get_nodes(request: Request) -> dict:
    """Return the widget catalog — {class_name: [spec_dict, ...]}.

    Lazily built on first call; the catalog construction instantiates every
    registered node class, which is expensive (~3s) and crashes some Qt
    setups if done eagerly alongside NodeGraphQt's own registration path.
    """
    from synapse.server.app import _get_catalog
    return _get_catalog(request.app)
