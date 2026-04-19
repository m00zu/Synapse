"""GET /api/nodes + /api/nodes/categories."""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["nodes"])


# Maps the first two segments of __identifier__ to a user-facing category.
# The mapping is intentionally hand-curated (not auto-derived) so new
# packages can pick a display-friendly label without renaming their package.
_CATEGORY_MAP = {
    "nodes.io": "I/O",
    "nodes.data": "I/O",
    "nodes.dataframe": "Table",
    "nodes.image_process": "Image",
    "nodes.analysis": "Stats",
    "nodes.plotting": "Plot",
    "nodes.display": "Display",
    "nodes.utility": "Utility",
    "nodes.Collection": "Collection",
    "plugins": "Plugins",
}


def _category_for(identifier: str) -> str:
    """Derive a display category from a node's __identifier__.

    The identifier is dotted (e.g. 'nodes.image_process.filter'). We try the
    longest prefix first, then fall back to the second segment, then 'Other'.
    """
    if not identifier:
        return "Other"
    parts = identifier.split(".")
    # Try 2-segment prefix first (nodes.image_process), then 1 (plugins).
    for n in (2, 1):
        key = ".".join(parts[:n])
        if key in _CATEGORY_MAP:
            return _CATEGORY_MAP[key]
    return "Other"


@router.get("/nodes")
async def get_nodes(request: Request) -> dict:
    """Return the widget catalog — {class_name: [spec_dict, ...]}.

    Lazily built on first call; the catalog construction instantiates every
    registered node class, which is expensive (~3s) and crashes some Qt
    setups if done eagerly alongside NodeGraphQt's own registration path.
    """
    from synapse.server.app import _get_catalog
    return _get_catalog(request.app)


@router.get("/nodes/categories")
async def get_node_categories(request: Request) -> dict:
    """Return ``{class_name: {identifier, category, display_name}}`` for every
    registered node. Frontend uses this to group the palette and label nodes
    with their human-readable names (from ``NODE_NAME``) instead of raw
    Python class names."""
    # Reuse the same subclass walk that catalog uses.
    from synapse.widgets.catalog import (
        _iter_subclasses, _install_legacy_shims, _import_all_plugins,
    )
    _install_legacy_shims()
    _import_all_plugins()
    from synapse.nodes.base import BaseExecutionNode
    out: dict[str, dict] = {}
    for cls in _iter_subclasses(BaseExecutionNode):
        ident = getattr(cls, "__identifier__", "") or ""
        display = getattr(cls, "NODE_NAME", "") or cls.__name__
        out[cls.__name__] = {
            "identifier": ident,
            "category": _category_for(ident),
            "display_name": display,
        }
    return out
