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


def _subcategory_for(identifier: str) -> str:
    """Return everything in the identifier after the category-mapping prefix.

    Examples:
      nodes.image_process.filter      → 'filter'
      nodes.dataframe.Combine         → 'Combine'
      nodes.io                        → ''           (no sub-category)
      plugins.Report          → 'Report'     (strip redundant 'Plugins')
      nodes.image_process.sub.nested  → 'sub.nested' (preserves deeper nesting)
    """
    if not identifier:
        return ""
    parts = identifier.split(".")
    # Match the longest category prefix first.
    for n in (2, 1):
        key = ".".join(parts[:n])
        if key in _CATEGORY_MAP:
            remainder = parts[n:]
            # 'plugins.Report' -- the 'Plugins' segment after the
            # 'plugins' namespace is redundant noise; strip it so the sub-
            # category is just 'Report'.
            if remainder and remainder[0] == "Plugins":
                remainder = remainder[1:]
            return ".".join(remainder)
    return ""


@router.get("/nodes")
async def get_nodes(request: Request) -> dict:
    """Return the widget catalog -- {class_name: [spec_dict, ...]}.

    Lazily built on first call; the catalog construction instantiates every
    registered node class, which is expensive (~3s) and crashes some Qt
    setups if done eagerly alongside NodeGraphQt's own registration path.
    """
    from synapse.server.app import _get_catalog
    return _get_catalog(request.app)


def _port_list(cls, instance, side: str) -> list[dict]:
    """Return ``[{name, type}]`` for the given side's ports.

    Prefers the runtime port names from *instance* (what ``add_input`` /
    ``add_output`` actually created), then resolves each port's type by
    positionally matching against ``PORT_SPEC[side]``. Falls back to PORT_SPEC
    alone if the instance isn't available. Mirrors the logic in
    ``synapse.widgets.catalog._auto_preview_for``.
    """
    spec_side = (getattr(cls, "PORT_SPEC", None) or {}).get(side) or []
    # Flatten PORT_SPEC entries to a name→type dict + positional order.
    named: dict[str, str] = {}
    types_in_order: list[str] = []
    for p in spec_side:
        if isinstance(p, dict):
            n = p.get("name") or p.get("type") or ""
            t = p.get("type") or ""
        else:
            n = p
            t = p
        types_in_order.append(t)
        if n:
            named[n] = t
    if instance is not None:
        try:
            port_fn = instance.inputs if side == "inputs" else instance.outputs
            runtime_names = list(port_fn().keys())
        except Exception:
            runtime_names = []
        result: list[dict] = []
        for i, name in enumerate(runtime_names):
            t = named.get(name)
            if t is None and i < len(types_in_order):
                t = types_in_order[i]
            result.append({"name": name, "type": t or "any"})
        return result
    # No instance -- just emit what PORT_SPEC declared.
    return [{"name": n, "type": t or "any"} for n, t in named.items()]


@router.get("/nodes/categories")
async def get_node_categories(request: Request) -> dict:
    """Return ``{class_name: {identifier, category, display_name, inputs, outputs}}``
    for every registered node. Frontend uses this to group the palette, label
    nodes by ``NODE_NAME``, and render the correct number of handles per
    side (one per port) with type-based colors."""
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
        # Try to instantiate so we see real runtime port names (e.g.
        # SplitRGBNode's red/green/blue instead of PORT_SPEC's "image/image/image").
        instance = None
        try:
            instance = cls()
        except Exception:
            pass
        out[cls.__name__] = {
            "identifier": ident,
            "category": _category_for(ident),
            "subcategory": _subcategory_for(ident),
            "display_name": display,
            "inputs": _port_list(cls, instance, "inputs"),
            "outputs": _port_list(cls, instance, "outputs"),
        }
    return out
