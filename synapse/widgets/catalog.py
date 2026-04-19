"""Walk every registered BaseExecutionNode subclass and emit its widget spec
as a JSON-serializable dict, keyed by class name.

Instantiates each class once (throwaway) to run its __init__ and capture the
spec via the _spec_builder side-effect wired up in synapse.nodes.base. This
is wasteful in absolute terms but the catalog is built once per server
startup, so it doesn't matter in practice.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

from synapse.widgets.spec import spec_to_json, Preview

logger = logging.getLogger(__name__)

_PREVIEW_KINDS = {"image", "table", "figure"}


def _auto_preview_for(cls, instance=None) -> list:
    """Emit one Preview spec per image/table/figure output port.

    Prefers the runtime port names from the instance (actual ports registered
    via add_output) over the PORT_SPEC type hints when available.  This handles
    nodes like FileReadNode / ImageReadNode that list ``['table']`` or
    ``['image']`` in PORT_SPEC but actually register a port named ``'out'``.

    When no instance is available (instantiation failed) falls back to
    PORT_SPEC alone — name == type in that case, which may not match the
    executor's preview filenames, but is the best we can do.
    """
    out: list = []
    ports_type_hints = (getattr(cls, "PORT_SPEC", None) or {}).get("outputs") or []

    # Build name→type map from PORT_SPEC.
    # For string entries the single string serves as BOTH name and type.
    type_by_spec_name: dict[str, str] = {}
    spec_types_in_order: list[str] = []  # ordered list of types from PORT_SPEC
    for p in ports_type_hints:
        if isinstance(p, dict):
            n = p.get("name") or p.get("type") or ""
            t = p.get("type") or ""
        else:
            n = t = str(p)
        if n:
            type_by_spec_name[n] = t
        if t:
            spec_types_in_order.append(t)

    if instance is not None:
        try:
            real_port_names = list(instance.outputs().keys())
        except Exception:
            real_port_names = []

        if real_port_names:
            # Map each real port name to a type, using PORT_SPEC for guidance.
            # Strategy: if the real name exists as a PORT_SPEC key, use that type.
            # Otherwise, fall back positionally to spec_types_in_order.
            for idx, port_name in enumerate(real_port_names):
                ptype = type_by_spec_name.get(port_name)
                if ptype is None and idx < len(spec_types_in_order):
                    ptype = spec_types_in_order[idx]
                if ptype in _PREVIEW_KINDS:
                    out.append(Preview(preview_kind=ptype, source=f"output:{port_name}"))
            return out

    # No instance (or no real ports found) — fall back to PORT_SPEC alone.
    for name, ptype in type_by_spec_name.items():
        if ptype in _PREVIEW_KINDS:
            out.append(Preview(preview_kind=ptype, source=f"output:{name}"))
    return out


def _iter_subclasses(cls):
    """Recursively yield every subclass of *cls*."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def _install_legacy_shims() -> None:
    """Inject module aliases required by plugin files that use bare imports.

    Plugins written before the synapse package was created import via bare
    names like ``from data_models import …`` or ``from nodes.base import …``.
    Registering aliases in sys.modules lets those imports succeed without
    modifying the plugin source files.
    """
    if "data_models" not in sys.modules:
        from synapse import data_models as _dm
        sys.modules.setdefault("data_models", _dm)
    if "nodes" not in sys.modules:
        from synapse import nodes as _nodes
        sys.modules.setdefault("nodes", _nodes)
    if "nodes.base" not in sys.modules:
        from synapse.nodes import base as _nodes_base
        sys.modules.setdefault("nodes.base", _nodes_base)
    if "custom_nodes" not in sys.modules:
        from synapse import custom_nodes as _cn
        sys.modules.setdefault("custom_nodes", _cn)


def _import_all_plugins() -> None:
    """Import every bundled plugin so their node subclasses get registered."""
    plugin_package_dirs = [
        "image_analysis",
        "statistical_analysis",
        "figure_plotting",
        "data_processing",
    ]
    for name in plugin_package_dirs:
        try:
            importlib.import_module(f"synapse.plugins.{name}")
        except Exception as e:
            logger.warning("catalog: failed to import plugin %s: %s", name, e)

    # Standalone .py plugin files directly under synapse/plugins/
    plugins_dir = Path(__file__).parent.parent / "plugins"
    for py_file in sorted(plugins_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        name = py_file.stem
        try:
            importlib.import_module(f"synapse.plugins.{name}")
        except Exception as e:
            logger.warning("catalog: failed to import plugin %s: %s", name, e)


def collect_widget_catalog() -> dict[str, list[dict]]:
    """Return ``{node_class_name: [spec_dict, ...]}`` for every registered node.

    Silently skips classes whose __init__ raises (plugin init errors should not
    break the catalog build). An empty list means the node has no widgets.
    """
    from synapse.nodes.base import BaseExecutionNode
    # Install shims so plugin bare-import files can load.
    try:
        _install_legacy_shims()
    except Exception as e:
        logger.warning("catalog: failed to install legacy shim: %s", e)
    _import_all_plugins()
    result: dict[str, list[dict]] = {}
    for cls in _iter_subclasses(BaseExecutionNode):
        try:
            node = cls()
        except Exception as e:
            # A node that can't be bare-instantiated (missing Qt, missing files,
            # abstract helpers, …) gets an empty entry so the catalog still
            # lists its name. This is preferable to silently omitting it.
            logger.warning(
                "catalog: %s could not be instantiated: %s",
                cls.__name__, e, exc_info=True,
            )
            # Still emit auto-preview specs based on PORT_SPEC alone.
            result[cls.__name__] = [
                spec_to_json(p) for p in _auto_preview_for(cls)
            ]
            continue
        result[cls.__name__] = (
            [spec_to_json(s) for s in node.get_widget_spec()]
            + [spec_to_json(p) for p in _auto_preview_for(cls, node)]
        )
    return result
