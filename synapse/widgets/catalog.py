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

from synapse.widgets.spec import spec_to_json

logger = logging.getLogger(__name__)


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
            result[cls.__name__] = []
            continue
        result[cls.__name__] = [spec_to_json(s) for s in node.get_widget_spec()]
    return result
