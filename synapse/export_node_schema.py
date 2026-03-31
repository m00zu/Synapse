import sys
import json
from PySide6 import QtWidgets
from NodeGraphQt import NodeGraph

from .nodes.base import PORT_COLORS, BaseExecutionNode

# Reverse map: (r, g, b) → type name, used to annotate port types in the schema
_COLOR_TO_TYPE: dict[tuple, str] = {tuple(v): k for k, v in PORT_COLORS.items()}

def _port_type(port) -> str:
    """Return the data-type name for a port by reverse-looking up its color."""
    return _COLOR_TO_TYPE.get(tuple(port.color), "any")


def _discover_node_classes():
    """Dynamically discover all node classes from core and plugins."""
    import inspect
    import NodeGraphQt

    classes = []
    seen = set()

    def _collect(module):
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (name not in seen
                    and hasattr(obj, '__identifier__')
                    and hasattr(obj, 'NODE_NAME')
                    and issubclass(obj, NodeGraphQt.BaseNode)):
                seen.add(name)
                classes.append(obj)

    # Core nodes
    from . import custom_nodes
    _collect(custom_nodes)

    # Bundled plugins
    for plugin_name in ['image_analysis', 'statistical_analysis',
                        'figure_plotting', 'filopodia_nodes',
                        'data_processing']:
        try:
            mod = __import__(f'synapse.plugins.{plugin_name}',
                             fromlist=[plugin_name])
            _collect(mod)
        except Exception as e:
            print(f"  [skip] {plugin_name}: {e}")

    return classes


def export_schema():
    # 1. Initialize compat shims (same as app.py)
    from synapse import nodes as _nodes_pkg, data_models as _dm_pkg, custom_nodes as _cn_pkg
    from synapse.nodes import base as _nodes_base_pkg
    sys.modules.setdefault('nodes', _nodes_pkg)
    sys.modules.setdefault('nodes.base', _nodes_base_pkg)
    sys.modules.setdefault('data_models', _dm_pkg)
    sys.modules.setdefault('custom_nodes', _cn_pkg)

    # 2. Initialize a dummy QApplication so PySide widgets can instantiate without crashing
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)

    # 3. Create the NodeGraph and register all discovered nodes
    graph = NodeGraph()
    nodes_to_register = _discover_node_classes()

    for cls in nodes_to_register:
        try:
            graph.register_node(cls)
        except Exception:
            pass  # skip nodes that fail to register (e.g. duplicates)

    schema = {
        "name": "build_workflow",
        "description": "Constructs a node graph execution workflow.",
        "parameters": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "List of nodes to create.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Sequential integer ID (1, 2, 3, …)."
                            },
                            "type": {
                                "type": "string",
                                "description": "The EXACT class type of the node.",
                                "enum": []
                            },
                            "props": {
                                "type": "object",
                                "description": "Optional configurable properties."
                            }
                        },
                        "required": ["id", "type"],
                        "allOf": []
                    }
                },
                "edges": {
                    "type": "array",
                    "description": "Connections as [source_id, target_id] pairs. Ports auto-resolved by type.",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2
                    }
                }
            }
        },
        "node_catalog": {}
    }

    # Reference to arrays in the dict
    enum_list = schema["parameters"]["properties"]["nodes"]["items"]["properties"]["type"]["enum"]
    all_of_list = schema["parameters"]["properties"]["nodes"]["items"]["allOf"]
    
    # These base properties are standard in the UI and should be ignored by the LLM
    ignore_props = [
        'name', 'color', 'border_color', 'text_color', 'type', 'id', 'pos',
        'layout_direction', 'selected', 'visible', 'custom', 'progress',
        'table_view', 'image_view', 'show_preview', 'live_preview'
    ]
    
    # Classes that should never appear in the LLM catalog
    _SKIP_CLASSES = {
        'BaseExecutionNode', 'BaseImageProcessNode',
    }

    # 4. Iterate and instantiate each node temporarily to extract metadata
    for identifier, node_cls in graph.node_factory.nodes.items():
        if not hasattr(node_cls, 'NODE_NAME') or not node_cls.NODE_NAME:
            continue
        if node_cls.__name__ in _SKIP_CLASSES:
            continue
            
        try:
            # Instantiate
            node = graph.create_node(identifier, push_undo=False)
            
            # The class name or NODE_NAME can be used. LLMs generally prefer ClassNames
            node_type_name = node_cls.__name__ 
            enum_list.append(node_type_name)
            
            # Build output list, annotating table ports with known columns if declared
            out_col_map: dict = getattr(node_cls, 'OUTPUT_COLUMNS', {})
            outputs_info = []
            for p in node.outputs().values():
                entry = {"name": p.name(), "type": _port_type(p)}
                if p.name() in out_col_map:
                    entry["columns"] = out_col_map[p.name()]
                outputs_info.append(entry)

            node_info = {
                "description": getattr(node_cls, '__doc__', '').strip().split('\n')[0] if getattr(node_cls, '__doc__') else "No description",
                "inputs":  [{"name": p.name(), "type": _port_type(p)} for p in node.inputs().values()],
                "outputs": outputs_info,
                "configurable_properties": {}
            }
            
            # Extract only custom user-facing properties
            props = node.model.custom_properties
            custom_props_schema = {"type": "object", "properties": {}}
            
            # Grab combo-box item lists from the graph model's common properties
            common_props = {}
            if node.model._graph_model is not None:
                common_props = node.model._graph_model.get_node_common_properties(
                    node.model.type_
                ) or {}
            
            prop_descriptions = getattr(node_cls, 'PROP_DESCRIPTIONS', {})

            for k, v in props.items():
                if k not in ignore_props and not k.startswith('_'):
                    prop_type = type(v).__name__
                    if prop_type == 'NoneType':
                        prop_type = 'string' # default fallback

                    # Check if this property has a list of allowed options (combo box)
                    combo_items = common_props.get(k, {}).get('items', None)

                    catalog_entry = {"type": prop_type, "default": v}
                    if combo_items:
                        catalog_entry["options"] = combo_items
                    if k in prop_descriptions:
                        catalog_entry["description"] = prop_descriptions[k]
                    node_info["configurable_properties"][k] = catalog_entry

                    if prop_type == 'str':
                        json_type = 'string'
                    elif prop_type == 'bool':
                        json_type = 'boolean'
                    elif prop_type in ('int', 'float'):
                        json_type = 'number'
                    else:
                        json_type = 'string' # catch-all

                    prop_schema: dict = {"type": json_type}
                    if combo_items:
                        # Use enum so the LLM is constrained to valid values
                        prop_schema["enum"] = combo_items
                        base_desc = f"Default: {v}. Must be one of the listed options."
                    else:
                        base_desc = f"Default: {v}"
                    extra = prop_descriptions.get(k, "")
                    prop_schema["description"] = f"{base_desc} {extra}".strip() if extra else base_desc

                    custom_props_schema["properties"][k] = prop_schema
                    
            schema["node_catalog"][node_type_name] = node_info
            
            # Apply strict if-then JSON schema rules
            if custom_props_schema["properties"]:
                all_of_list.append({
                    "if": {
                        "properties": {"type": {"const": node_type_name}}
                    },
                    "then": {
                        "properties": {"custom": custom_props_schema}
                    }
                })
            
            # Cleanup
            graph.delete_node(node, push_undo=False)
            
        except Exception as e:
            print(f"Failed to extract info for {identifier}: {e}")

    # 5. Save the output
    import os
    out_file = os.path.join(os.path.dirname(__file__), 'llm_node_schema.json')
    # In frozen (Nuitka onefile) mode, the bundled dir is temp — write to persistent location
    if getattr(sys, 'frozen', False):
        persistent = _get_persistent_schema_path()
        if persistent:
            persistent.parent.mkdir(parents=True, exist_ok=True)
            out_file = str(persistent)
    with open(out_file, 'w') as f:
        json.dump(schema, f, indent=4)
        
    print(f"Generated successfully: {out_file}")
    print(f"Total Nodes Processed: {len(schema['node_catalog'])}")

def _get_persistent_schema_path():
    """Return a persistent path for the schema in frozen builds."""
    import pathlib, platform, os
    try:
        from .plugin_loader import get_plugin_dir
        return get_plugin_dir() / 'llm_node_schema.json'
    except Exception:
        pass
    # Fallback
    system = platform.system()
    if system == 'Darwin':
        base = pathlib.Path.home() / 'Library' / 'Application Support' / 'Synapse'
    elif system == 'Windows':
        base = pathlib.Path(os.environ.get('APPDATA', str(pathlib.Path.home()))) / 'Synapse'
    else:
        base = pathlib.Path.home() / '.synapse'
    return base / 'llm_node_schema.json'


def _is_schema_stale() -> bool:
    """Quick mtime check — returns True if any node .py is newer than the schema."""
    import pathlib
    # In frozen mode, check persistent path first; otherwise check bundled path
    schema_path = pathlib.Path(__file__).parent / 'llm_node_schema.json'
    if getattr(sys, 'frozen', False):
        persistent = _get_persistent_schema_path()
        if persistent and persistent.exists():
            schema_path = persistent
    if not schema_path.exists():
        return True

    schema_mtime = schema_path.stat().st_mtime

    check_dirs = [
        pathlib.Path(__file__).parent / 'nodes',
        pathlib.Path(__file__).parent / 'plugins',
    ]
    try:
        from .plugin_loader import get_plugin_dir
        user_plugin_dir = get_plugin_dir()
        if user_plugin_dir.exists() and user_plugin_dir not in check_dirs:
            check_dirs.append(user_plugin_dir)
    except Exception:
        pass

    for d in check_dirs:
        if not d.exists():
            continue
        for f in d.rglob('*.py'):
            if f.stat().st_mtime > schema_mtime:
                print(f"[schema] {f.name} changed — will regenerate in background")
                return True
    return False


def auto_regenerate_if_stale():
    """Check if schema is stale (fast), then regenerate in a subprocess if needed.

    The mtime check runs on the main thread (instant).
    The actual regeneration runs as a separate Python process so
    the UI is never blocked.
    """
    if not _is_schema_stale():
        return

    import subprocess
    print("[schema] node files changed — regenerating in background…")
    # Run export_node_schema.py as a subprocess (it has __main__ support)
    subprocess.Popen(
        [sys.executable, "-m", "synapse.export_node_schema"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == '__main__':
    export_schema()
