"""
conftest.py for tests/rdkit_nodes

ViewerBridge lives in plugins/rdkit_nodes/viewer_nodes.py.  That module has
a module-level ``from nodes.base import ...`` which pulls in the full
Synapse node graph machinery — Qt widgets, scikit-image, etc.  None of that
is needed to exercise the pure-Qt ViewerBridge class.

This conftest stubs out the heavy dependencies before any test module is
imported so that ``from plugins.rdkit_nodes.viewer_nodes import ViewerBridge``
succeeds in a lean pytest environment (no installed synapse package, no RDKit,
no QWebEngineView, etc.).

Only the minimal surface used by viewer_nodes.py at module scope is faked:
  - nodes / nodes.base           (BaseExecutionNode, PORT_COLORS, NodeBaseWidget)
  - data_models                  (NodeData)
  - plugins.rdkit_nodes.protein_data  (ProteinData, ReceptorData, DockingResultData)

The real PySide6.QtCore is still required (that's what we're testing).
"""
from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
_project_root = Path(__file__).parents[2]  # .../PySide_Node
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── Stub: nodes / nodes.base ─────────────────────────────────────────────────
if 'nodes' not in sys.modules:
    _nodes_pkg = types.ModuleType('nodes')
    sys.modules['nodes'] = _nodes_pkg

if 'nodes.base' not in sys.modules:
    _nodes_base = types.ModuleType('nodes.base')
    _nodes_base.BaseExecutionNode = object
    _nodes_base.PORT_COLORS = {}

    class _FakeNodeBaseWidget:  # bare-minimum stand-in
        pass

    _nodes_base.NodeBaseWidget = _FakeNodeBaseWidget
    sys.modules['nodes.base'] = _nodes_base

# ── Stub: data_models ────────────────────────────────────────────────────────
if 'data_models' not in sys.modules:
    _dm = types.ModuleType('data_models')
    _dm.NodeData = object
    sys.modules['data_models'] = _dm

# ── Stub: plugins / plugins.rdkit_nodes (package only, no __init__ logic) ────
if 'plugins' not in sys.modules:
    sys.modules['plugins'] = types.ModuleType('plugins')

if 'plugins.rdkit_nodes' not in sys.modules:
    sys.modules['plugins.rdkit_nodes'] = types.ModuleType('plugins.rdkit_nodes')

# ── Stub: plugins.rdkit_nodes.protein_data ───────────────────────────────────
if 'plugins.rdkit_nodes.protein_data' not in sys.modules:
    _pdata = types.ModuleType('plugins.rdkit_nodes.protein_data')
    _pdata.ProteinData = object
    _pdata.ReceptorData = object
    _pdata.DockingResultData = object
    sys.modules['plugins.rdkit_nodes.protein_data'] = _pdata

# ── Load viewer_nodes directly (bypassing the rdkit_nodes __init__) ──────────
_viewer_nodes_path = _project_root / 'plugins' / 'rdkit_nodes' / 'viewer_nodes.py'
_spec = importlib.util.spec_from_file_location(
    'plugins.rdkit_nodes.viewer_nodes',
    _viewer_nodes_path,
)
_viewer_nodes_mod = importlib.util.module_from_spec(_spec)
sys.modules['plugins.rdkit_nodes.viewer_nodes'] = _viewer_nodes_mod
_spec.loader.exec_module(_viewer_nodes_mod)
