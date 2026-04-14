"""
llm_assistant.py — Local SLM integration for Synapse via Ollama.

Provides:
  - OllamaClient       : HTTP client for the local Ollama API
  - WorkflowLoader     : Creates nodes + edges in the NodeGraph from LLM JSON
  - LLMWorker          : QObject background-thread worker (mirrors GraphWorker pattern)
  - LLMAssistantPanel  : QWidget dock panel (model picker, question, JSON preview, load button)

"""

# Testing Prompt:
# 1: Detect blood vessels in a microscopy image and measure their lengths.
# 2.1: I have an image with cell in red channel and collagen in green channel. Extract the surrounding area of the cell and calculate the number of pixels with intensiity higher than 185 within this area.
# 2.2: Create a batch version of this workflow for all oir file under a directory and accumulate all results into a single dataframe.

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Secure API-key storage helpers (obfuscated JSON file + env var)
# ---------------------------------------------------------------------------
import base64 as _b64

# Map provider names to conventional env-var names
_ENV_VAR_MAP = {
    "OpenAI":      "OPENAI_API_KEY",
    "Groq":        "GROQ_API_KEY",
    "Gemini":      "GEMINI_API_KEY",
    "RunPod":      "RUNPOD_API_KEY",
    "Ollama Cloud": "OLLAMA_API_KEY",
}

_KEYS_PATH = Path(__file__).parent.parent / ".api_keys"


def _load_keys_file() -> dict:
    try:
        return json.loads(_KEYS_PATH.read_text())
    except Exception:
        return {}


def _store_api_key(provider: str, key: str):
    """Save an API key to an obfuscated local file."""
    keys = _load_keys_file()
    if key:
        keys[provider] = _b64.b85encode(key.encode()).decode()
    else:
        keys.pop(provider, None)
    _KEYS_PATH.write_text(json.dumps(keys))


def _retrieve_api_key(provider: str, json_fallback: str = "") -> str:
    """Load an API key: local file → env var → JSON fallback."""
    # 1. Obfuscated file
    keys = _load_keys_file()
    if provider in keys:
        try:
            return _b64.b85decode(keys[provider].encode()).decode()
        except Exception:
            pass
    # 2. Environment variable
    env_var = _ENV_VAR_MAP.get(provider)
    if env_var:
        val = os.environ.get(env_var, "")
        if val:
            return val
    # 3. JSON fallback (legacy config)
    return json_fallback
from PySide6 import QtCore, QtGui, QtWidgets

# ---------------------------------------------------------------------------
# Schema path — check multiple locations for compatibility across:
#   pip install (site-packages/synapse/), source run, Nuitka frozen build
# ---------------------------------------------------------------------------
def _find_schema() -> Path:
    # In frozen builds, prefer the persistent (user-data) copy over the bundled one,
    # since bundled files live in a temp dir and can't be updated.
    if getattr(sys, 'frozen', False) or "__compiled__" in globals():
        try:
            from .export_node_schema import _get_persistent_schema_path
            persistent = _get_persistent_schema_path()
            if persistent and persistent.exists():
                return persistent
        except Exception:
            pass
    candidates = [
        Path(__file__).parent / "llm_node_schema.json",   # pip install / source
        Path(__file__).parent.parent / "llm_node_schema.json",  # source root
    ]
    if "__compiled__" in globals():
        candidates.append(Path(sys.executable).parent / "llm_node_schema.json")  # Nuitka
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # fallback (will error on open)

_SCHEMA_PATH = _find_schema()

# Short system prompt for fine-tuned models (no catalog needed — knowledge is in weights)
_FINETUNE_SYS_PROMPT = (
    "You are a workflow assistant for Synapse, a scientific node-graph editor.\n"
    "Respond with ONLY a JSON object. No markdown, no explanation.\n\n"
    "Output format:\n"
    '{"nodes": [{"id": 1, "type": "ClassName"}, {"id": 2, "type": "...", "props": {"key": "val"}}], "edges": [[1,2], [2,3]]}\n\n'
    "Rules:\n"
    "- \"id\": sequential integers (1, 2, 3, …)\n"
    "- \"type\": exact node class name\n"
    "- \"props\": optional, only when non-default values needed\n"
    "- \"edges\": [[src, dst], ...] — ports auto-resolved by type. "
    "For multi-output nodes, add port hint: [src, dst, \"port_name\"] e.g. [2, 3, \"red\"]\n"
    "- <image> ≠ <mask>: always threshold first\n"
    "- Terminal nodes: table→DataTableCellNode, image→ImageCellNode, figure→DataFigureCellNode\n"
    "- After thresholding, consider FillHolesNode + RemoveSmallObjectsNode\n"
    "- WatershedNode outputs table + label_image — never add ParticlePropsNode after it\n"
    "- Plot nodes use 'data' as table input port"
)

# ---------------------------------------------------------------------------
# Response schema — sent as Ollama's `format` parameter.
# Stripped of allOf rules; only describes the required output shape.
# ---------------------------------------------------------------------------
_NODE_SELECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of node class names relevant to the user's request.",
        },
    },
    "required": ["nodes"],
}

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":    {"type": "integer"},
                    "type":  {"type": "string"},
                    "props": {"type": "object"},
                },
                "required": ["id", "type"],
            },
        },
        "edges": {
            "type": "array",
            "description": "Connections as [src, dst] or [src, dst, \"out_port\"] or [src, dst, \"out_port\", \"in_port\"] for multi-output/input nodes.",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
            },
        },
    },
    "required": ["nodes", "edges"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_class_docs() -> dict[str, str]:
    """
    Returns a mapping of ClassName → docstring for ALL registered node classes
    (core nodes + plugins).  Preserves paragraph structure but strips excessive
    indentation and the trailing Keywords line.
    Falls back to an empty dict if the import fails (e.g., missing Qt display).
    """
    try:
        from .nodes.base import BaseExecutionNode
    except Exception:
        return {}

    docs: dict[str, str] = {}
    # Walk all subclasses (core + plugins — all loaded at this point)
    for cls in BaseExecutionNode.__subclasses_recursive__() if hasattr(
        BaseExecutionNode, '__subclasses_recursive__'
    ) else _iter_all_subclasses(BaseExecutionNode):
        name = cls.__name__
        raw = getattr(cls, "__doc__", None) or ""
        if not raw.strip():
            continue
        # Strip leading/trailing whitespace per line, drop empty leading lines
        lines = [ln.strip() for ln in raw.splitlines()]
        # Remove trailing "Keywords: ..." line (not useful for the LLM)
        while lines and lines[-1].lower().startswith("keywords"):
            lines.pop()
        # Drop trailing empty lines
        while lines and not lines[-1]:
            lines.pop()
        # Drop leading empty lines
        while lines and not lines[0]:
            lines.pop(0)
        cleaned = "\n".join(lines)
        if cleaned:
            docs[name] = cleaned
    return docs


def _iter_all_subclasses(cls):
    """Recursively yield all subclasses of *cls*."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_all_subclasses(sub)


def _get_all_node_names(schema_path: Path = _SCHEMA_PATH) -> list[str]:
    """Return all node class names from the schema."""
    try:
        with open(schema_path, encoding="utf-8") as fh:
            catalog = json.load(fh).get("node_catalog", {})
        return list(catalog.keys())
    except Exception:
        return []


_catalog_cache: dict[str, str] = {}  # keyed by f"{schema_path}:{verbose}"

def build_condensed_catalog(
    schema_path: Path = _SCHEMA_PATH,
    verbose: bool = False,
) -> str:
    """
    Reads llm_node_schema.json and returns a compact one-line-per-node catalog
    suitable for embedding in an LLM system prompt.

    Compact mode (default): port types only, key props only.
    Verbose mode: includes full docstrings and all configurable properties.
    """
    cache_key = f"{schema_path}:{verbose}"
    if cache_key in _catalog_cache:
        return _catalog_cache[cache_key]

    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)

    class_docs: dict[str, str] = _load_class_docs() if verbose else {}

    def _fmt_ports(ports) -> str:
        parts = []
        for p in ports:
            if isinstance(p, dict):
                ptype = p.get('type', 'any')
                parts.append(f"{p['name']}<{ptype}>")
            else:
                parts.append(str(p))
        return ", ".join(parts) or "—"

    # Properties that are frequently needed by the LLM
    _KEY_PROPS = {
        'operation', 'method', 'separator', 'x_col', 'y_col', 'group_col',
        'value_col', 'order', 'palette', 'stain', 'sigma', 'radius',
        'min_distance', 'col_prefix', 'per_channel', 'auto_otsu_per_image',
        'thresh_state', 'channels', 'inner',
    }

    catalog = schema.get("node_catalog", {})
    lines: list[str] = []
    for name, info in catalog.items():
        if verbose and name in class_docs:
            desc = class_docs[name]
        else:
            desc = info.get("description", "").strip()

        inputs  = _fmt_ports(info.get("inputs",  []))
        outputs = _fmt_ports(info.get("outputs", []))

        prop_str = ""
        if verbose:
            cfg = info.get("configurable_properties", {})
            prop_parts = []
            for prop_name, prop_info in cfg.items():
                options = prop_info.get("options")
                default = prop_info.get("default", "")
                default_str = f'"{default}"' if isinstance(default, str) else str(default)
                if options:
                    prop_parts.append(f"{prop_name}=[{' | '.join(options)}](default:{default_str})")
                else:
                    prop_parts.append(f"{prop_name}={default_str}")
            prop_str = f" | props:{{{', '.join(prop_parts)}}}" if prop_parts else ""
        else:
            # Compact: only include key props
            cfg = info.get("configurable_properties", {})
            prop_parts = []
            for prop_name, prop_info in cfg.items():
                if prop_name not in _KEY_PROPS:
                    continue
                options = prop_info.get("options")
                if options:
                    prop_parts.append(f"{prop_name}=[{'|'.join(options)}]")
                else:
                    prop_parts.append(prop_name)
            prop_str = f" | props:{{{', '.join(prop_parts)}}}" if prop_parts else ""

        lines.append(f"- {name}: {desc} | in:[{inputs}] → out:[{outputs}]{prop_str}")

    # Append plugin nodes not already in the schema (e.g. user-installed third-party plugins)
    try:
        from .plugin_loader import get_plugin_catalog_entries
        for entry in get_plugin_catalog_entries():
            if entry['class_name'] in catalog:
                continue  # already in schema — skip to avoid duplicates
            ins  = _fmt_ports(entry['inputs'])
            outs = _fmt_ports(entry['outputs'])
            # Build props string from configurable_properties (same format as schema entries)
            cfg = entry.get('configurable_properties', {})
            prop_parts = []
            for prop_name, prop_info in cfg.items():
                options = prop_info.get('options')
                default = prop_info.get('default', '')
                default_str = f'"{default}"' if isinstance(default, str) else str(default)
                if options:
                    prop_parts.append(f"{prop_name}=[{'|'.join(options)}](default:{default_str})")
                else:
                    prop_parts.append(f"{prop_name}(default:{default_str})")
            prop_str = f" | props: {{{', '.join(prop_parts)}}}" if prop_parts else ""
            lines.append(
                f"- {entry['class_name']}: {entry['description']} | in:[{ins}] → out:[{outs}]{prop_str}"
            )
    except ImportError:
        pass

    result = "\n".join(lines)
    _catalog_cache[cache_key] = result
    return result


def build_system_prompt(catalog_text: str) -> str:
    # Fan-out example: node 1 connects to both node 5 and node 6
    example_mask = json.dumps({
        "nodes": [
            {"id": 1, "type": "ImageReadNode"},
            {"id": 2, "type": "SplitRGBNode"},
            {"id": 3, "type": "EqualizeAdapthistNode"},
            {"id": 4, "type": "BinaryThresholdNode"},
            {"id": 5, "type": "KeepMaxIntensityRegionNode"},
            {"id": 6, "type": "ImageMathNode", "props": {"operation": "A × B (apply mask)"}},
            {"id": 7, "type": "ImageCellNode"}
        ],
        "edges": [[1,2], [2,3], [3,4], [4,5], [3,5], [5,6], [1,6], [6,7]]
    }, indent=2)

    # Stats with fan-out: node 2 feeds both node 3 AND node 4
    example_stats = json.dumps({
        "nodes": [
            {"id": 1, "type": "FileReadNode", "props": {"separator": ","}},
            {"id": 2, "type": "OutlierDetectionNode"},
            {"id": 3, "type": "PairwiseComparisonNode"},
            {"id": 4, "type": "BarPlotNode", "props": {"x_col": "Group", "y_col": "Value"}},
            {"id": 5, "type": "DataFigureCellNode"}
        ],
        "edges": [[1,2], [2,3], [2,4], [3,4], [4,5]]
    }, indent=2)

    # Nucleus segmentation: linear chain with fan-out at node 7
    example_nuclei = json.dumps({
        "nodes": [
            {"id": 1, "type": "ImageReadNode"},
            {"id": 2, "type": "RollingBallNode", "props": {"radius": 50}},
            {"id": 3, "type": "GaussianBlurNode", "props": {"sigma": 2.0}},
            {"id": 4, "type": "BinaryThresholdNode"},
            {"id": 5, "type": "FillHolesNode"},
            {"id": 6, "type": "RemoveSmallObjectsNode"},
            {"id": 7, "type": "WatershedNode", "props": {"min_distance": 12}},
            {"id": 8, "type": "ImageCellNode"},
            {"id": 9, "type": "DataTableCellNode"}
        ],
        "edges": [[1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [7,9]]
    }, indent=2)

    # Batch colocalization: FolderIterator + SplitRGB fan-out + accumulator
    example_batch_coloc = json.dumps({
        "nodes": [
            {"id": 1, "type": "FolderIteratorNode"},
            {"id": 2, "type": "ImageReadNode"},
            {"id": 3, "type": "SplitRGBNode"},
            {"id": 4, "type": "RollingBallNode"},
            {"id": 5, "type": "RollingBallNode"},
            {"id": 6, "type": "GaussianBlurNode"},
            {"id": 7, "type": "GaussianBlurNode"},
            {"id": 8, "type": "ColocalizationNode"},
            {"id": 9, "type": "BatchAccumulatorNode"},
            {"id": 10, "type": "DataTableCellNode"}
        ],
        "edges": [[1,2], [2,3], [3,4,"red"], [3,5,"green"],
                  [4,6], [5,7], [6,8,"ch1"], [7,8,"ch2"], [8,9], [9,10]]
    }, indent=2)

    # Plate reader: data cleaning + stats + fan-out to plot with significance
    example_plate_reader = json.dumps({
        "nodes": [
            {"id": 1, "type": "FileReadNode"},
            {"id": 2, "type": "BlankSubtractNode"},
            {"id": 3, "type": "GroupNormalizationNode"},
            {"id": 4, "type": "OutlierDetectionNode"},
            {"id": 5, "type": "GroupedComparisonNode"},
            {"id": 6, "type": "BarPlotNode"},
            {"id": 7, "type": "FigureEditNode"},
            {"id": 8, "type": "DataFigureCellNode"}
        ],
        "edges": [[1,2], [2,3], [3,4], [4,5,"kept"], [4,6,"kept","data"],
                  [5,6,"stats"], [6,7], [7,8]]
    }, indent=2)

    # Standard curve: two FileReadNodes, regression + predict + plot fan-out
    example_std_curve = json.dumps({
        "nodes": [
            {"id": 1, "type": "FileReadNode"},
            {"id": 2, "type": "LinearRegressionNode", "props": {"degree": 2}},
            {"id": 3, "type": "FileReadNode"},
            {"id": 4, "type": "RegressionPlotNode"},
            {"id": 5, "type": "DataFigureCellNode"},
            {"id": 6, "type": "ModelPredictNode", "props": {"inverse": True, "x_col": "OD"}},
            {"id": 7, "type": "DataTableCellNode"}
        ],
        "edges": [[1,2], [1,4,"data"], [2,4,"curve","curve"], [4,5],
                  [2,6,"model"], [3,6,"data"], [6,7]]
    }, indent=2)

    # Batch particle analysis with size filtering
    example_batch_particle = json.dumps({
        "nodes": [
            {"id": 1, "type": "FolderIteratorNode"},
            {"id": 2, "type": "ImageReadNode"},
            {"id": 3, "type": "BinaryThresholdNode"},
            {"id": 4, "type": "RemoveSmallObjectsNode"},
            {"id": 5, "type": "ParticlePropsNode"},
            {"id": 6, "type": "FilterTableNode", "props": {"query": "area > 100"}},
            {"id": 7, "type": "BatchAccumulatorNode"},
            {"id": 8, "type": "DataTableCellNode"}
        ],
        "edges": [[1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8]]
    }, indent=2)

    return (
        "You are a workflow assistant for Synapse, a scientific node-graph editor.\n"
        "Respond with ONLY a JSON object: {\"nodes\": [...], \"edges\": [...]}.\n\n"
        "FORMAT:\n"
        "- nodes: [{\"id\": 1, \"type\": \"ClassName\"}, {\"id\": 2, \"type\": \"...\", \"props\": {\"key\": val}}]\n"
        "- edges: [[src_id, dst_id], ...] — ports are auto-resolved by type matching. "
        "For multi-output nodes (e.g. SplitRGBNode), add port hint: [src, dst, \"port_name\"] e.g. [2, 3, \"red\"]\n"
        "- 'props' is optional — only include when the user specifies non-default values.\n"
        "- IDs are integers: 1, 2, 3, …\n\n"
        "RULES:\n"
        "1. Use ONLY class names from the catalog below.\n"
        "2. Port types must match: <image>≠<mask>. Always threshold first: image→BinaryThresholdNode→mask.\n"
        "3. WatershedNode outputs table+label_image — never add ParticlePropsNode after it.\n"
        "4. Plot nodes use 'data' as table input. Terminal nodes: table→DataTableCellNode, image→ImageCellNode, figure→DataFigureCellNode.\n"
        "5. SplitRGBNode for channels; ColorDeconvolutionNode ONLY for named stains (H&E, DAB, etc.).\n"
        "6. After thresholding, consider FillHolesNode + RemoveSmallObjectsNode before measurement.\n"
        "7. For fluorescence: RollingBallNode before threshold. For noise: GaussianBlurNode. For low contrast: EqualizeAdapthistNode.\n"
        "8. thresh_state: [value, direction] (1=above, 0=below). Set auto_otsu_per_image:false when using explicit threshold.\n"
        "9. AVOID MathColumnNode when a dedicated node exists. Prefer: NormalizeColumnNode (Z-score, min-max, log), "
        "BlankSubtractNode, GroupNormalizationNode, AggregateTableNode (sum, mean, auc), TwoTableMathNode.\n"
        "10. PythonScriptNode: ONLY use when no dedicated node can do the job. Dynamic ports via n_inputs/n_outputs props. "
        "script_code prop contains the Python code. Variables: in_1, in_2 (DataFrames/ndarrays), assign to out_1, out_2. "
        "Pre-imported: pd, np, scipy, skimage, PIL, plt. "
        "Type wrappers: TableData, ImageData, MaskData, FigureData, StatData. "
        "Use for: custom formulas (2**(-ddCt)), regex parsing, scipy functions not available as nodes, conditional multi-output splits.\n"
        "11. For multi-input nodes, add input port hint as 4th element: [src, dst, \"out_port\", \"in_port\"]. "
        "Use \"\" for output hint if only input needs disambiguation: [src, dst, \"\", \"in_port\"].\n"
        "12. JSON only: true/false/null. No markdown fences.\n\n"
        f"Example 1 — CSV → stats → bar plot (fan-out: node 2→3 and 2→4):\n{example_stats}\n\n"
        f"Example 2 — mask pipeline with fan-out (node 1→2 and 1→6):\n{example_mask}\n\n"
        f"Example 3 — nucleus segmentation (fan-out: node 7→8 and 7→9):\n{example_nuclei}\n\n"
        f"Example 4 — batch colocalization (SplitRGB fan-out + accumulator):\n{example_batch_coloc}\n\n"
        f"Example 5 — plate reader analysis (data cleaning + stats → plot with significance):\n{example_plate_reader}\n\n"
        f"Example 6 — standard curve (two inputs, regression + predict fan-out):\n{example_std_curve}\n\n"
        f"Example 7 — batch particle analysis (folder → threshold → measure → accumulate):\n{example_batch_particle}\n\n"
        "Node catalog:\n"
        f"{catalog_text}"
    )


# ---------------------------------------------------------------------------
# Two-pass helpers: node selection (pass 1) → detailed prompt (pass 2)
# ---------------------------------------------------------------------------

def build_selection_prompt(catalog_text: str) -> str:
    """
    Build a lightweight prompt for Pass 1: given the user's request,
    pick which nodes from the catalog are relevant.
    """
    return (
        "You are a node selector for Synapse, a scientific node-graph editor.\n"
        "Given the user's request, select ALL node class names from the catalog "
        "that could be needed to build the workflow.\n\n"
        "IMPORTANT: Be generous — select 15-25 nodes. It is much better to "
        "include a node that might not be needed than to miss one that is. "
        "Include nodes for every plausible interpretation of the request.\n\n"
        "Always include:\n"
        "- Nodes directly requested (e.g. blur → GaussianBlurNode)\n"
        "- Prerequisite nodes (e.g. mask tasks need BinaryThresholdNode)\n"
        "- Terminal/display nodes (DataTableCellNode, ImageCellNode, DataFigureCellNode)\n"
        "- I/O nodes if the user mentions files or folders (ImageReadNode, FileReadNode, FolderIteratorNode)\n"
        "- Save/export nodes if the user mentions saving, exporting, or generating output (SaveNode, SaveFigureNode, SaveTableNode, ReportNode)\n"
        "- Common preprocessing nodes for the domain (RollingBallNode, GaussianBlurNode for images; DropFillNaNNode, OutlierDetectionNode for tables)\n"
        "- Alternative nodes that could serve the same purpose\n\n"
        "Respond with ONLY a JSON object: {\"nodes\": [\"ClassName1\", \"ClassName2\", ...]}\n"
        "No markdown, no explanation.\n\n"
        "Node catalog:\n"
        f"{catalog_text}"
    )


def build_detailed_cards(
    node_names: list[str],
    schema_path: Path = _SCHEMA_PATH,
) -> str:
    """
    Build detailed description cards for a list of selected nodes.
    Pulls full class docstrings (when available), property info, and output
    columns from the schema.
    """
    try:
        with open(schema_path, encoding="utf-8") as fh:
            catalog = json.load(fh).get("node_catalog", {})
    except Exception:
        return ""

    # Load full class docstrings (richer than schema's one-line description)
    class_docs = _load_class_docs()

    lines: list[str] = []
    for name in node_names:
        info = catalog.get(name)
        if not info:
            continue

        # Prefer full docstring; fall back to schema description
        desc = class_docs.get(name) or info.get("description", "").strip()
        lines.append(f"\n### {name}")
        lines.append(f"  {desc}")

        # Inputs with types
        inputs = info.get("inputs", [])
        if inputs:
            parts = []
            for p in inputs:
                if isinstance(p, dict):
                    parts.append(f"{p['name']}<{p.get('type', 'any')}>")
                else:
                    parts.append(str(p))
            lines.append(f"  Inputs: {', '.join(parts)}")

        # Outputs with types and columns
        outputs = info.get("outputs", [])
        if outputs:
            for p in outputs:
                if isinstance(p, dict):
                    cols = p.get("columns")
                    col_str = f" — columns: [{', '.join(cols)}]" if cols else ""
                    lines.append(
                        f"  Output '{p['name']}' <{p.get('type', 'any')}>{col_str}"
                    )

        # All configurable properties with full detail
        cfg = info.get("configurable_properties", {})
        if cfg:
            lines.append("  Properties:")
            for pname, pinfo in cfg.items():
                ptype = pinfo.get("type", "")
                default = pinfo.get("default", "")
                options = pinfo.get("options")
                pdesc = pinfo.get("description", "")

                detail_parts = []
                if ptype:
                    detail_parts.append(ptype)
                if options:
                    detail_parts.append(f"options=[{' | '.join(str(o) for o in options)}]")
                if default != "":
                    def_str = f'"{default}"' if isinstance(default, str) else str(default)
                    detail_parts.append(f"default={def_str}")
                if pdesc:
                    detail_parts.append(f"— {pdesc}")

                lines.append(f"    {pname}: {', '.join(detail_parts)}")

    if not lines:
        return ""
    return "\nDETAILED INFO for selected nodes:\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Ollama HTTP client — moved to synapse/ai/clients/ollama.py
# ---------------------------------------------------------------------------

from synapse.ai.clients.ollama import OllamaClient  # re-export

# ---------------------------------------------------------------------------
# OpenAI client — moved to synapse/ai/clients/openai.py
# ---------------------------------------------------------------------------

from synapse.ai.clients.openai import OpenAIClient  # re-export

# ---------------------------------------------------------------------------
# LlamaCpp client  (local GGUF model — no Ollama, no Torch, CPU-friendly)
# ---------------------------------------------------------------------------
#
# Install:  pip install llama-cpp-python
#   Mac/Linux prebuilt wheels (CPU):
#       pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
#   Windows prebuilt:
#       pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
#
# Usage:
#   client = LlamaCppClient("path/to/synapse-qwen.gguf")
#   workflow_json = client.chat(system_prompt, user_prompt)
#
# Shipping with Nuitka:
#   --include-package=llama_cpp
#   --include-data-files=synapse-qwen.gguf=synapse-qwen.gguf
#

class LlamaCppClient:
    """
    Drop-in replacement for OllamaClient that loads a GGUF model directly
    via llama-cpp-python.  No server, no Torch, no Ollama required.

    The model is loaded once on first use and cached for subsequent calls.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096,
                 n_threads: int = 0, n_gpu_layers: int = 0):
        """
        model_path   : path to the .gguf file
        n_ctx        : context window in tokens (must cover system + user + JSON reply)
        n_threads    : CPU threads (0 = auto-detect)
        n_gpu_layers : layers to offload to GPU (0 = CPU-only, the default)
        """
        self.model_path   = model_path
        self.n_ctx        = n_ctx
        self.n_threads    = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._llm         = None   # lazy-loaded on first chat() call

    # ------------------------------------------------------------------
    def _load(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python"
            ) from e

        import os
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"GGUF model not found: {self.model_path}\n"
                "Fine-tune and export the model first (see finetune/train.py)."
            )

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads or None,   # None → llama.cpp auto-selects
            n_gpu_layers=self.n_gpu_layers,      # 0 = CPU-only
            verbose=False,
        )

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Returns the model filename so the UI can display it."""
        import os
        name = os.path.basename(self.model_path)
        return [name] if os.path.isfile(self.model_path) else []

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        """
        Runs inference and returns the raw JSON string.
        Raises FileNotFoundError if the GGUF file is missing.
        Note: images are ignored — GGUF models don't support vision.
        """
        self._load()
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},   # constrained JSON output
        )
        return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Groq client (OpenAI-compatible, cloud)
# ---------------------------------------------------------------------------

class GroqClient:
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    BASE_URL      = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model   = model

    def list_models(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            resp = requests.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []))
        except Exception:
            return []

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        if images:
            user_content: list | str = [{"type": "text", "text": user}]
            for b64 in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        else:
            user_content = user
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        resp = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def chat_multi(self, system: str, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": 0.1,
        }
        resp = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Gemini client (Google, cloud)
# ---------------------------------------------------------------------------

class GeminiClient:
    DEFAULT_MODEL = "gemini-2.5-flash-lite"   # fallback; real list populated via Refresh
    BASE_URL      = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model   = model

    def list_models(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            resp = requests.get(
                f"{self.BASE_URL}/models",
                params={"key": self.api_key},
                timeout=5,
            )
            resp.raise_for_status()
            return sorted(
                m["name"].replace("models/", "")
                for m in resp.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            )
        except Exception:
            return []

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        parts = [{"text": user}]
        if images:
            for b64 in images:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64,
                    }
                })
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }
        resp = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            # Gemini blocks responses via safety filters without raising an HTTP error
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "unknown")
            raise ValueError(f"Gemini returned no candidates (blockReason: {block_reason})")
        candidate = candidates[0]
        if "content" not in candidate:
            finish = candidate.get("finishReason", "unknown")
            raise ValueError(f"Gemini candidate has no content (finishReason: {finish})")
        return candidate["content"]["parts"][0]["text"]

    def chat_multi(self, system: str, messages: list[dict]) -> str:
        """Multi-turn chat for Gemini."""
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        # Convert messages to Gemini format
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.1},
        }
        resp = requests.post(url, params={"key": self.api_key},
                             json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"Gemini returned no candidates")
        candidate = candidates[0]
        if "content" not in candidate:
            raise ValueError(f"Gemini candidate has no content")
        return candidate["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# Claude / Anthropic client (cloud)
# ---------------------------------------------------------------------------

class ClaudeClient:
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    BASE_URL      = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model   = model

    def list_models(self) -> list[str]:
        """Return a curated list of Claude models (Anthropic has no list endpoint)."""
        if not self.api_key:
            return []
        return [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250506",
        ]

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        if images:
            user_content = [{"type": "text", "text": user}]
            for b64 in images:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                })
        else:
            user_content = user
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
        }
        resp = requests.post(
            f"{self.BASE_URL}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract text from the first content block
        content = data.get("content", [])
        if not content:
            raise ValueError(f"Claude returned no content (stop_reason: {data.get('stop_reason', 'unknown')})")
        return content[0].get("text", "")

    def chat_multi(self, system: str, messages: list[dict]) -> str:
        """Multi-turn chat for Claude."""
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
            "temperature": 0.1,
        }
        resp = requests.post(
            f"{self.BASE_URL}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if not content:
            raise ValueError(f"Claude returned no content")
        return content[0].get("text", "")


# ---------------------------------------------------------------------------
# RunPod serverless client (async polling)
# ---------------------------------------------------------------------------

class RunPodClient:
    """
    Client for RunPod serverless vLLM endpoints using the synchronous runsync API.

    Uses raw prompt completion (not OpenAI chat format) — the actual output format is:
      {"output": [{"choices": [{"tokens": ["..."]}], "usage": {...}}], "status": "COMPLETED"}

    System + user message are combined using the ChatML template (Qwen2.5 / instruct models).
    No model field is needed — the model is fixed by the RunPod deployment.
    """
    BASE_URL = "https://api.runpod.ai/v2"
    # Sentinel attributes so _on_model_changed doesn't crash; not used in requests
    model         = ""
    DEFAULT_MODEL = ""

    def __init__(self, api_key: str = "", endpoint_id: str = ""):
        self.api_key     = api_key
        self.endpoint_id = endpoint_id

    def _headers(self) -> dict:
        return {
            "Content-Type":  "application/json",
            "Authorization": f"{self.api_key}",
        }

    def list_models(self) -> list[str]:
        """No model list for RunPod — model is fixed at deployment."""
        return []

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        if not self.endpoint_id:
            raise ValueError("RunPod endpoint ID is not configured.")

        # ChatML / Qwen2.5 instruct template
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        payload = {
            "input": {
                "prompt": prompt,
                "sampling_params": {
                    "temperature": 0.1,
                    "max_tokens":  4096,
                    "stop":        ["<|im_end|>"],
                },
            }
        }
        url  = f"{self.BASE_URL}/{self.endpoint_id}/runsync"
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=120)
        resp.raise_for_status()
        data   = resp.json()
        status = data.get("status", "")
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise ValueError(f"RunPod job {status}: {data.get('error', 'no detail')}")

        # Output is a list: [{"choices": [{"tokens": ["..."]}]}]
        output = data.get("output", [])
        if isinstance(output, list) and output:
            choices = output[0].get("choices", [])
            if choices:
                tokens = choices[0].get("tokens", [])
                return "".join(tokens)
        raise ValueError(f"Unexpected RunPod output format: {output}")


# Properties that are framework-internal and should not be sent to the LLM
_IGNORE_PROPS: frozenset = frozenset([
    'name', 'color', 'border_color', 'text_color', 'type', 'id', 'pos',
    'layout_direction', 'selected', 'visible', 'custom', 'progress',
    'table_view', 'image_view', 'show_preview', 'live_preview',
])


def serialize_graph(graph) -> dict:
    """
    Converts the current node graph canvas into the compact LLM JSON format:
    {"nodes": [{"id": 1, "type": "...", "props": {...}}, ...], "edges": [[1,2], ...]}
    """
    all_nodes = graph.all_nodes()

    # Map internal NodeGraphQt UUID → sequential integer ID
    id_map: dict[str, int] = {}
    nodes_out: list[dict] = []
    edges_out: list[list] = []

    for idx, node in enumerate(all_nodes):
        llm_id = idx + 1
        id_map[node.id] = llm_id

        props: dict = {}
        try:
            for k, v in node.model.custom_properties.items():
                if k not in _IGNORE_PROPS and not k.startswith("_"):
                    props[k] = v
        except Exception:
            pass

        entry = {"id": llm_id, "type": type(node).__name__}
        if props:
            entry["props"] = props
        nodes_out.append(entry)

    for node in all_nodes:
        src_id = id_map.get(node.id)
        if src_id is None:
            continue
        for _, port in node.outputs().items():
            for connected in port.connected_ports():
                dst_id = id_map.get(connected.node().id)
                if dst_id:
                    edges_out.append([src_id, dst_id])

    return {"nodes": nodes_out, "edges": edges_out}


# ---------------------------------------------------------------------------
# WorkflowLoader — builds nodes + edges in the NodeGraph
# ---------------------------------------------------------------------------

class WorkflowLoader:
    X_PAD = 80   # horizontal gap between columns (px)
    Y_PAD = 60   # vertical gap between stacked nodes (px)

    def __init__(self, graph):
        self.graph = graph

    # ------------------------------------------------------------------
    # Port type compatibility for auto-wiring.
    # Maps each type to the set of types it can connect TO (as input).
    # Reflects data_models inheritance: StatData→TableData, MaskData→ImageData, etc.
    _TYPE_COMPAT = {
        'image':    {'image', 'any'},
        'mask':     {'mask', 'image', 'any'},
        'skeleton': {'skeleton', 'mask', 'image', 'any'},
        'label':    {'label', 'any'},
        'table':    {'table', 'any'},
        'stat':     {'stat', 'table', 'any'},
        'figure':   {'figure', 'any'},
        'confocal': {'confocal', 'any'},
        'path':     {'path', 'any'},
        'collection': {'collection', 'any'},
        'any':      {'any', 'image', 'mask', 'table', 'figure', 'label',
                     'stat', 'skeleton', 'confocal', 'path', 'collection'},
    }

    @staticmethod
    def _port_type(port) -> str:
        """Return the data-type name for a port by reverse-looking up its color."""
        from .nodes.base import PORT_COLORS
        _c2t = {tuple(v): k for k, v in PORT_COLORS.items()}
        return _c2t.get(tuple(port.color), 'any')

    @staticmethod
    def _resolve_port(ports: dict, hint: str):
        """Resolve a port hint to an actual port, using fuzzy matching.

        Tries in order:
          1. Exact match  — e.g. "A (image/mask)"
          2. Prefix match — e.g. "A" matches "A (image/mask)"
          3. Case-insensitive prefix match
        Returns the port object, or None.
        """
        if not hint:
            return None
        # Exact match
        if hint in ports:
            return ports[hint]
        # Prefix match (LLM often abbreviates "A (image/mask)" → "A")
        hint_lower = hint.lower()
        for name, port in ports.items():
            if name.startswith(hint) or name.lower().startswith(hint_lower):
                return port
        return None

    def _auto_wire(self, src_node, dst_node, used_pairs: set = None) -> tuple:
        """Find the best (out_port, in_port) pair by matching port types.

        Uses type compatibility (respects data model inheritance).
        Prefers exact matches, then compatible matches, then <any>.
        Skips already-connected input ports AND pairs in *used_pairs*.
        Returns (None, None) if no match found.
        """
        if used_pairs is None:
            used_pairs = set()

        out_ports = [(p, self._port_type(p)) for p in src_node.outputs().values()]
        in_ports = [(p, self._port_type(p)) for p in dst_node.inputs().values()
                     if not p.connected_ports()]

        def _skip(op, ip):
            return (op.name(), ip.name()) in used_pairs

        # Pass 1: exact type match
        for op, ot in out_ports:
            for ip, it in in_ports:
                if ot == it and not _skip(op, ip):
                    return op, ip

        # Pass 2: compatible type match (e.g. mask→image, stat→table)
        for op, ot in out_ports:
            compat = self._TYPE_COMPAT.get(ot, {'any'})
            for ip, it in in_ports:
                if it in compat and not _skip(op, ip):
                    return op, ip

        # Pass 3: any remaining unconnected input
        for op, ot in out_ports:
            for ip, it in in_ports:
                if not _skip(op, ip):
                    return op, ip

        return None, None

    # ------------------------------------------------------------------
    def build(
        self,
        workflow: dict,
        origin_x: int = 100,
        origin_y: int = 100,
    ) -> tuple[bool, str]:
        """
        Creates nodes and edges described by *workflow*.

        Supports two edge formats:
        - Compact: [[1, 2], [2, 3]] — ports auto-resolved by type matching
        - Verbose: [{"from_node_id": "n1", "from_port": "out", ...}] — legacy

        Returns (success, message).
        """
        node_map: dict = {}  # LLM id (str or int) → node instance
        warnings: list[str] = []
        created: list[tuple] = []  # (llm_id, node) in order

        # --- Pass 1: create nodes at origin, apply properties ------------
        for idx, node_def in enumerate(workflow.get("nodes", [])):
            llm_id     = node_def.get("id", idx + 1)
            class_name = node_def.get("type", "")
            # Support both "props" (new) and "custom" (legacy)
            custom     = node_def.get("props") or node_def.get("custom") or {}

            identifier = self._find_identifier(class_name)
            if identifier is None:
                warnings.append(f"Unknown node type '{class_name}' — skipped.")
                continue

            node = self.graph.create_node(identifier, push_undo=True)
            node.set_pos(origin_x, origin_y)

            # Suppress live_preview during property setup
            try:
                was_live = bool(node.get_property('live_preview'))
            except Exception:
                was_live = False
            if was_live:
                try:
                    node.set_property('live_preview', False)
                except Exception:
                    was_live = False

            for k, v in custom.items():
                try:
                    # push_undo=False to avoid undo stack errors on invalid props
                    node.set_property(k, v, push_undo=False)
                except Exception:
                    pass

            if was_live:
                try:
                    node.set_property('live_preview', True)
                except Exception:
                    pass

            # Store with both int and string keys for compat
            node_map[llm_id] = node
            node_map[str(llm_id)] = node
            # Also map "nX" format for legacy
            if isinstance(llm_id, int):
                node_map[f"n{llm_id}"] = node
            created.append((llm_id, node))

        # Force scene to compute node sizes before we read bounding rects
        QtWidgets.QApplication.processEvents()

        # --- Pass 2: topological layout based on graph depth ---------------
        # Assigns X by longest-path depth, Y by parent positions so that:
        #  - Linear chains stay on the same Y
        #  - Fan-out siblings are stacked vertically
        #  - Merge nodes sit at the average Y of their parents

        edges = workflow.get("edges", [])
        id_set = {llm_id for llm_id, _ in created}

        # Build adjacency
        children: dict[int | str, list] = {llm_id: [] for llm_id, _ in created}
        parents:  dict[int | str, list] = {llm_id: [] for llm_id, _ in created}
        for edge in edges:
            if not isinstance(edge, list) or len(edge) < 2:
                continue
            src, dst = edge[0], edge[1]
            if isinstance(src, str) and src.startswith("n") and src[1:].isdigit():
                src = int(src[1:])
            if isinstance(dst, str) and dst.startswith("n") and dst[1:].isdigit():
                dst = int(dst[1:])
            if src in id_set and dst in id_set:
                if dst not in children.get(src, []):
                    children.setdefault(src, []).append(dst)
                if src not in parents.get(dst, []):
                    parents.setdefault(dst, []).append(src)

        # Assign depth (column) = longest path from any root
        depth: dict = {}
        def _calc_depth(nid):
            if nid in depth:
                return depth[nid]
            p = parents.get(nid, [])
            if not p:
                depth[nid] = 0
            else:
                depth[nid] = max(_calc_depth(pid) for pid in p) + 1
            return depth[nid]

        for llm_id, _ in created:
            _calc_depth(llm_id)

        # Group nodes by depth column (preserve creation order within column)
        columns: dict[int, list] = {}
        for llm_id, node in created:
            d = depth.get(llm_id, 0)
            columns.setdefault(d, []).append((llm_id, node))

        # Measure node sizes
        node_sizes: dict = {}
        for llm_id, node in created:
            try:
                rect = node.view.boundingRect()
                node_sizes[llm_id] = (rect.width(), rect.height())
            except Exception:
                node_sizes[llm_id] = (200, 120)

        # --- Assign Y positions column by column, left-to-right -----------
        node_y: dict = {}   # nid → assigned Y

        # Compute X positions per column
        col_x: dict[int, float] = {}
        cur_x = float(origin_x)
        for col_idx in sorted(columns.keys()):
            col_x[col_idx] = cur_x
            col_max_w = max(node_sizes[nid][0] for nid, _ in columns[col_idx])
            cur_x += col_max_w + self.X_PAD

        for col_idx in sorted(columns.keys()):
            col_nodes = columns[col_idx]

            if col_idx == 0:
                # Root nodes: stack vertically centred around origin_y
                total_h = (
                    sum(node_sizes[nid][1] for nid, _ in col_nodes)
                    + self.Y_PAD * max(len(col_nodes) - 1, 0)
                )
                y = origin_y - total_h / 2
                for nid, node in col_nodes:
                    _, h = node_sizes[nid]
                    node_y[nid] = y
                    y += h + self.Y_PAD
            else:
                # Non-root nodes: Y derived from parents
                for nid, node in col_nodes:
                    p_ids = parents.get(nid, [])
                    p_with_y = [pid for pid in p_ids if pid in node_y]
                    if p_with_y:
                        # Average Y of all parents (merge nodes sit in the middle)
                        avg_y = sum(node_y[pid] for pid in p_with_y) / len(p_with_y)
                        node_y[nid] = avg_y
                    else:
                        node_y[nid] = float(origin_y)

                # Resolve overlaps within the column: push nodes apart
                col_sorted = sorted(col_nodes, key=lambda pair: node_y[pair[0]])
                for i in range(1, len(col_sorted)):
                    prev_id = col_sorted[i - 1][0]
                    curr_id = col_sorted[i][0]
                    prev_bottom = node_y[prev_id] + node_sizes[prev_id][1]
                    min_y = prev_bottom + self.Y_PAD
                    if node_y[curr_id] < min_y:
                        node_y[curr_id] = min_y

        # Apply positions
        for llm_id, node in created:
            d = depth.get(llm_id, 0)
            node.set_pos(col_x[d], node_y.get(llm_id, origin_y))

        # --- Pass 3: connect edges ----------------------------------------
        # Track used (out_port_name, in_port_name) per (src_id, dst_id) pair
        # so repeat edges between the same nodes use different ports
        _used_pairs: dict[tuple, set] = {}

        for edge in workflow.get("edges", []):
            if isinstance(edge, list):
                # Compact format:
                #   [src_id, dst_id]                  — auto-wire by type
                #   [src_id, dst_id, "out_port"]      — hint source port
                #   [src_id, dst_id, "out", "in"]     — hint both ports
                src_id, dst_id = edge[0], edge[1]
                hint_out = edge[2] if len(edge) > 2 and isinstance(edge[2], str) else None
                hint_in  = edge[3] if len(edge) > 3 and isinstance(edge[3], str) else None

                src_node = node_map.get(src_id) or node_map.get(str(src_id))
                dst_node = node_map.get(dst_id) or node_map.get(str(dst_id))

                if src_node is None or dst_node is None:
                    warnings.append(f"Edge {edge}: node not found — skipped.")
                    continue

                # If port hints provided, use them directly
                if hint_out or hint_in:
                    out_port = self._resolve_port(src_node.outputs(), hint_out) if hint_out else None
                    in_port  = self._resolve_port(dst_node.inputs(), hint_in) if hint_in else None

                    # LLMs often put the hint on the wrong side
                    # e.g. [2, 3, "", "red"] when "red" is an OUTPUT on node 2.
                    # Try swapping unresolved hints to the opposite side.
                    if not out_port and hint_in:
                        swapped = self._resolve_port(src_node.outputs(), hint_in)
                        if swapped:
                            out_port = swapped
                            in_port = None  # re-resolve input via auto-wire below
                    if not in_port and hint_out:
                        swapped = self._resolve_port(dst_node.inputs(), hint_out)
                        if swapped:
                            in_port = swapped
                            out_port = out_port  # keep if already resolved

                    # Auto-wire the still-missing side using type compatibility
                    if out_port and not in_port:
                        ot = self._port_type(out_port)
                        for ip in dst_node.inputs().values():
                            if ip.connected_ports():
                                continue
                            it = self._port_type(ip)
                            compat = self._TYPE_COMPAT.get(ot, {'any'})
                            if it == ot or it in compat:
                                in_port = ip
                                break
                    elif in_port and not out_port:
                        it = self._port_type(in_port)
                        for op in src_node.outputs().values():
                            ot = self._port_type(op)
                            compat = self._TYPE_COMPAT.get(ot, {'any'})
                            if ot == it or it in compat:
                                out_port = op
                                break

                    # Last resort: if hints failed entirely, fall through to auto-wire
                    if out_port is None and in_port is None:
                        pair_key = (src_id, dst_id)
                        used = _used_pairs.setdefault(pair_key, set())
                        out_port, in_port = self._auto_wire(src_node, dst_node, used)
                        if out_port and in_port:
                            used.add((out_port.name(), in_port.name()))
                else:
                    pair_key = (src_id, dst_id)
                    used = _used_pairs.setdefault(pair_key, set())
                    out_port, in_port = self._auto_wire(src_node, dst_node, used)
                    if out_port and in_port:
                        used.add((out_port.name(), in_port.name()))

                if out_port is None or in_port is None:
                    warnings.append(f"Edge {edge}: no compatible ports — skipped.")
                    continue

                try:
                    out_port.connect_to(in_port)
                except Exception as exc:
                    warnings.append(f"Edge {edge}: connect failed ({exc}) — skipped.")

            elif isinstance(edge, dict):
                # Legacy verbose format
                src_id   = edge.get("from_node_id", "")
                src_port = edge.get("from_port", "")
                dst_id   = edge.get("to_node_id", "")
                dst_port = edge.get("to_port", "")

                src_node = node_map.get(src_id)
                dst_node = node_map.get(dst_id)

                if src_node is None or dst_node is None:
                    warnings.append(f"Edge {src_id}.{src_port} → {dst_id}.{dst_port}: node not found — skipped.")
                    continue

                out_port = src_node.get_output(src_port)
                in_port  = dst_node.get_input(dst_port)

                if out_port is None or in_port is None:
                    warnings.append(f"Edge {src_id}.{src_port} → {dst_id}.{dst_port}: port not found — skipped.")
                    continue

                try:
                    out_port.connect_to(in_port)
                except Exception as exc:
                    warnings.append(f"Edge {src_id}.{src_port} → {dst_id}.{dst_port}: connect failed ({exc}) — skipped.")

        if warnings:
            return True, "Workflow loaded with warnings:\n" + "\n".join(f"  • {w}" for w in warnings)
        return True, "Workflow loaded successfully."

    # ------------------------------------------------------------------
    def _find_identifier(self, class_name: str) -> Optional[str]:
        """Looks up the node graph identifier string for a given class name."""
        for identifier, cls in self.graph.node_factory.nodes.items():
            if cls.__name__ == class_name:
                return identifier
        return None


# ---------------------------------------------------------------------------
# LLMWorker — runs the blocking Ollama HTTP call on a background QThread
# ---------------------------------------------------------------------------

class LLMWorker(QtCore.QObject):
    result  = QtCore.Signal(str)   # emits the raw JSON string
    error   = QtCore.Signal(str)   # emits a human-readable error message

    def __init__(self, client, system: str, user: str):
        super().__init__()
        self._client = client
        self._system = system
        self._user   = user

    # ------------------------------------------------------------------
    def run(self):
        try:
            json_str = self._client.chat(self._system, self._user)
            self.result.emit(json_str)
        except requests.exceptions.ConnectionError:
            self.error.emit(
                "Could not connect to the API endpoint.\n"
                "Check your network connection, base URL, or API key."
            )
        except requests.exceptions.Timeout:
            self.error.emit(
                "Request timed out (>120 s).\n"
                "The model may be too slow — try a smaller/faster model."
            )
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            # Include the response body — Gemini/Groq put helpful detail there
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text if exc.response is not None else ""
            if status == 401:
                self.error.emit(f"Authentication failed (HTTP 401).\nCheck your API key.\n{detail}")
            elif status == 400:
                self.error.emit(f"Bad request (HTTP 400).\n{detail}")
            elif status == 404:
                self.error.emit(
                    f"Model '{self._client.model}' not found (HTTP 404).\n"
                    f"Check the model name.\n{detail}"
                )
            else:
                self.error.emit(f"HTTP error {status}: {exc}\n{detail}")
        except ValueError as exc:
            self.error.emit(str(exc))
        except Exception:
            self.error.emit(f"Unexpected error:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# TwoPassLLMWorker — Pass 1: select nodes, Pass 2: generate with detail
# ---------------------------------------------------------------------------

class TwoPassLLMWorker(QtCore.QObject):
    """
    Two-pass workflow generation:
      Pass 1 — Send compact catalog + user question to the LLM and ask it to
               pick relevant node class names.  (cheap: small prompt, tiny output)
      Pass 2 — Build an enriched system prompt with detailed cards for only the
               selected nodes, then ask the LLM to generate the full workflow.
    """
    result   = QtCore.Signal(str)   # emits the raw JSON workflow string
    error    = QtCore.Signal(str)   # emits a human-readable error message
    progress = QtCore.Signal(str)   # status updates between passes

    def __init__(self, client, catalog_text: str, user: str):
        super().__init__()
        self._client       = client
        self._catalog_text = catalog_text
        self._user         = user

    # ------------------------------------------------------------------
    def run(self):
        try:
            # --- Pass 1: node selection ----------------------------------
            self.progress.emit("Pass 1/2 — selecting relevant nodes…")
            sel_system = build_selection_prompt(self._catalog_text)
            sel_raw = self._client.chat(sel_system, self._user)

            # Parse selected node names from response
            try:
                sel_json = json.loads(sel_raw)
                selected = sel_json.get("nodes", [])
                if not isinstance(selected, list):
                    selected = []
                # Filter to strings only
                selected = [n for n in selected if isinstance(n, str)]
            except (json.JSONDecodeError, AttributeError):
                # If the model returned a bare list or malformed JSON, try
                # to salvage node names from the raw text
                selected = []

            # --- Pass 2: generate workflow with enriched detail ----------
            # Detailed cards for selected nodes + a flat name-only list
            # of ALL other nodes as fallback (so the LLM can still use
            # nodes that Pass 1 missed, just without detailed guidance).
            if selected:
                self.progress.emit(
                    f"Pass 2/2 — generating workflow ({len(selected)} nodes selected)…"
                )
                detail = build_detailed_cards(selected)

                # Build a minimal fallback: just class names for unselected nodes
                all_names = _get_all_node_names()
                other = [n for n in all_names if n not in set(selected)]
                if other:
                    fallback = (
                        "\n\nOther available nodes (use if needed): "
                        + ", ".join(other)
                    )
                else:
                    fallback = ""

                gen_system = build_system_prompt(detail + fallback)
            else:
                self.progress.emit("Pass 2/2 — generating workflow (using full catalog)…")
                gen_system = build_system_prompt(self._catalog_text)
            json_str = self._client.chat(gen_system, self._user)

            self.result.emit(json_str)

        except requests.exceptions.ConnectionError:
            self.error.emit(
                "Could not connect to the API endpoint.\n"
                "Check your network connection, base URL, or API key."
            )
        except requests.exceptions.Timeout:
            self.error.emit(
                "Request timed out (>120 s).\n"
                "The model may be too slow — try a smaller/faster model."
            )
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text if exc.response is not None else ""
            if status == 401:
                self.error.emit(f"Authentication failed (HTTP 401).\nCheck your API key.\n{detail}")
            elif status == 400:
                self.error.emit(f"Bad request (HTTP 400).\n{detail}")
            elif status == 404:
                self.error.emit(
                    f"Model '{self._client.model}' not found (HTTP 404).\n"
                    f"Check the model name.\n{detail}"
                )
            else:
                self.error.emit(f"HTTP error {status}: {exc}\n{detail}")
        except ValueError as exc:
            self.error.emit(str(exc))
        except Exception:
            self.error.emit(f"Unexpected error:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# LLMAssistantPanel — dock-widget content
# ---------------------------------------------------------------------------

class LLMAssistantPanel(QtWidgets.QWidget):
    """
    Dock widget body for the AI Assistant.
    Accepts a NodeGraph instance so it can call WorkflowLoader after generation.
    """

    _PROVIDERS   = ("Ollama", "Ollama Cloud", "OpenAI", "Claude", "Groq", "Gemini", "RunPod", "Synapse Fine-tune")
    _GGUF_DIR    = Path(__file__).parent.parent / "finetune" / "output"
    _CONFIG_PATH = Path.home() / ".synapse_llm_config.json"

    def __init__(self, graph, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.graph    = graph
        self._client  = OllamaClient()
        self._catalog = build_condensed_catalog()
        self._system  = build_system_prompt(self._catalog)
        self._last_workflow: Optional[dict] = None  # parsed JSON from last generation
        self._config_model: str = ""        # model to restore on next _refresh_models call
        self._current_provider: str = "Ollama"  # tracks provider for key-save on switch

        self._build_ui()
        self._load_config()     # restore saved provider / API keys / model
        self._refresh_models()  # populate model list on startup

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Provider row ---------------------------------------------
        provider_row = QtWidgets.QHBoxLayout()
        provider_row.addWidget(QtWidgets.QLabel("Provider:"))
        self._provider_combo = QtWidgets.QComboBox()
        self._provider_combo.addItems(self._PROVIDERS)
        self._provider_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._provider_combo.currentTextChanged.connect(self._on_provider_changed)
        provider_row.addWidget(self._provider_combo)
        layout.addLayout(provider_row)

        # --- API key row (hidden for Ollama) --------------------------
        self._apikey_widget = QtWidgets.QWidget()
        apikey_row = QtWidgets.QHBoxLayout(self._apikey_widget)
        apikey_row.setContentsMargins(0, 0, 0, 0)
        apikey_row.addWidget(QtWidgets.QLabel("API Key:"))
        self._apikey_edit = QtWidgets.QLineEdit()
        self._apikey_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._apikey_edit.setPlaceholderText("Paste your API key here")
        apikey_row.addWidget(self._apikey_edit)
        self._apikey_widget.setVisible(False)
        layout.addWidget(self._apikey_widget)

        # --- GGUF path row (Synapse Fine-tune only) -------------------
        self._gguf_widget = QtWidgets.QWidget()
        gguf_row = QtWidgets.QHBoxLayout(self._gguf_widget)
        gguf_row.setContentsMargins(0, 0, 0, 0)
        gguf_row.addWidget(QtWidgets.QLabel("GGUF:"))
        self._gguf_edit = QtWidgets.QLineEdit()
        self._gguf_edit.setPlaceholderText("Path to .gguf file")
        gguf_row.addWidget(self._gguf_edit)
        gguf_browse_btn = QtWidgets.QPushButton("…")
        gguf_browse_btn.setFixedWidth(36)
        gguf_browse_btn.setToolTip("Browse for a .gguf model file")
        gguf_browse_btn.clicked.connect(self._browse_gguf)
        gguf_row.addWidget(gguf_browse_btn)
        self._gguf_widget.setVisible(False)
        layout.addWidget(self._gguf_widget)

        # --- Endpoint ID row (RunPod only) ----------------------------
        self._endpoint_widget = QtWidgets.QWidget()
        endpoint_row = QtWidgets.QHBoxLayout(self._endpoint_widget)
        endpoint_row.setContentsMargins(0, 0, 0, 0)
        endpoint_row.addWidget(QtWidgets.QLabel("Endpoint ID:"))
        self._endpoint_edit = QtWidgets.QLineEdit()
        self._endpoint_edit.setPlaceholderText("e.g. 9rv05hrgr8rn4o")
        endpoint_row.addWidget(self._endpoint_edit)
        self._endpoint_widget.setVisible(False)
        layout.addWidget(self._endpoint_widget)

        # --- Model row (hidden for RunPod — model is fixed at deployment) ---
        self._model_widget = QtWidgets.QWidget()
        model_row = QtWidgets.QHBoxLayout(self._model_widget)
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(QtWidgets.QLabel("Model:"))
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_combo)

        refresh_btn = QtWidgets.QPushButton("⟳")
        refresh_btn.setFixedWidth(36)
        refresh_btn.setToolTip("Refresh model list from the selected provider")
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(refresh_btn)
        layout.addWidget(self._model_widget)

        layout.addWidget(_make_separator())

        # --- Question input -------------------------------------------
        layout.addWidget(QtWidgets.QLabel("Describe the workflow you want to build:"))
        self._question_edit = QtWidgets.QPlainTextEdit()
        self._question_edit.setPlaceholderText(
            "e.g. Load a CSV, remove outliers, then create a grouped swarm plot"
        )
        self._question_edit.setFixedHeight(80)
        layout.addWidget(self._question_edit)

        # --- Context checkbox -----------------------------------------
        self._ctx_check = QtWidgets.QCheckBox("Include current workflow as context")
        self._ctx_check.setToolTip(
            "Serializes the current canvas and sends it to the model.\n"
            "Use 'Replace Canvas' to load the full revised workflow,\n"
            "or 'Load into Canvas' to append new nodes alongside existing ones."
        )
        layout.addWidget(self._ctx_check)

        # --- Verbose descriptions checkbox ----------------------------
        self._verbose_check = QtWidgets.QCheckBox("Verbose node descriptions")
        self._verbose_check.setToolTip(
            "Include full class docstrings in the node catalog sent to the model.\n"
            "Produces a larger prompt (~2× size) but gives the model richer context\n"
            "about each node's purpose and behaviour. Useful for complex queries."
        )
        self._verbose_check.stateChanged.connect(self._on_verbose_changed)
        layout.addWidget(self._verbose_check)

        # --- Generate / Copy buttons ----------------------------------
        btn_row = QtWidgets.QHBoxLayout()
        self._generate_btn = QtWidgets.QPushButton("Generate Workflow")
        self._generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._generate_btn)

        self._copy_web_btn = QtWidgets.QPushButton("Copy for Web AI")
        self._copy_web_btn.setToolTip(
            "Copy the full prompt to clipboard.\n"
            "Paste into ChatGPT, Claude.ai, Gemini, etc.\n"
            "Then paste the JSON response into the box below."
        )
        self._copy_web_btn.clicked.connect(self._on_copy_for_web)
        btn_row.addWidget(self._copy_web_btn)
        layout.addLayout(btn_row)

        layout.addWidget(_make_separator())

        # --- Status label ---------------------------------------------
        self._status_label = QtWidgets.QLabel("Ready")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # --- JSON preview (editable so users can paste responses) -----
        layout.addWidget(QtWidgets.QLabel("Workflow JSON (paste or generate):"))
        self._preview_edit = QtWidgets.QPlainTextEdit()
        self._preview_edit.setFont(QtGui.QFont("Courier New", 9))
        self._preview_edit.setMinimumHeight(160)
        self._preview_edit.textChanged.connect(self._on_preview_changed)
        layout.addWidget(self._preview_edit, stretch=1)

        # --- Load / Replace buttons (hidden until valid JSON is ready) ---
        load_row = QtWidgets.QHBoxLayout()
        self._load_btn = QtWidgets.QPushButton("Load into Canvas")
        self._load_btn.setToolTip("Append the generated nodes to the right of existing nodes.")
        self._load_btn.clicked.connect(self._on_load)
        self._load_btn.setVisible(False)
        load_row.addWidget(self._load_btn)

        self._replace_btn = QtWidgets.QPushButton("Replace Canvas")
        self._replace_btn.setToolTip(
            "Clear the entire canvas and load the generated workflow from scratch.\n"
            "Best used with 'Include current workflow as context' to apply improvements."
        )
        self._replace_btn.clicked.connect(self._on_replace)
        self._replace_btn.setVisible(False)
        load_row.addWidget(self._replace_btn)
        layout.addLayout(load_row)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------
    def _load_config(self):
        """Restore saved provider / API keys / model from disk."""
        try:
            cfg = json.loads(self._CONFIG_PATH.read_text())
        except Exception:
            return

        provider = cfg.get("provider", "Ollama")
        # Set provider combo without triggering _on_provider_changed
        self._provider_combo.blockSignals(True)
        idx = self._provider_combo.findText(provider)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)
        self._provider_combo.blockSignals(False)
        self._current_provider = provider

        # Show/hide credential rows for the loaded provider
        is_local    = (provider in ("Ollama", "Synapse Fine-tune"))
        is_runpod   = (provider == "RunPod")
        is_finetune = (provider == "Synapse Fine-tune")
        self._apikey_widget.setVisible(not is_local)
        self._gguf_widget.setVisible(is_finetune)
        self._endpoint_widget.setVisible(is_runpod)
        self._model_widget.setVisible(not is_runpod)

        # Restore API key (keyring → env var → legacy JSON fallback)
        json_key = cfg.get("api_keys", {}).get(provider, "")
        self._apikey_edit.setText(_retrieve_api_key(provider, json_key))
        self._endpoint_edit.setText(cfg.get("endpoint_ids", {}).get("RunPod", ""))
        self._gguf_edit.setText(cfg.get("gguf_path", ""))

        # Remember the model; _refresh_models will apply it
        self._config_model = cfg.get("last_models", {}).get(provider, "")

    def _save_config(self):
        """Persist current UI state to disk (best-effort, silently ignored on error)."""
        provider = self._provider_combo.currentText()
        try:
            cfg = json.loads(self._CONFIG_PATH.read_text())
        except Exception:
            cfg = {}

        cfg["provider"] = provider
        api_key = self._apikey_edit.text().strip()
        _store_api_key(provider, api_key)
        # Remove any legacy plain-text key from JSON
        cfg.get("api_keys", {}).pop(provider, None)
        cfg.setdefault("endpoint_ids", {})["RunPod"] = self._endpoint_edit.text().strip()
        if hasattr(self, "_gguf_edit"):
            gguf = self._gguf_edit.text().strip()
            if gguf:
                cfg["gguf_path"] = gguf
        model = self._model_combo.currentText()
        if model:
            cfg.setdefault("last_models", {})[provider] = model

        try:
            self._CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_provider_changed(self, provider: str):
        """Switch client and show/hide credential fields."""
        # Save API key for the provider we're leaving
        old = self._current_provider
        if old and old != provider:
            old_key = self._apikey_edit.text().strip()
            _store_api_key(old, old_key)
            # Also persist to JSON config (model selection, etc.)
            try:
                cfg = json.loads(self._CONFIG_PATH.read_text())
            except Exception:
                cfg = {}
            cfg.get("api_keys", {}).pop(old, None)
            try:
                self._CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
            except Exception:
                pass

        self._current_provider = provider
        is_local    = (provider in ("Ollama", "Synapse Fine-tune"))
        is_runpod   = (provider == "RunPod")
        is_finetune = (provider == "Synapse Fine-tune")
        self._apikey_widget.setVisible(not is_local)
        self._apikey_edit.setPlaceholderText(
            "ollama signin → copy key" if provider == "Ollama Cloud"
            else "Paste your API key here"
        )
        self._gguf_widget.setVisible(is_finetune)
        self._endpoint_widget.setVisible(is_runpod)
        self._model_widget.setVisible(not is_runpod)

        # Load saved key / model for the new provider
        try:
            cfg = json.loads(self._CONFIG_PATH.read_text())
            json_key = cfg.get("api_keys", {}).get(provider, "")
            self._apikey_edit.setText(_retrieve_api_key(provider, json_key))
            self._config_model = cfg.get("last_models", {}).get(provider, "")
        except Exception:
            pass

        self._refresh_models()

    def _browse_gguf(self):
        """Open a file dialog to select a .gguf model file."""
        start = str(Path(self._gguf_edit.text()).parent) if self._gguf_edit.text() else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select GGUF model", start, "GGUF files (*.gguf);;All files (*)"
        )
        if path:
            self._gguf_edit.setText(path)
            self._save_config()
            self._refresh_models()

    def _refresh_models(self):
        """(Re-)create the client from current UI settings and fetch its model list."""
        provider = self._provider_combo.currentText()
        api_key  = self._apikey_edit.text().strip() if hasattr(self, "_apikey_edit") else ""

        # Build the appropriate client
        if provider == "OpenAI":
            self._client = OpenAIClient(api_key=api_key)
            default_model = OpenAIClient.DEFAULT_MODEL
            no_model_msg  = (
                "No models returned — check your OpenAI API key and click ⟳.\n"
                "Get a key at platform.openai.com/api-keys"
            )
        elif provider == "Groq":
            self._client = GroqClient(api_key=api_key)
            default_model = GroqClient.DEFAULT_MODEL
            no_model_msg  = (
                "No models returned — check your Groq API key and click ⟳.\n"
                "Get a free key at console.groq.com"
            )
        elif provider == "Gemini":
            self._client = GeminiClient(api_key=api_key)
            default_model = GeminiClient.DEFAULT_MODEL
            no_model_msg  = (
                "No models returned — check your Gemini API key and click ⟳.\n"
                "Get a free key at aistudio.google.com"
            )
        elif provider == "Claude":
            self._client = ClaudeClient(api_key=api_key)
            default_model = ClaudeClient.DEFAULT_MODEL
            no_model_msg  = (
                "No models returned — check your Anthropic API key and click ⟳.\n"
                "Get a key at console.anthropic.com"
            )
        elif provider == "RunPod":
            endpoint_id   = self._endpoint_edit.text().strip() if hasattr(self, "_endpoint_edit") else ""
            self._client  = RunPodClient(api_key=api_key, endpoint_id=endpoint_id)
            default_model = ""
            no_model_msg  = "Ready — enter your API key and Endpoint ID, then click Generate."
        elif provider == "Ollama Cloud":
            self._client  = OllamaClient(base_url=OllamaClient.CLOUD_BASE_URL, api_key=api_key)
            default_model = "gpt-oss:120b"
            no_model_msg  = (
                "No models returned — run 'ollama signin' then paste your key and click ⟳.\n"
                "Get an account at ollama.com"
            )
        elif provider == "Synapse Fine-tune":
            _GGUF_URL = "https://github.com/m00zu/Synapse-Plugins/releases/download/models-v0.1.0/synapse-qwen0_8B-v0.1.gguf"
            _GGUF_NAME = "synapse-qwen0_8B-v0.1.gguf"

            # Check if llama-cpp-python is installed
            try:
                import llama_cpp  # noqa: F401
            except ImportError:
                self._status_label.setText(
                    "llama-cpp-python is not installed.\n"
                    "Install: pip install llama-cpp-python\n"
                    "Or use 'Copy for Web AI' / API providers instead."
                )
                self._model_combo.clear()
                return

            # Find GGUF file
            _user_path = Path(self._gguf_edit.text().strip()) if hasattr(self, "_gguf_edit") and self._gguf_edit.text().strip() else None
            if _user_path and _user_path.exists():
                _gguf = _user_path
            else:
                _default = self._GGUF_DIR / _GGUF_NAME
                _q4 = self._GGUF_DIR / "synapse-qwen-0.8b.Q4_K_M.gguf"
                _q8 = self._GGUF_DIR / "synapse-qwen-0.8b.q8_0.gguf"
                if _default.exists():
                    _gguf = _default
                elif _q4.exists():
                    _gguf = _q4
                elif _q8.exists():
                    _gguf = _q8
                else:
                    # Prompt user to download — don't block the UI
                    self._status_label.setText(
                        "No GGUF model found. Click 'Download Model' or\n"
                        "set the path manually and click ⟳."
                    )
                    self._model_combo.clear()
                    # Add download button if not already present
                    if not hasattr(self, '_dl_btn'):
                        self._dl_btn = QtWidgets.QPushButton("Download Model (503 MB)")
                        self._dl_btn.clicked.connect(lambda: self._download_gguf(_GGUF_URL, _GGUF_NAME))
                        # Insert before the status label
                        idx = self.layout().indexOf(self._status_label)
                        if idx >= 0:
                            self.layout().insertWidget(idx, self._dl_btn)
                    self._dl_btn.setVisible(True)
                    return

            # Hide download button if model found
            if hasattr(self, '_dl_btn'):
                self._dl_btn.setVisible(False)

            self._status_label.setText(f"Loading model: {_gguf.name}…")
            QtWidgets.QApplication.processEvents()
            try:
                self._client = LlamaCppClient(str(_gguf), n_ctx=4096, n_gpu_layers=-1)
            except Exception as e:
                self._status_label.setText(f"Failed to load model: {e}")
                self._model_combo.clear()
                return
            default_model = _gguf.name
            no_model_msg  = (
                "GGUF model not found.\n"
                "Enter the path to your .gguf file and click ⟳."
            )
        else:  # Ollama (local)
            self._client = OllamaClient()
            default_model = OllamaClient.DEFAULT_MODEL
            no_model_msg  = (
                "Ollama not detected — start Ollama and click ⟳.\n"
                "Default model set; generation will fail until Ollama is running."
            )

        self._status_label.setText(f"Connecting to {provider}…")
        QtWidgets.QApplication.processEvents()

        # Prefer config-saved model, fall back to whatever was shown before
        restore_target = self._config_model or self._model_combo.currentText()
        self._config_model = ""  # consume — only used once per load/switch
        models = self._client.list_models()

        self._model_combo.blockSignals(True)
        self._model_combo.clear()

        if not models:
            self._model_combo.addItem(default_model)
            self._status_label.setText(no_model_msg)
        else:
            self._model_combo.addItems(models)
            restore_idx = self._model_combo.findText(restore_target)
            if restore_idx >= 0:
                self._model_combo.setCurrentIndex(restore_idx)
            else:
                default_idx = self._model_combo.findText(default_model)
                if default_idx >= 0:
                    self._model_combo.setCurrentIndex(default_idx)
            self._status_label.setText(f"Found {len(models)} model(s). Ready.")

        self._model_combo.blockSignals(False)
        self._on_model_changed(self._model_combo.currentText())

    def _on_model_changed(self, model_name: str):
        self._client.model = model_name
        self.graph._llm_client = self._client
        self._save_config()

    def _on_verbose_changed(self):
        """Rebuild the system prompt with or without full node docstrings."""
        verbose = self._verbose_check.isChecked()
        self._catalog = build_condensed_catalog(verbose=verbose)
        self._system  = build_system_prompt(self._catalog)
        size_kb = len(self._system.encode()) / 1024
        self._status_label.setText(
            f"{'Verbose' if verbose else 'Compact'} descriptions loaded "
            f"(prompt ~{size_kb:.1f} KB)."
        )

    def _download_gguf(self, url: str, filename: str):
        """Download GGUF model in a background thread."""
        self._dl_btn.setEnabled(False)
        self._dl_btn.setText("Downloading…")
        self._status_label.setText("Downloading model (503 MB)… This may take a few minutes.")

        class _DLWorker(QtCore.QObject):
            finished = QtCore.Signal(str)  # path on success, empty on error
            error = QtCore.Signal(str)

            def __init__(self, url, dest):
                super().__init__()
                self._url = url
                self._dest = dest

            def run(self):
                try:
                    import urllib.request
                    Path(self._dest).parent.mkdir(parents=True, exist_ok=True)
                    tmp = self._dest + '.part'
                    urllib.request.urlretrieve(self._url, tmp)
                    Path(tmp).rename(self._dest)
                    self.finished.emit(self._dest)
                except Exception as e:
                    self.error.emit(str(e))

        dest = str(self._GGUF_DIR / filename)
        self._dl_worker = _DLWorker(url, dest)
        self._dl_thread = QtCore.QThread()
        self._dl_worker.moveToThread(self._dl_thread)
        self._dl_thread.started.connect(self._dl_worker.run)
        self._dl_worker.finished.connect(self._on_dl_finished)
        self._dl_worker.error.connect(self._on_dl_error)
        self._dl_worker.finished.connect(self._dl_thread.quit)
        self._dl_worker.error.connect(self._dl_thread.quit)
        self._dl_thread.finished.connect(self._dl_worker.deleteLater)
        self._dl_thread.finished.connect(self._dl_thread.deleteLater)
        self._dl_thread.start()

    def _on_dl_finished(self, path: str):
        self._gguf_edit.setText(path)
        self._dl_btn.setVisible(False)
        self._status_label.setText("Model downloaded. Click ⟳ to load.")
        self._refresh_models()

    def _on_dl_error(self, msg: str):
        self._dl_btn.setEnabled(True)
        self._dl_btn.setText("Download Model (503 MB)")
        self._status_label.setText(f"Download failed: {msg}")

    def _on_copy_for_web(self):
        """Copy a ready-to-paste prompt to clipboard for use with web AI interfaces.

        For web UIs (ChatGPT, Claude.ai, Gemini) context windows are huge and
        there is no per-token API cost, so we build a verbose prompt with full
        docstrings for all nodes — giving the model the richest possible context.
        """
        question = self._question_edit.toPlainText().strip()
        if not question:
            self._status_label.setText("Please enter a question first.")
            return

        # Build verbose catalog with full docstrings (web UIs can handle it)
        verbose_catalog = build_condensed_catalog(verbose=True)
        system = build_system_prompt(verbose_catalog)

        # Build the full prompt
        use_context = (
            self._ctx_check.isChecked() and bool(self.graph.all_nodes())
        )
        prompt_parts = [system, ""]
        if use_context:
            current_wf = serialize_graph(self.graph)
            prompt_parts.append(
                "Here is the user's CURRENT workflow (JSON):\n"
                f"{json.dumps(current_wf, indent=2)}\n\n"
                "Return the COMPLETE updated workflow — include ALL existing nodes "
                "plus any new or modified nodes.\n"
            )
        prompt_parts.append(f"User request: {question}\n")
        prompt_parts.append(
            "Respond with ONLY a JSON object containing \"nodes\" and \"edges\". "
            "No markdown fences, no explanation."
        )

        full_prompt = "\n".join(prompt_parts)
        QtWidgets.QApplication.clipboard().setText(full_prompt)
        self._status_label.setText(
            f"Copied to clipboard ({len(full_prompt)//1024}KB). "
            "Paste into ChatGPT / Claude.ai / Gemini, "
            "then paste the JSON response below."
        )

    def _on_preview_changed(self):
        """Show Load/Replace buttons when the preview contains valid JSON."""
        text = self._preview_edit.toPlainText().strip()
        if not text:
            self._load_btn.setVisible(False)
            self._replace_btn.setVisible(False)
            self._last_workflow = None
            return
        try:
            wf = json.loads(text)
            if isinstance(wf, dict) and "nodes" in wf:
                self._last_workflow = wf
                self._load_btn.setVisible(True)
                self._replace_btn.setVisible(True)
            else:
                self._load_btn.setVisible(False)
                self._replace_btn.setVisible(False)
        except json.JSONDecodeError:
            self._load_btn.setVisible(False)
            self._replace_btn.setVisible(False)

    def _on_generate(self):
        question = self._question_edit.toPlainText().strip()
        if not question:
            self._status_label.setText("Please enter a question first.")
            return

        # Always sync credentials from UI fields (avoids stale values from before last Refresh)
        if hasattr(self._client, "api_key"):
            self._client.api_key = self._apikey_edit.text().strip()
        if hasattr(self._client, "endpoint_id"):
            self._client.endpoint_id = self._endpoint_edit.text().strip()

        # Persist current settings (captures API key even if user didn't click Refresh)
        self._save_config()

        # Build the user message — optionally prepend the current canvas
        use_context = (
            self._ctx_check.isChecked() and bool(self.graph.all_nodes())
        )
        if use_context:
            current_wf = serialize_graph(self.graph)
            user_msg = (
                "Here is the user's CURRENT workflow (JSON):\n"
                f"{json.dumps(current_wf, indent=2)}\n\n"
                f"User request: {question}\n\n"
                "Return the COMPLETE updated workflow — include ALL existing nodes "
                "(preserving their types and custom properties) plus any new or "
                "modified nodes. Re-number all node IDs sequentially from n1."
            )
        else:
            user_msg = question

        # Reset state
        self._last_workflow = None
        self._load_btn.setVisible(False)
        self._replace_btn.setVisible(False)
        self._preview_edit.setPlainText("")
        self._generate_btn.setEnabled(False)
        self._status_label.setText("Generating… (this may take up to 60 s)")

        # Use short system prompt for fine-tuned model (catalog baked into weights)
        provider = self._provider_combo.currentText()

        if provider == "Synapse Fine-tune":
            # Single-pass: knowledge baked into fine-tuned weights
            self._worker = LLMWorker(self._client, _FINETUNE_SYS_PROMPT, user_msg)
        else:
            # Two-pass: select relevant nodes → generate with enriched detail
            self._status_label.setText(
                "Pass 1/2 — selecting relevant nodes…"
            )
            self._worker = TwoPassLLMWorker(
                self._client, self._catalog, user_msg
            )
            self._worker.progress.connect(self._status_label.setText)

        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.result.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_result(self, json_str: str):
        """Called from main thread via Qt signal when the worker succeeds."""
        self._generate_btn.setEnabled(True)

        # Attempt to parse so we can pretty-print and validate
        try:
            workflow = json.loads(json_str)
            pretty   = json.dumps(workflow, indent=2)
            self._last_workflow = workflow
            self._preview_edit.setPlainText(pretty)
            n_nodes = len(workflow.get("nodes", []))
            n_edges = len(workflow.get("edges", []))
            self._status_label.setText(
                f"Done — {n_nodes} node(s), {n_edges} edge(s). "
                f"Review the JSON above, then load it into the canvas."
            )
            self._load_btn.setVisible(True)
            self._replace_btn.setVisible(True)
        except json.JSONDecodeError as exc:
            self._preview_edit.setPlainText(json_str)
            self._status_label.setText(
                f"Model returned invalid JSON ({exc}).\n"
                "Try regenerating, or use a more capable model."
            )

    def _on_error(self, message: str):
        """Called from main thread via Qt signal when the worker fails."""
        self._generate_btn.setEnabled(True)
        self._preview_edit.setPlainText("")
        self._status_label.setText(f"Error: {message}")

    def _on_load(self):
        """Load the last generated workflow into the canvas."""
        if self._last_workflow is None:
            return

        loader = WorkflowLoader(self.graph)

        # Place new nodes to the right / below existing nodes
        all_nodes = self.graph.all_nodes()
        if all_nodes:
            max_x = max(n.pos()[0] for n in all_nodes) + 300
            min_y = min(n.pos()[1] for n in all_nodes)
            origin = (int(max_x), int(min_y))
        else:
            origin = (100, 100)

        success, message = loader.build(self._last_workflow, *origin)

        self.graph.fit_to_selection()

        if not success:
            QtWidgets.QMessageBox.critical(self, "Load Error", message)
        elif "warning" in message.lower():
            # Warnings mean some edges failed — show them prominently so the
            # user can see exactly which port names caused the mismatch.
            self._status_label.setText(message)
            QtWidgets.QMessageBox.warning(self, "Workflow Loaded with Warnings", message)
        else:
            self._status_label.setText(message)

    def _on_replace(self):
        """Clear the entire canvas, then load the generated workflow from scratch."""
        if self._last_workflow is None:
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Replace Canvas",
            "This will delete ALL existing nodes and load the generated workflow.\n"
            "This action cannot be undone. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # Delete all existing nodes
        for node in list(self.graph.all_nodes()):
            self.graph.delete_node(node, push_undo=False)

        loader = WorkflowLoader(self.graph)
        success, message = loader.build(self._last_workflow, 100, 100)
        # fit_to_selection() frames all nodes when nothing is selected,
        # which is exactly the state after a replace (new nodes are unselected).
        self.graph.fit_to_selection()

        if not success:
            QtWidgets.QMessageBox.critical(self, "Load Error", message)
        elif "warning" in message.lower():
            self._status_label.setText(message)
            QtWidgets.QMessageBox.warning(self, "Workflow Loaded with Warnings", message)
        else:
            self._status_label.setText(message)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _make_separator() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    return line


# ---------------------------------------------------------------------------
# AIChatPanel — conversational workflow refinement
# ---------------------------------------------------------------------------

class AIChatPanel(QtWidgets.QWidget):
    """
    Chat-based dock panel for iterative workflow building.
    Reuses the LLM provider/key from the AI Assistant panel config.
    Each turn automatically includes the current canvas as context.
    """

    _CONFIG_PATH = Path.home() / ".synapse_llm_config.json"

    _PROVIDERS = ("Ollama", "Ollama Cloud", "OpenAI", "Claude", "Groq", "Gemini")

    def __init__(self, graph, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.graph = graph
        self._client = None
        self._messages: list[dict] = []  # conversation history
        self._catalog = build_condensed_catalog()
        self._system = self._build_chat_system_prompt()
        self._dark = True  # default; updated by theme signal

        self._build_ui()
        self._load_config()

    # ------------------------------------------------------------------
    def _build_chat_system_prompt(self) -> str:
        """System prompt for conversational workflow editing.
        Reuses the full system prompt (with examples and rules) and adds
        conversation-specific instructions.
        """
        base = build_system_prompt(self._catalog)
        chat_addendum = (
            "\n\nADDITIONAL CONVERSATION RULES:\n"
            "- You are in a multi-turn CONVERSATION. The user may ask to create, "
            "modify, or ask questions about workflows.\n"
            "- When the user asks to CREATE or MODIFY a workflow, respond with ONLY "
            "a JSON object: {\"nodes\": [...], \"edges\": [...]}. No explanation, "
            "no markdown fences.\n"
            "- Return the COMPLETE workflow every time (all nodes + all edges), "
            "not just the changed parts.\n"
            "- When the user asks a QUESTION (not requesting a workflow change), "
            "respond with a helpful text answer.\n"
        )
        return base + chat_addendum

    # ------------------------------------------------------------------
    def _build_style(self) -> str:
        """Generate a theme-aware stylesheet."""
        dark = self._dark
        if dark:
            bg = "#0d1117"; bg2 = "#161b22"; fg = "#c9d1d9"
            border = "#30363d"; btn_bg = "#21262d"; btn_hover = "#30363d"
            dis_fg = "#484f58"; dis_border = "#21262d"; sep = "#21262d"
        else:
            bg = "#ffffff"; bg2 = "#f6f8fa"; fg = "#24292f"
            border = "#d0d7de"; btn_bg = "#f3f4f6"; btn_hover = "#e1e4e8"
            dis_fg = "#8c959f"; dis_border = "#d0d7de"; sep = "#d0d7de"

        return f"""
        AIChatPanel {{ background: {bg}; }}
        QLabel {{ color: {fg}; font-family: sans-serif; font-size: 12px; }}
        QComboBox {{
            background: {bg2}; color: {fg}; border: 1px solid {border};
            border-radius: 6px; padding: 4px 8px; font-size: 12px; min-height: 22px;
        }}
        QComboBox::drop-down {{ border: none; width: 20px; }}
        QComboBox QAbstractItemView {{
            background: {bg2}; color: {fg}; border: 1px solid {border};
            selection-background-color: #1f6feb;
        }}
        QLineEdit {{
            background: {bg}; color: {fg}; border: 1px solid {border};
            border-radius: 6px; padding: 5px 8px; font-size: 12px;
        }}
        QLineEdit:focus {{ border-color: #1f6feb; }}
        QPushButton {{
            background: {btn_bg}; color: {fg}; border: 1px solid {border};
            border-radius: 6px; padding: 6px 14px;
            font-family: sans-serif;
            font-size: 12px; font-weight: 500;
        }}
        QPushButton:hover {{ background: {btn_hover}; border-color: #8b949e; }}
        QPushButton:pressed {{ background: {bg2}; }}
        QPushButton:disabled {{ color: {dis_fg}; border-color: {dis_border}; }}
        QPushButton#sendBtn {{ background: #238636; color: #fff; border: 1px solid #2ea043; font-weight: 600; }}
        QPushButton#sendBtn:hover {{ background: #2ea043; }}
        QPushButton#loadBtn {{ background: #1f6feb; color: #fff; border: 1px solid #388bfd; }}
        QPushButton#loadBtn:hover {{ background: #388bfd; }}
        QPushButton#loadBtn:disabled {{ background: {btn_bg}; color: {dis_fg}; border-color: {dis_border}; }}
        QPushButton#replaceBtn {{ background: #da3633; color: #fff; border: 1px solid #f85149; }}
        QPushButton#replaceBtn:hover {{ background: #f85149; }}
        QPushButton#replaceBtn:disabled {{ background: {btn_bg}; color: {dis_fg}; border-color: {dis_border}; }}
        QPushButton#refreshBtn {{ padding: 4px 6px; font-size: 14px; }}
        QTextBrowser {{
            background: {bg}; color: {fg}; border: none;
            font-family: sans-serif; font-size: 13px;
        }}
        QPlainTextEdit {{
            background: {bg2}; color: {fg}; border: 1px solid {border};
            border-radius: 8px; padding: 8px;
            font-family: sans-serif; font-size: 13px;
        }}
        QPlainTextEdit:focus {{ border-color: #1f6feb; }}
        QFrame[frameShape="4"] {{ color: {sep}; max-height: 1px; }}
        """

    def _apply_theme(self, is_dark=None):
        """Apply the current theme stylesheet and store colors for bubbles."""
        if is_dark is not None:
            self._dark = is_dark
        dark = self._dark
        self._bubble_colors = {
            "user_bg":   "#1f6feb",
            "user_fg":   "#ffffff",
            "ai_bg":     "#161b22" if dark else "#f1f3f5",
            "ai_fg":     "#c9d1d9" if dark else "#24292f",
            "ai_label":  "#58a6ff" if dark else "#0969da",
            "ai_border": "#30363d" if dark else "#d0d7de",
            "err_bg":    "#3d1214" if dark else "#ffeef0",
            "err_fg":    "#f85149",
            "sys_fg":    "#8b949e" if dark else "#57606a",
        }
        self.setStyleSheet(self._build_style())

    def _build_ui(self):
        self._apply_theme()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # --- Settings bar (collapsible) --------------------------------
        settings_frame = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_frame)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(4)

        # Provider row
        prov_row = QtWidgets.QHBoxLayout()
        prov_row.addWidget(QtWidgets.QLabel("Provider"))
        self._provider_combo = QtWidgets.QComboBox()
        self._provider_combo.addItems(self._PROVIDERS)
        self._provider_combo.currentTextChanged.connect(self._on_provider_changed)
        prov_row.addWidget(self._provider_combo, stretch=1)
        settings_layout.addLayout(prov_row)

        # API key row (hidden for Ollama)
        self._apikey_widget = QtWidgets.QWidget()
        key_row = QtWidgets.QHBoxLayout(self._apikey_widget)
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.addWidget(QtWidgets.QLabel("Key"))
        self._apikey_edit = QtWidgets.QLineEdit()
        self._apikey_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._apikey_edit.setPlaceholderText("API key")
        key_row.addWidget(self._apikey_edit)
        settings_layout.addWidget(self._apikey_widget)
        self._apikey_widget.hide()

        # Model row
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel("Model"))
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setMinimumWidth(120)
        model_row.addWidget(self._model_combo, stretch=1)
        refresh_btn = QtWidgets.QPushButton("⟳")
        refresh_btn.setObjectName("refreshBtn")
        refresh_btn.setFixedWidth(32)
        refresh_btn.setToolTip("Refresh model list")
        refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(refresh_btn)
        settings_layout.addLayout(model_row)

        layout.addWidget(settings_frame)
        layout.addWidget(_make_separator())

        # --- Chat history display -------------------------------------
        self._chat_display = QtWidgets.QTextBrowser()
        self._chat_display.setOpenExternalLinks(False)
        self._chat_display.setReadOnly(True)
        layout.addWidget(self._chat_display, stretch=1)

        # --- Input area -----------------------------------------------
        self._input_edit = QtWidgets.QPlainTextEdit()
        self._input_edit.setPlaceholderText("Ask me to build or modify a workflow…")
        self._input_edit.setMaximumHeight(64)
        layout.addWidget(self._input_edit)

        # --- Action buttons -------------------------------------------
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)

        self._send_btn = QtWidgets.QPushButton("Send")
        self._send_btn.setObjectName("sendBtn")
        self._send_btn.clicked.connect(self._on_send)
        btn_row.addWidget(self._send_btn)

        self._load_btn = QtWidgets.QPushButton("Load")
        self._load_btn.setObjectName("loadBtn")
        self._load_btn.setToolTip("Load the last generated workflow into the canvas")
        self._load_btn.clicked.connect(self._on_load)
        self._load_btn.setEnabled(False)
        btn_row.addWidget(self._load_btn)

        self._replace_btn = QtWidgets.QPushButton("Replace")
        self._replace_btn.setObjectName("replaceBtn")
        self._replace_btn.setToolTip("Clear canvas and load the last generated workflow")
        self._replace_btn.clicked.connect(self._on_replace)
        self._replace_btn.setEnabled(False)
        btn_row.addWidget(self._replace_btn)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

        # --- Status ---------------------------------------------------
        self._status = QtWidgets.QLabel("Ready")
        self._status.setStyleSheet("color: #484f58; font-size: 11px;")
        layout.addWidget(self._status)

        self._last_workflow: Optional[dict] = None

    # ------------------------------------------------------------------
    def _load_config(self):
        """Restore provider/model/key from the shared AI Assistant config."""
        try:
            cfg = json.loads(self._CONFIG_PATH.read_text())
        except Exception:
            self._on_provider_changed(self._provider_combo.currentText())
            return

        provider = cfg.get("provider", "Ollama")
        idx = self._provider_combo.findText(provider)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)

        json_key = cfg.get("api_keys", {}).get(provider, "")
        self._apikey_edit.setText(_retrieve_api_key(provider, json_key))

        self._on_provider_changed(provider)

        # Restore model after refresh
        saved_model = cfg.get("last_models", {}).get(provider, "")
        if saved_model:
            idx = self._model_combo.findText(saved_model)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
            else:
                self._model_combo.setCurrentText(saved_model)

        self._rebuild_client()

    # ------------------------------------------------------------------
    def _on_provider_changed(self, provider: str):
        """Show/hide API key field and refresh models."""
        is_local = provider == "Ollama"
        self._apikey_widget.setVisible(not is_local)
        self._refresh_models()

    # ------------------------------------------------------------------
    def _refresh_models(self):
        """Populate the model dropdown from the selected provider."""
        provider = self._provider_combo.currentText()
        api_key = self._apikey_edit.text().strip()

        # Create a temporary client just to list models
        if provider == "Ollama":
            tmp = OllamaClient()
        elif provider == "Ollama Cloud":
            tmp = OllamaClient(base_url=OllamaClient.CLOUD_BASE_URL, api_key=api_key)
        elif provider == "OpenAI":
            tmp = OpenAIClient(api_key=api_key)
        elif provider == "Claude":
            tmp = ClaudeClient(api_key=api_key)
        elif provider == "Groq":
            tmp = GroqClient(api_key=api_key)
        elif provider == "Gemini":
            tmp = GeminiClient(api_key=api_key)
        else:
            return

        prev = self._model_combo.currentText()
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        models = tmp.list_models()
        if models:
            self._model_combo.addItems(models)
            idx = self._model_combo.findText(prev)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)

        self._rebuild_client()

    # ------------------------------------------------------------------
    def _rebuild_client(self):
        """Create the LLM client from current UI selections."""
        provider = self._provider_combo.currentText()
        model = self._model_combo.currentText()
        api_key = self._apikey_edit.text().strip()

        if provider == "Ollama":
            self._client = OllamaClient(model=model or OllamaClient.DEFAULT_MODEL)
        elif provider == "Ollama Cloud":
            self._client = OllamaClient(
                base_url=OllamaClient.CLOUD_BASE_URL,
                model=model or OllamaClient.DEFAULT_MODEL, api_key=api_key)
        elif provider == "OpenAI":
            self._client = OpenAIClient(api_key=api_key, model=model or OpenAIClient.DEFAULT_MODEL)
        elif provider == "Claude":
            self._client = ClaudeClient(api_key=api_key, model=model or ClaudeClient.DEFAULT_MODEL)
        elif provider == "Groq":
            self._client = GroqClient(api_key=api_key, model=model or GroqClient.DEFAULT_MODEL)
        elif provider == "Gemini":
            self._client = GeminiClient(api_key=api_key, model=model or GeminiClient.DEFAULT_MODEL)
        else:
            self._client = None

        if self._client:
            self.graph._llm_client = self._client
            self._status.setText(f"{provider} / {self._client.model}")

    # ------------------------------------------------------------------
    def _on_send(self):
        text = self._input_edit.toPlainText().strip()
        if not text:
            return
        if self._client is None:
            self._rebuild_client()
            if self._client is None:
                return

        # Store clean user text in conversation history
        self._messages.append({"role": "user", "content": text})
        self._append_bubble("user", text)
        self._input_edit.clear()

        # Build messages for LLM: inject fresh canvas context before the
        # conversation history so the LLM always sees the current state
        canvas_ctx = ""
        if self.graph.all_nodes():
            canvas_json = serialize_graph(self.graph)
            canvas_ctx = (
                f"Current workflow on canvas:\n"
                f"{json.dumps(canvas_json, indent=2)}\n\n"
                "When modifying, return the COMPLETE updated workflow as JSON. "
                "When answering a question, respond with text."
            )

        # Prepend canvas context as a system-injected user message at the start
        llm_messages = []
        if canvas_ctx:
            llm_messages.append({"role": "user", "content": canvas_ctx})
            llm_messages.append({"role": "assistant", "content": "Understood. I can see your current workflow. What would you like to change?"})
        llm_messages.extend(self._messages)

        # Disable send while processing
        self._send_btn.setEnabled(False)
        self._send_btn.setText("…")
        self._status.setText("Thinking…")

        # Background thread for LLM call
        self._worker = _ChatWorker(self._client, self._system, llm_messages)
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.result.connect(self._on_response)
        self._worker.error.connect(self._on_error)
        self._worker.result.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    # ------------------------------------------------------------------
    def _on_response(self, response: str):
        self._send_btn.setEnabled(True)
        self._send_btn.setText("Send")
        self._messages.append({"role": "assistant", "content": response})

        # Check if response is a workflow JSON
        try:
            workflow = json.loads(response)
            if "nodes" in workflow and "edges" in workflow:
                self._last_workflow = workflow
                n = len(workflow["nodes"])
                e = len(workflow["edges"])
                # print(f"[AI Chat] Workflow: {json.dumps(workflow, indent=2)[:2000]}")
                self._append_bubble("assistant",
                    f"Workflow generated: {n} node(s), {e} edge(s).\n"
                    f"Click 'Load into Canvas' or 'Replace Canvas'.")
                self._load_btn.setEnabled(True)
                self._replace_btn.setEnabled(True)
                self._status.setText(f"Workflow ready — {n} nodes")
                return
        except (json.JSONDecodeError, TypeError):
            pass

        # Plain text response
        self._append_bubble("assistant", response)
        self._status.setText("Ready")

    # ------------------------------------------------------------------
    def _on_error(self, msg: str):
        self._send_btn.setEnabled(True)
        self._send_btn.setText("Send")
        self._append_bubble("error", msg)
        self._status.setText("Error")
        # Remove the last user message that failed
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()

    # ------------------------------------------------------------------
    def _on_load(self):
        if not self._last_workflow:
            return
        loader = WorkflowLoader(self.graph)
        all_nodes = self.graph.all_nodes()
        if all_nodes:
            max_x = max(n.pos()[0] for n in all_nodes) + 300
            min_y = min(n.pos()[1] for n in all_nodes)
        else:
            max_x, min_y = 100, 100
        _ok, msg = loader.build(self._last_workflow, origin_x=int(max_x), origin_y=int(min_y))
        self._append_bubble("system", msg)

    def _on_replace(self):
        if not self._last_workflow:
            return
        for node in list(self.graph.all_nodes()):
            self.graph.remove_node(node)
        loader = WorkflowLoader(self.graph)
        _ok, msg = loader.build(self._last_workflow)
        self._append_bubble("system", msg)

    def _on_clear(self):
        self._messages.clear()
        self._chat_display.clear()
        self._last_workflow = None
        self._load_btn.setEnabled(False)
        self._replace_btn.setEnabled(False)
        self._status.setText("Chat cleared")

    # ------------------------------------------------------------------
    def _append_bubble(self, role: str, text: str):
        """Append a chat-style message bubble with tail to the display."""
        c = self._bubble_colors
        text_escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text_html = text_escaped.replace("\n", "<br>")

        # ◢ = bottom-right tail (U+25E2), ◣ = bottom-left tail (U+25E3)
        # Add spacing before each message
        self._chat_display.append("<div style='margin:0; padding:0; line-height:6px;'>&nbsp;</div>")

        if role == "user":
            html = (
                f"<table width='100%' cellpadding='0' cellspacing='0'>"
                f"<tr><td width='15%'></td>"
                f"<td style='background:{c['user_bg']}; color:{c['user_fg']}; "
                f"padding:8px 14px; border-radius:14px 14px 4px 14px;'>"
                f"<span style='font-size:13px; line-height:1.5;'>"
                f"{text_html}</span></td></tr>"
                f"<tr><td></td>"
                f"<td align='right' style='padding:0; line-height:0; font-size:0;'>"
                f"<span style='color:{c['user_bg']}; font-size:14px; "
                f"line-height:0;'>&#9698;</span></td></tr>"
                f"</table>"
            )
        elif role == "assistant":
            html = (
                f"<table width='100%' cellpadding='0' cellspacing='0'>"
                f"<tr><td style='background:{c['ai_bg']}; color:{c['ai_fg']}; "
                f"padding:8px 14px; border-radius:4px 14px 14px 14px; "
                f"border:1px solid {c['ai_border']};'>"
                f"<span style='color:{c['ai_label']}; font-size:10px; "
                f"font-weight:600;'>AI</span><br>"
                f"<span style='font-size:13px; line-height:1.5;'>"
                f"{text_html}</span></td>"
                f"<td width='15%'></td></tr>"
                f"<tr><td style='padding:0; line-height:0; font-size:0;'>"
                f"<span style='color:{c['ai_bg']}; font-size:14px; "
                f"line-height:0;'>&#9699;</span></td>"
                f"<td></td></tr>"
                f"</table>"
            )
        elif role == "error":
            html = (
                f"<table width='100%' cellpadding='0' cellspacing='0'>"
                f"<tr><td style='background:{c['err_bg']}; color:{c['err_fg']}; "
                f"padding:8px 14px; border-radius:12px;'>"
                f"<span style='font-size:10px; font-weight:600;'>Error</span><br>"
                f"<span style='font-size:13px;'>{text_html}</span></td>"
                f"<td width='15%'></td></tr>"
                f"<tr><td style='padding:0; line-height:0; font-size:0;'>"
                f"<span style='color:{c['err_bg']}; font-size:14px; "
                f"line-height:0;'>&#9699;</span></td>"
                f"<td></td></tr>"
                f"</table>"
            )
        else:  # system
            html = (
                f"<table width='100%' cellpadding='0' cellspacing='0'><tr>"
                f"<td width='10%'></td>"
                f"<td align='center' style='color:{c['sys_fg']}; "
                f"font-size:11px; font-style:italic; padding:4px 0;'>"
                f"{text_html}</td>"
                f"<td width='10%'></td></tr></table>"
            )

        self._chat_display.append(html)
        sb = self._chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())


class _ChatWorker(QtCore.QObject):
    """Background worker for multi-turn LLM calls."""
    result = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, client, system: str, messages: list[dict]):
        super().__init__()
        self._client = client
        self._system = system
        self._messages = messages

    def run(self):
        try:
            if hasattr(self._client, 'chat_multi'):
                text = self._client.chat_multi(self._system, self._messages)
            else:
                # Fallback: concatenate history into single user message
                combined = "\n\n".join(
                    f"[{m['role']}]: {m['content']}" for m in self._messages
                )
                text = self._client.chat(self._system, combined)
            self.result.emit(text)
        except Exception as exc:
            self.error.emit(str(exc))
