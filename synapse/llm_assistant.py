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
    candidates = [
        Path(__file__).parent / "llm_node_schema.json",   # pip install / source
        Path(__file__).parent.parent / "llm_node_schema.json",  # source root
    ]
    if "__compiled__" in globals():
        import sys
        candidates.append(Path(sys.executable).parent / "llm_node_schema.json")  # Nuitka
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # fallback (will error on open)

_SCHEMA_PATH = _find_schema()

# ---------------------------------------------------------------------------
# Response schema — sent as Ollama's `format` parameter.
# Stripped of allOf rules; only describes the required output shape.
# ---------------------------------------------------------------------------
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":     {"type": "string"},
                    "type":   {"type": "string"},
                    "custom": {"type": "object"},
                },
                "required": ["id", "type"],
            },
        },
        "edges": {
            "type": "array",
            "description": "Connections between node ports. Must not be empty when there are 2 or more nodes.",
            "items": {
                "type": "object",
                "properties": {
                    "from_node_id": {"type": "string"},
                    "from_port":    {"type": "string"},
                    "to_node_id":   {"type": "string"},
                    "to_port":      {"type": "string"},
                },
                "required": ["from_node_id", "from_port", "to_node_id", "to_port"],
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
    Imports the nodes package and returns a mapping of ClassName → cleaned docstring.
    Falls back to an empty dict if the import fails (e.g., missing Qt display).
    """
    try:
        from . import nodes as _nodes_mod
        from .nodes import __all__ as _node_all
        docs: dict[str, str] = {}
        for name in _node_all:
            cls = getattr(_nodes_mod, name, None)
            if cls is None or not isinstance(cls, type):
                continue
            raw = getattr(cls, "__doc__", None) or ""
            # Collapse indentation and blank lines → single clean paragraph
            cleaned = " ".join(
                line.strip() for line in raw.splitlines() if line.strip()
            )
            if cleaned:
                docs[name] = cleaned
        return docs
    except Exception:
        return {}


def build_condensed_catalog(
    schema_path: Path = _SCHEMA_PATH,
    verbose: bool = False,
) -> str:
    """
    Reads llm_node_schema.json and returns a compact one-line-per-node catalog
    suitable for embedding in an LLM system prompt.

    Args:
        verbose: When True, replaces the single-line schema description with the
                 full class docstring loaded directly from the Python source.
                 Produces a larger (~6 KB) but more informative catalog.
    """
    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)

    class_docs: dict[str, str] = _load_class_docs() if verbose else {}

    def _fmt_ports(ports) -> str:
        """Format a port list into 'name<type>' or 'name<type; cols:a,b,c>' strings."""
        parts = []
        for p in ports:
            if isinstance(p, dict):
                cols = p.get("columns")
                if cols:
                    parts.append(f"{p['name']}<{p['type']}; cols:{','.join(cols)}>")
                else:
                    parts.append(f"{p['name']}<{p['type']}>")
            else:
                parts.append(str(p))
        return ", ".join(parts) or "—"

    catalog = schema.get("node_catalog", {})
    lines: list[str] = []
    for name, info in catalog.items():
        if verbose and name in class_docs:
            desc = class_docs[name]
        else:
            desc = info.get("description", "").strip()

        inputs  = _fmt_ports(info.get("inputs",  []))
        outputs = _fmt_ports(info.get("outputs", []))
        cfg     = info.get("configurable_properties", {})

        prop_parts = []
        for prop_name, prop_info in cfg.items():
            options     = prop_info.get("options")
            default     = prop_info.get("default", "")
            desc_note   = prop_info.get("description", "")
            default_str = f'"{default}"' if isinstance(default, str) else str(default)
            suffix      = f" /* {desc_note} */" if desc_note else ""
            if options:
                prop_parts.append(f"{prop_name}=[{' | '.join(options)}](default:{default_str}){suffix}")
            else:
                prop_parts.append(f"{prop_name}(default:{default_str}){suffix}")
        prop_str = f" | props: {{{', '.join(prop_parts)}}}" if prop_parts else ""

        lines.append(f"- {name}: {desc} | in:[{inputs}] → out:[{outputs}]{prop_str}")

    # Append plugin nodes (auto-discovered at runtime via plugin_loader)
    try:
        from .plugin_loader import get_plugin_catalog_entries
        for entry in get_plugin_catalog_entries():
            ins  = ', '.join(p['name'] for p in entry['inputs'])
            outs = ', '.join(p['name'] for p in entry['outputs'])
            lines.append(
                f"- {entry['class_name']}: {entry['description']} | in:[{ins}] → out:[{outs}]"
            )
    except ImportError:
        pass

    return "\n".join(lines)


def build_system_prompt(catalog_text: str) -> str:
    # Few-shot example 1: CSV → outlier removal → swarm plot → figure cell
    example_csv = json.dumps({
        "nodes": [
            {"id": "n1", "type": "FileReadNode",         "custom": {"separator": ","}},
            {"id": "n2", "type": "OutlierDetectionNode", "custom": {"method": "Grubbs", "threshold": "0.05"}},
            {"id": "n3", "type": "SwarmPlotNode",        "custom": {}},
            {"id": "n4", "type": "DataFigureCellNode",   "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",  "to_node_id": "n2", "to_port": "in"},
            {"from_node_id": "n2", "from_port": "kept", "to_node_id": "n3", "to_port": "data"},
            {"from_node_id": "n3", "from_port": "plot", "to_node_id": "n4", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 2: single image file → split RGB → CLAHE on red channel → image cell
    example_image = json.dumps({
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",        "custom": {}},
            {"id": "n2", "type": "SplitRGBNode",         "custom": {}},
            {"id": "n3", "type": "EqualizeAdapthistNode","custom": {}},
            {"id": "n4", "type": "ImageCellNode",        "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",   "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "red",   "to_node_id": "n3", "to_port": "image"},
            {"from_node_id": "n3", "from_port": "image", "to_node_id": "n4", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 3: <any> port — typed output → <any> input is always valid
    example_any = json.dumps({
        "nodes": [
            {"id": "n1", "type": "FileReadNode",       "custom": {}},
            {"id": "n2", "type": "DataSummaryNode",    "custom": {}},
            {"id": "n3", "type": "DataFigureCellNode", "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out", "to_node_id": "n2", "to_port": "in"},
            {"from_node_id": "n2", "from_port": "fig", "to_node_id": "n3", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 4: mask pipeline — image → threshold → KeepMaxIntensity → ImageMath
    # Shows: (a) you MUST threshold an image before feeding it to a <mask> port,
    #         (b) KeepMaxIntensityRegionNode needs BOTH mask + intensity_image inputs,
    #         (c) ImageMathNode applies the resulting mask to another image.
    example_mask = json.dumps({
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",               "custom": {}},
            {"id": "n2", "type": "SplitRGBNode",                "custom": {}},
            {"id": "n3", "type": "EqualizeAdapthistNode",        "custom": {}},
            {"id": "n4", "type": "BinaryThresholdNode",          "custom": {}},
            {"id": "n5", "type": "KeepMaxIntensityRegionNode",   "custom": {}},
            {"id": "n6", "type": "ImageMathNode",               "custom": {"operation": "A × B (apply mask)"}},
            {"id": "n7", "type": "ImageCellNode",               "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",              "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "red",              "to_node_id": "n3", "to_port": "image"},
            {"from_node_id": "n3", "from_port": "image",            "to_node_id": "n4", "to_port": "image"},
            {"from_node_id": "n4", "from_port": "mask",             "to_node_id": "n5", "to_port": "mask"},
            {"from_node_id": "n3", "from_port": "image",            "to_node_id": "n5", "to_port": "intensity_image"},
            {"from_node_id": "n5", "from_port": "mask",             "to_node_id": "n6", "to_port": "B (mask)"},
            {"from_node_id": "n1", "from_port": "out",              "to_node_id": "n6", "to_port": "A (image/mask)"},
            {"from_node_id": "n6", "from_port": "image",            "to_node_id": "n7", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 5: long-format table → ViolinPlotNode
    # Demonstrates: plot node input port is called 'data' (NOT 'in'), and
    # 'x_col' / 'y_col' must match the actual column names in the table.
    example_violin = json.dumps({
        "nodes": [
            {"id": "n1", "type": "FileReadNode",        "custom": {"separator": ","}},
            {"id": "n2", "type": "OutlierDetectionNode", "custom": {}},
            {"id": "n3", "type": "ViolinPlotNode",       "custom": {"x_col": "Group", "y_col": "Value", "order": "Control,Treatment"}},
            {"id": "n4", "type": "DataFigureCellNode",   "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",  "to_node_id": "n2", "to_port": "in"},
            {"from_node_id": "n2", "from_port": "kept", "to_node_id": "n3", "to_port": "data"},
            {"from_node_id": "n3", "from_port": "plot", "to_node_id": "n4", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 6: full stats pipeline → BarPlotNode with significance brackets
    # Demonstrates:
    #   (a) 'kept' fans out to BOTH PairwiseComparisonNode AND the plot node
    #   (b) PairwiseComparisonNode.stats_table<stat> → BarPlotNode.stats<stat>
    #   (c) plot node data input is always 'data'; stats input is always 'stats'
    example_bar_stats = json.dumps({
        "nodes": [
            {"id": "n1", "type": "FileReadNode",           "custom": {"separator": ","}},
            {"id": "n2", "type": "OutlierDetectionNode",   "custom": {}},
            {"id": "n3", "type": "PairwiseComparisonNode", "custom": {}},
            {"id": "n4", "type": "BarPlotNode",            "custom": {"x_col": "Group", "y_col": "Value"}},
            {"id": "n5", "type": "DataFigureCellNode",     "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",         "to_node_id": "n2", "to_port": "in"},
            {"from_node_id": "n2", "from_port": "kept",        "to_node_id": "n3", "to_port": "in"},
            {"from_node_id": "n2", "from_port": "kept",        "to_node_id": "n4", "to_port": "data"},
            {"from_node_id": "n3", "from_port": "stats_table", "to_node_id": "n4", "to_port": "stats"},
            {"from_node_id": "n4", "from_port": "plot",        "to_node_id": "n5", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 7: column-aware plotting
    # Shows: when a catalog entry lists `cols:col1,col2,...` on a table output,
    # use those exact names as property values in downstream plot/filter nodes.
    example_cols = json.dumps({  # noqa: keep var name
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",       "custom": {}},
            {"id": "n2", "type": "BinaryThresholdNode",  "custom": {}},
            {"id": "n3", "type": "ParticlePropsNode",    "custom": {}},
            {"id": "n4", "type": "HistogramNode",        "custom": {"value_col": "area"}},
            {"id": "n5", "type": "DataFigureCellNode",   "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",   "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "mask",  "to_node_id": "n3", "to_port": "mask"},
            {"from_node_id": "n3", "from_port": "table", "to_node_id": "n4", "to_port": "data"},
            {"from_node_id": "n4", "from_port": "plot",  "to_node_id": "n5", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 9: nucleus segmentation
    # Source: NEUBIAS training resource / scikit-image "Segment nuclei" tutorial
    # Shows: full preprocessing chain before watershed
    #   RollingBall (background) → GaussianBlur (denoise) → threshold → fill holes
    #   → remove debris → watershed → props table
    # NOTE: GaussianBlurNode ports are 'in'/'out', NOT 'image'/'image'
    example_nuclei = json.dumps({
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",          "custom": {}},
            {"id": "n2", "type": "RollingBallNode",        "custom": {"radius": 50}},
            {"id": "n3", "type": "GaussianBlurNode",       "custom": {"sigma": 2.0}},
            {"id": "n4", "type": "BinaryThresholdNode",    "custom": {}},
            {"id": "n5", "type": "FillHolesNode",          "custom": {}},
            {"id": "n6", "type": "RemoveSmallObjectsNode", "custom": {"max_size": 200}},
            {"id": "n7", "type": "WatershedNode",          "custom": {"min_distance": 12}},
            {"id": "n8", "type": "ImageCellNode",          "custom": {}},
            {"id": "n9", "type": "DataTableCellNode",      "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",         "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "image",       "to_node_id": "n3", "to_port": "image"},
            {"from_node_id": "n3", "from_port": "image",       "to_node_id": "n4", "to_port": "image"},
            {"from_node_id": "n4", "from_port": "mask",        "to_node_id": "n5", "to_port": "mask"},
            {"from_node_id": "n5", "from_port": "mask",        "to_node_id": "n6", "to_port": "mask"},
            {"from_node_id": "n6", "from_port": "mask",        "to_node_id": "n7", "to_port": "mask"},
            {"from_node_id": "n7", "from_port": "label_image", "to_node_id": "n8", "to_port": "in"},
            {"from_node_id": "n7", "from_port": "table",       "to_node_id": "n9", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 10: colocalization of two fluorescent channels
    # Source: scikit-image colocalization / ImageJ Coloc2 protocol
    # Shows: SplitRGBNode feeds ch1/ch2 directly into ColocalizationNode
    # ColocalizationNode outputs a metrics table AND a scatter-plot figure
    example_coloc = json.dumps({
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",        "custom": {}},
            {"id": "n2", "type": "SplitRGBNode",         "custom": {}},
            {"id": "n3", "type": "ColocalizationNode",   "custom": {}},
            {"id": "n4", "type": "DataTableCellNode",    "custom": {}},
            {"id": "n5", "type": "DataFigureCellNode",   "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",    "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "ch1",    "to_node_id": "n3", "to_port": "ch1"},
            {"from_node_id": "n2", "from_port": "ch2",    "to_node_id": "n3", "to_port": "ch2"},
            {"from_node_id": "n3", "from_port": "table",  "to_node_id": "n4", "to_port": "in"},
            {"from_node_id": "n3", "from_port": "figure", "to_node_id": "n5", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 11: spot / puncta detection with BlobDetect
    # Source: scikit-image "Blob detection" example (LoG / DoH)
    # Shows: BlobDetectNode outputs an overlay <image> AND a measurements <table>
    # Good for: vesicles, fluorescent dots, cell nuclei as spots, synaptic puncta
    example_blob = json.dumps({
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",      "custom": {}},
            {"id": "n2", "type": "BlobDetectNode",     "custom": {"min_sigma": 3, "max_sigma": 15}},
            {"id": "n3", "type": "ImageCellNode",      "custom": {}},
            {"id": "n4", "type": "DataTableCellNode",  "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",     "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "overlay", "to_node_id": "n3", "to_port": "in"},
            {"from_node_id": "n2", "from_port": "table",   "to_node_id": "n4", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 12: vessel / filament detection with Frangi filter
    # Source: scikit-image "Frangi vesselness filter" example
    # Shows: Frangi enhances tubular structures → threshold → skeletonize → measure
    # Good for: blood vessels, axons, actin filaments, retinal vasculature
    example_frangi = json.dumps({
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",      "custom": {}},
            {"id": "n2", "type": "FrangiNode",         "custom": {}},
            {"id": "n3", "type": "BinaryThresholdNode","custom": {}},
            {"id": "n4", "type": "SkeletonizeNode",    "custom": {}},
            {"id": "n5", "type": "ParticlePropsNode",  "custom": {}},
            {"id": "n6", "type": "DataTableCellNode",  "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",   "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "image", "to_node_id": "n3", "to_port": "image"},
            {"from_node_id": "n3", "from_port": "mask",  "to_node_id": "n4", "to_port": "mask"},
            {"from_node_id": "n4", "from_port": "mask",  "to_node_id": "n5", "to_port": "mask"},
            {"from_node_id": "n5", "from_port": "table", "to_node_id": "n6", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 13: GLCM texture feature extraction
    # Source: scikit-image "GLCM texture features" example
    # Shows: convert to grayscale first, then GLCMTextureNode outputs a feature table
    # Good for: tissue classification, surface roughness, material characterisation
    example_glcm = json.dumps({
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",      "custom": {}},
            {"id": "n2", "type": "RGBToGrayNode",      "custom": {}},
            {"id": "n3", "type": "GLCMTextureNode",    "custom": {}},
            {"id": "n4", "type": "DataTableCellNode",  "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",   "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "image", "to_node_id": "n3", "to_port": "image"},
            {"from_node_id": "n3", "from_port": "table", "to_node_id": "n4", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 8: watershed — overlapping object segmentation + measurement
    # Shows:
    #   (a) WatershedNode requires a <mask> input — image MUST be thresholded first
    #   (b) WatershedNode already outputs region props in its 'table' port — NEVER chain
    #       ParticlePropsNode or ImageStatsNode after watershed
    #   (c) Both outputs (label_image and table) can be consumed independently
    example_watershed = json.dumps({
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",       "custom": {}},
            {"id": "n2", "type": "BinaryThresholdNode", "custom": {}},
            {"id": "n3", "type": "WatershedNode",       "custom": {"min_distance": 15}},
            {"id": "n4", "type": "ImageCellNode",       "custom": {}},
            {"id": "n5", "type": "DataTableCellNode",   "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",         "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "mask",        "to_node_id": "n3", "to_port": "mask"},
            {"from_node_id": "n3", "from_port": "label_image", "to_node_id": "n4", "to_port": "in"},
            {"from_node_id": "n3", "from_port": "table",       "to_node_id": "n5", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 14: IHC / histology stain ratio
    # Source: standard colour-deconvolution quantification protocol
    # Shows:
    #   (a) ColorDeconvolutionNode outputs ch1/ch2/ch3 — all <image> type
    #   (b) MUST threshold each channel (BinaryThresholdNode) before ImageStatsNode
    #   (c) Use col_prefix to distinguish the two tables when both feed TwoTableMathNode
    #   (d) TwoTableMathNode computes the scalar ratio between the two area values
    example_ihc_ratio = json.dumps({
        "nodes": [
            {"id": "n1", "type": "ImageReadNode",          "custom": {}},
            {"id": "n2", "type": "ColorDeconvolutionNode", "custom": {"stain": "Masson Trichrome"}},
            {"id": "n3", "type": "BinaryThresholdNode",    "custom": {}},
            {"id": "n4", "type": "BinaryThresholdNode",    "custom": {}},
            {"id": "n5", "type": "ImageStatsNode",         "custom": {"col_prefix": "ch1_", "per_channel": False}},
            {"id": "n6", "type": "ImageStatsNode",         "custom": {"col_prefix": "ch2_", "per_channel": False}},
            {"id": "n7", "type": "TwoTableMathNode",         "custom": {"operation": "left / right",
                                                                       "left_column": "ch1_area_px",
                                                                       "right_column": "ch2_area_px"}},
            {"id": "n8", "type": "DataTableCellNode",      "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1", "from_port": "out",    "to_node_id": "n2", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "ch1",    "to_node_id": "n3", "to_port": "image"},
            {"from_node_id": "n2", "from_port": "ch2",    "to_node_id": "n4", "to_port": "image"},
            {"from_node_id": "n3", "from_port": "mask",   "to_node_id": "n5", "to_port": "mask"},
            {"from_node_id": "n4", "from_port": "mask",   "to_node_id": "n6", "to_port": "mask"},
            {"from_node_id": "n5", "from_port": "table",  "to_node_id": "n7", "to_port": "left"},
            {"from_node_id": "n6", "from_port": "table",  "to_node_id": "n7", "to_port": "right"},
            {"from_node_id": "n7", "from_port": "result", "to_node_id": "n8", "to_port": "in"}
        ]
    }, indent=2)

    # Few-shot example 15: pericellar ring analysis with cleaned cell mask
    # Shows the CORRECT placement of FillHolesNode + RemoveSmallObjectsNode:
    # they apply to the CELL mask BEFORE DistanceRingMaskNode, not to the
    # measurement mask after thresholding the signal channel.
    example_ring_cleaned = json.dumps({
        "nodes": [
            {"id": "n1",  "type": "ImageReadNode",           "custom": {}},
            {"id": "n2",  "type": "SplitRGBNode",            "custom": {}},
            {"id": "n3",  "type": "EqualizeAdapthistNode",   "custom": {}},
            {"id": "n4",  "type": "BinaryThresholdNode",     "custom": {}},
            {"id": "n5",  "type": "FillHolesNode",           "custom": {}},
            {"id": "n6",  "type": "RemoveSmallObjectsNode",  "custom": {"max_size": 500}},
            {"id": "n7",  "type": "DistanceRingMaskNode",    "custom": {"local_distance": 100}},
            {"id": "n8",  "type": "ImageMathNode",           "custom": {"operation": "A \u00d7 B (apply mask)"}},
            {"id": "n9",  "type": "BinaryThresholdNode",     "custom": {"thresh_state": [185.0, 1], "auto_otsu_per_image": False}},
            {"id": "n10", "type": "ImageStatsNode",          "custom": {"col_prefix": "collagen_", "per_channel": False}},
            {"id": "n11", "type": "DataTableCellNode",       "custom": {}}
        ],
        "edges": [
            {"from_node_id": "n1",  "from_port": "out",       "to_node_id": "n2",  "to_port": "image"},
            {"from_node_id": "n2",  "from_port": "red",       "to_node_id": "n3",  "to_port": "image"},
            {"from_node_id": "n3",  "from_port": "image",     "to_node_id": "n4",  "to_port": "image"},
            {"from_node_id": "n4",  "from_port": "mask",      "to_node_id": "n5",  "to_port": "mask"},
            {"from_node_id": "n5",  "from_port": "mask",      "to_node_id": "n6",  "to_port": "mask"},
            {"from_node_id": "n6",  "from_port": "mask",      "to_node_id": "n7",  "to_port": "mask"},
            {"from_node_id": "n2",  "from_port": "green",     "to_node_id": "n8",  "to_port": "A (image/mask)"},
            {"from_node_id": "n7",  "from_port": "ring_mask", "to_node_id": "n8",  "to_port": "B (mask)"},
            {"from_node_id": "n8",  "from_port": "image",     "to_node_id": "n9",  "to_port": "image"},
            {"from_node_id": "n9",  "from_port": "mask",      "to_node_id": "n10", "to_port": "mask"},
            {"from_node_id": "n10", "from_port": "table",     "to_node_id": "n11", "to_port": "in"}
        ]
    }, indent=2)

    return (
        "You are a workflow assistant for a scientific node-graph editor called Synapse.\n"
        "When the user describes a task, respond with ONLY a JSON workflow object.\n\n"
        "Rules:\n"
        "1. Use ONLY class names from the catalog.\n"
        "2. Port types must match: <image>/<mask>/<table>/<figure>/<any>. "
        "<image> ≠ <mask> — always threshold (<image>→BinaryThresholdNode→<mask>) before a mask port. "
        "ColorDeconvolutionNode ch1/ch2/ch3 are <image>, not <mask>. "
        "SplitRGBNode for 'red/green/blue channel' queries; ColorDeconvolutionNode ONLY for named stains (H&E, DAB, Masson, etc.).\n"
        "3. Use exact port names from the catalog. Ports can share the same name across nodes.\n"
        "4. Node IDs: n1, n2, n3, …  Every non-terminal node must have at least one outgoing edge.\n"
        "5. 'custom': never set thresh_widget or bc_widget (UI internals). "
        "Use thresh_state:[value,direction] for BinaryThresholdNode "
        "(direction 1 = keep pixels > threshold, e.g. 'brighter than X', 'higher than X', 'intensity above X'; "
        "direction 0 = keep pixels <= threshold, e.g. 'darker than X', 'below or equal to X'); "
        "bc_range:[min,max] for BrightnessContrastNode. "
        "IMPORTANT: whenever you set an explicit thresh_state value, also set auto_otsu_per_image:false — "
        "the default is true (auto-Otsu), which would override your explicit value otherwise.\n"
        "6. JSON: true/false/null (lowercase). No Python literals. No markdown fences around the output.\n"
        "7. Single image → ImageReadNode. Batch directory for blanked-ratio → ImageGroupDatasetNode.\n"
        "8. Terminal nodes: <table>→DataTableCellNode | <image>→ImageCellNode | <figure>→DataFigureCellNode | mixed→DisplayNode.\n"
        "9. Catalog 'cols:...' are actual output column names — use them in x_col/y_col/value_col/etc.\n"
        "10. DistanceRingMaskNode is the ONLY node with a 'ring_mask' output. "
        "RemoveSmallObjectsNode/FillHolesNode/BinaryThresholdNode all output 'mask'. "
        "Never skip DistanceRingMaskNode in the edge chain.\n"
        "11. WatershedNode needs <mask> input; its outputs are 'label_image' and 'table' (not 'props') — do not add ParticlePropsNode after it.\n"
        "12. All plot nodes (Violin, Box, Bar, Swarm, Scatter, Histogram, Kde) use 'data' as the table input port.\n\n"
        "Image analysis tips (apply when relevant):\n"
        "- After thresholding, ALWAYS consider FillHolesNode (close interior gaps) and RemoveSmallObjectsNode "
        "(remove debris/noise specks) before using the mask for measurement or as input to another node.\n"
        "- For fluorescence images with uneven illumination, add RollingBallNode before thresholding.\n"
        "- For noisy images, add GaussianBlurNode before thresholding to avoid fragmented masks.\n"
        "- For low-contrast images, add EqualizeAdapthistNode before thresholding.\n"
        "- For touching or overlapping objects, use WatershedNode after mask cleanup to separate them.\n\n"
        f"Example 1 — CSV → outlier removal → swarm plot:\n{example_csv}\n\n"
        f"Example 2 — single image → SplitRGB → per-channel processing:\n{example_image}\n\n"
        f"Example 3 — <any> port (DataSummaryNode accepts any type):\n{example_any}\n\n"
        f"Example 4 — threshold → keep largest object → apply mask to image "
        f"(ImageMathNode op: 'A × B (apply mask)'):\n{example_mask}\n\n"
        f"Example 5 — long-format CSV → violin plot:\n{example_violin}\n\n"
        f"Example 6 — CSV → outlier removal → pairwise stats → bar plot with brackets "
        f"(OutlierDetectionNode.kept → both PairwiseComparisonNode.in AND BarPlotNode.data):\n{example_bar_stats}\n\n"
        f"Example 7 — threshold → particle props → histogram:\n{example_cols}\n\n"
        f"Example 8 — watershed for overlapping/touching objects "
        f"(image → BinaryThresholdNode → WatershedNode.mask; "
        f"WatershedNode.table already contains area/centroid/shape — pipe it directly to DataTableCellNode, never add ParticlePropsNode after WatershedNode):\n{example_watershed}\n\n"
        f"Example 9 — nucleus segmentation: RollingBall → Gaussian → threshold → FillHoles → RemoveSmall → watershed:\n{example_nuclei}\n\n"
        f"Example 10 — two-channel colocalization (Pearson/Manders):\n{example_coloc}\n\n"
        f"Example 11 — blob/puncta detection (BlobDetectNode takes <image> directly):\n{example_blob}\n\n"
        f"Example 12 — Frangi filament/vessel segmentation:\n{example_frangi}\n\n"
        f"Example 13 — GLCM texture features (RGBToGray first):\n{example_glcm}\n\n"
        f"Example 14 — histological stain ratio via ColorDeconvolution "
        f"(ch1/ch2 are <image> → threshold each → ImageStatsNode with col_prefix (per_channel=false) → TwoTableMathNode):\n{example_ihc_ratio}\n\n"
        f"Example 15 — pericellar ring analysis with cleaned cell mask:\n{example_ring_cleaned}\n\n"
        "Now respond to the user's request using the catalog below.\n\n"
        "Node catalog (ClassName: description | in:[name<type>] → out:[name<type; cols:...]>] | props):\n"
        f"{catalog_text}"
    )


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------

class OllamaClient:
    DEFAULT_MODEL    = "gemma3:12b"
    DEFAULT_BASE_URL = "http://localhost:11434"
    CLOUD_BASE_URL   = "https://ollama.com"

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.api_key  = api_key

    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Returns available model names, or [] if Ollama is not reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5,
                                headers=self._headers())
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str) -> str:
        """
        Sends a chat request to Ollama and returns the raw JSON string
        from the model's message content.
        Raises requests.RequestException on network/HTTP errors.
        """
        payload = {
            "model":   self.model,
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
            "stream":  False,
            "options": {"temperature": 0.1},
        }
        # Structured output (format) is only supported by locally-run models.
        # Cloud-hosted models ignore it, so we only send it for local Ollama.
        if not self.api_key:
            payload["format"] = RESPONSE_SCHEMA
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers=self._headers(),
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# OpenAI client (cloud)
# ---------------------------------------------------------------------------

class OpenAIClient:
    DEFAULT_MODEL = "gpt-4o-mini"
    BASE_URL      = "https://api.openai.com/v1"

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
            # Keep only GPT chat-completion models; exclude fine-tune / embedding / tts / whisper / dall-e
            return sorted(
                m["id"] for m in resp.json().get("data", [])
                if m["id"].startswith("gpt-") or m["id"].startswith("o1") or m["id"].startswith("o3")
            )
        except Exception:
            return []

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
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
    def chat(self, system: str, user: str) -> str:
        """
        Runs inference and returns the raw JSON string.
        Raises FileNotFoundError if the GGUF file is missing.
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

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
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

    def chat(self, system: str, user: str) -> str:
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
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

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": [
                {"role": "user", "content": user},
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

    def chat(self, system: str, user: str) -> str:
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
    Converts the current node graph canvas into the same LLM JSON format
    used by WorkflowLoader: {"nodes": [...], "edges": [...]}.

    Node IDs are assigned sequentially (n1, n2, …) in the order returned by
    graph.all_nodes().  Edge port names are taken directly from the
    NodeGraphQt port objects, so they match the catalog exactly.
    """
    all_nodes = graph.all_nodes()

    # Map internal NodeGraphQt UUID → our sequential LLM id
    id_map: dict[str, str] = {}
    nodes_out: list[dict] = []
    edges_out: list[dict] = []

    for idx, node in enumerate(all_nodes):
        llm_id = f"n{idx + 1}"
        id_map[node.id] = llm_id

        # Collect only user-facing custom properties
        custom: dict = {}
        try:
            for k, v in node.model.custom_properties.items():
                if k not in _IGNORE_PROPS and not k.startswith("_"):
                    custom[k] = v
        except Exception:
            pass

        nodes_out.append({
            "id":     llm_id,
            "type":   type(node).__name__,
            "custom": custom,
        })

    # Collect edges by iterating every output port of every node
    for node in all_nodes:
        src_id = id_map.get(node.id)
        if src_id is None:
            continue
        for port_name, port in node.outputs().items():
            for connected in port.connected_ports():
                dst_id = id_map.get(connected.node().id)
                if dst_id:
                    edges_out.append({
                        "from_node_id": src_id,
                        "from_port":    port_name,
                        "to_node_id":   dst_id,
                        "to_port":      connected.name(),
                    })

    return {"nodes": nodes_out, "edges": edges_out}


# ---------------------------------------------------------------------------
# WorkflowLoader — builds nodes + edges in the NodeGraph
# ---------------------------------------------------------------------------

class WorkflowLoader:
    COLS  = 15   # nodes per row before wrapping
    X_PAD = 80   # horizontal gap between nodes (px)
    Y_PAD = 60   # vertical gap between nodes (px)

    def __init__(self, graph):
        self.graph = graph

    # ------------------------------------------------------------------
    def build(
        self,
        workflow: dict,
        origin_x: int = 100,
        origin_y: int = 100,
    ) -> tuple[bool, str]:
        """
        Creates nodes and edges described by *workflow*.
        Returns (success, message).  On partial success (some edges skipped)
        returns (True, warning_text).
        """
        node_map: dict[str, object] = {}  # LLM id → node instance
        warnings: list[str] = []
        created: list[tuple[str, object]] = []  # (llm_id, node) in order

        # --- Pass 1: create nodes at origin, apply properties ------------
        for idx, node_def in enumerate(workflow.get("nodes", [])):
            llm_id     = node_def.get("id", f"n{idx + 1}")
            class_name = node_def.get("type", "")
            custom     = node_def.get("custom") or {}

            identifier = self._find_identifier(class_name)
            if identifier is None:
                warnings.append(f"Unknown node type '{class_name}' — skipped.")
                continue

            node = self.graph.create_node(identifier, push_undo=True)
            node.set_pos(origin_x, origin_y)  # temporary; repositioned in pass 2

            # Apply configurable properties.
            # Some nodes (BaseImageProcessNode subclasses) auto-evaluate when a
            # property changes and live_preview=True.  With no inputs connected
            # yet the evaluation fails and the node turns red.  Suppress this by
            # temporarily disabling live_preview while we apply the custom props.
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
                    node.set_property(k, v)
                except Exception:
                    pass  # silently skip invalid/unsupported properties

            if was_live:
                try:
                    node.set_property('live_preview', True)
                except Exception:
                    pass

            node_map[llm_id] = node
            created.append((llm_id, node))

        # Force scene to compute node sizes before we read bounding rects
        QtWidgets.QApplication.processEvents()

        # --- Pass 2: layout based on actual rendered node sizes ----------
        cur_x     = origin_x
        cur_y     = origin_y
        row_h     = 0
        col       = 0

        for _, node in created:
            try:
                rect = node.view.boundingRect()
                w = rect.width()
                h = rect.height()
            except Exception:
                w, h = 200, 120  # fallback if view isn't available yet

            if col > 0 and col >= self.COLS:
                cur_x  = origin_x
                cur_y += row_h + self.Y_PAD
                row_h  = 0
                col    = 0

            node.set_pos(cur_x, cur_y)
            cur_x += w + self.X_PAD
            row_h  = max(row_h, h)
            col   += 1

        # --- Step 2: connect edges ----------------------------------------
        for edge in workflow.get("edges", []):
            src_id    = edge.get("from_node_id", "")
            src_port  = edge.get("from_port", "")
            dst_id    = edge.get("to_node_id", "")
            dst_port  = edge.get("to_port", "")

            src_node = node_map.get(src_id)
            dst_node = node_map.get(dst_id)

            if src_node is None or dst_node is None:
                warnings.append(
                    f"Edge {src_id}.{src_port} → {dst_id}.{dst_port}: "
                    f"node not found — skipped."
                )
                continue

            out_port = src_node.get_output(src_port)
            in_port  = dst_node.get_input(dst_port)

            if out_port is None or in_port is None:
                warnings.append(
                    f"Edge {src_id}.{src_port} → {dst_id}.{dst_port}: "
                    f"port not found — skipped."
                )
                continue

            try:
                out_port.connect_to(in_port)
            except Exception as exc:
                warnings.append(
                    f"Edge {src_id}.{src_port} → {dst_id}.{dst_port}: "
                    f"connect failed ({exc}) — skipped."
                )

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
        gguf_browse_btn.setFixedWidth(28)
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
        refresh_btn.setFixedWidth(28)
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

        # --- Generate button ------------------------------------------
        self._generate_btn = QtWidgets.QPushButton("Generate Workflow")
        self._generate_btn.clicked.connect(self._on_generate)
        layout.addWidget(self._generate_btn)

        layout.addWidget(_make_separator())

        # --- Status label ---------------------------------------------
        self._status_label = QtWidgets.QLabel("Ready")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # --- JSON preview --------------------------------------------
        layout.addWidget(QtWidgets.QLabel("Generated workflow (JSON preview):"))
        self._preview_edit = QtWidgets.QPlainTextEdit()
        self._preview_edit.setReadOnly(True)
        self._preview_edit.setFont(QtGui.QFont("Courier New", 9))
        self._preview_edit.setMinimumHeight(160)
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
            # User-specified path takes priority; otherwise auto-detect in package dir
            _user_path = Path(self._gguf_edit.text().strip()) if hasattr(self, "_gguf_edit") and self._gguf_edit.text().strip() else None
            if _user_path and _user_path.exists():
                _gguf = _user_path
            else:
                _q4 = self._GGUF_DIR / "synapse-qwen-0.8b.Q4_K_M.gguf"
                _q8 = self._GGUF_DIR / "synapse-qwen-0.8b.q8_0.gguf"
                _gguf = _q4 if _q4.exists() else _q8
            self._client  = LlamaCppClient(str(_gguf), n_ctx=16384, n_gpu_layers=-1)
            default_model = _gguf.name
            no_model_msg  = (
                "GGUF model not found.\n"
                "Enter the path to your .gguf file and click ⟳, or\n"
                "run: python finetune/convert.py"
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

        # Spin up background thread (mirrors GraphWorker pattern)
        self._worker = LLMWorker(self._client, self._system, user_msg)
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
