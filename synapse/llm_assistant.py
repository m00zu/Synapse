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
            "description": "Connections as [source_id, target_id] pairs.",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
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

    Compact mode (default): port types only, key props only.
    Verbose mode: includes full docstrings and all configurable properties.
    """
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

    # Append plugin nodes (auto-discovered at runtime via plugin_loader)
    try:
        from .plugin_loader import get_plugin_catalog_entries
        for entry in get_plugin_catalog_entries():
            ins  = _fmt_ports(entry['inputs'])
            outs = _fmt_ports(entry['outputs'])
            lines.append(
                f"- {entry['class_name']}: {entry['description']} | in:[{ins}] → out:[{outs}]"
            )
    except ImportError:
        pass

    return "\n".join(lines)


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
        "9. JSON only: true/false/null. No markdown fences.\n\n"
        f"Example 1 — CSV → stats → bar plot (fan-out: node 2→3 and 2→4):\n{example_stats}\n\n"
        f"Example 2 — mask pipeline with fan-out (node 1→2 and 1→6):\n{example_mask}\n\n"
        f"Example 3 — nucleus segmentation (fan-out: node 7→8 and 7→9):\n{example_nuclei}\n\n"
        "Node catalog:\n"
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
    COLS  = 15   # nodes per row before wrapping
    X_PAD = 80   # horizontal gap between nodes (px)
    Y_PAD = 60   # vertical gap between nodes (px)

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
                w, h = 200, 120

            if col > 0 and col >= self.COLS:
                cur_x  = origin_x
                cur_y += row_h + self.Y_PAD
                row_h  = 0
                col    = 0

            node.set_pos(cur_x, cur_y)
            cur_x += w + self.X_PAD
            row_h  = max(row_h, h)
            col   += 1

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
                    out_port = src_node.get_output(hint_out) if hint_out else None
                    in_port  = dst_node.get_input(hint_in) if hint_in else None
                    # Fall back to auto-wire for the missing side
                    if out_port and not in_port:
                        # Find compatible input for this specific output
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
        """Copy a ready-to-paste prompt to clipboard for use with web AI interfaces."""
        question = self._question_edit.toPlainText().strip()
        if not question:
            self._status_label.setText("Please enter a question first.")
            return

        # Build the full prompt
        use_context = (
            self._ctx_check.isChecked() and bool(self.graph.all_nodes())
        )
        prompt_parts = [self._system, ""]
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
        system = _FINETUNE_SYS_PROMPT if provider == "Synapse Fine-tune" else self._system

        # Spin up background thread (mirrors GraphWorker pattern)
        self._worker = LLMWorker(self._client, system, user_msg)
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
