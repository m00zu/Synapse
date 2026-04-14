"""Read-only accessors for Synapse AI feature flags.

Flags live in ``~/.synapse_llm_config.json`` under the ``ai`` key, e.g.

    {
      "ai": {
        "use_orchestrator": false
      }
    }

Missing/unreadable config → defaults (all flags False).
"""
from __future__ import annotations

import json
from pathlib import Path

_CONFIG_PATH = Path.home() / ".synapse_llm_config.json"


def _load() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text())
    except Exception:
        return {}


def get_use_orchestrator() -> bool:
    """True when the ChatOrchestrator path is enabled (Phase 2b)."""
    return bool(_load().get("ai", {}).get("use_orchestrator", False))
