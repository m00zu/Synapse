"""Synapse AI package — clients, tools, orchestrator (Phase 2b)."""
from synapse.ai.feature_flags import get_use_orchestrator
from synapse.ai.prompts import BASE_SYSTEM_PROMPT, WRITE_PYTHON_SCRIPT_SUBPROMPT
from synapse.ai.context import graph_summary, estimate_tokens, HistoryRoller
from synapse.ai.tools import TOOLS, TOOL_NAMES, ToolDispatcher

__all__ = [
    "get_use_orchestrator",
    "BASE_SYSTEM_PROMPT", "WRITE_PYTHON_SCRIPT_SUBPROMPT",
    "graph_summary", "estimate_tokens", "HistoryRoller",
    "TOOLS", "TOOL_NAMES", "ToolDispatcher",
]
