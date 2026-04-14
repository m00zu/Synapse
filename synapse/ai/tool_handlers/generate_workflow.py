"""generate_workflow tool handler — two-pass workflow generation from a goal."""
from __future__ import annotations

import json


def make_generate_workflow_handler(graph, client):
    """Build a handler bound to (graph, client).

    ``client`` must expose ``chat_multi(system: str, messages: list[dict]) -> str``.
    Phase 1's OllamaClient / OpenAIClient / ClaudeClient etc. all satisfy this.
    """

    def _handler(tool_input: dict) -> dict:
        goal = (tool_input or {}).get("goal")
        if not goal:
            return {"error": "generate_workflow requires 'goal' (string)."}
        constraints = (tool_input or {}).get("constraints") or ""

        # Import lazily — llm_assistant is heavy.
        from synapse.llm_assistant import (
            build_condensed_catalog,
            build_selection_prompt,
            build_detailed_cards,
            build_system_prompt,
        )

        catalog = build_condensed_catalog()
        user_message = goal + (("\n\nConstraints: " + constraints) if constraints else "")

        # Pass 1: select relevant node class names.
        sel_prompt = build_selection_prompt(catalog)
        try:
            raw1 = client.chat_multi(
                system=sel_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            selection = json.loads(raw1)
            selected_names: list[str] = list(selection.get("nodes", []))
        except Exception as e:
            return {"error": f"Pass 1 (node selection) failed: {type(e).__name__}: {e}"}

        # Pass 2: generate full workflow with detailed cards.
        detail_sys = build_system_prompt(build_detailed_cards(selected_names) or catalog)
        try:
            raw2 = client.chat_multi(
                system=detail_sys,
                messages=[{"role": "user", "content": user_message}],
            )
            workflow = json.loads(raw2)
        except Exception as e:
            return {"error": f"Pass 2 (workflow JSON) failed: {type(e).__name__}: {e}"}

        nodes = workflow.get("nodes") or []
        edges = workflow.get("edges") or []
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "preview_types": [n.get("type", "?") for n in nodes],
            "workflow": workflow,
            "canvas_was_empty": len(list(graph.all_nodes())) == 0,
        }

    return _handler
