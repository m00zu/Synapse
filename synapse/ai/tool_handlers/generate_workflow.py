"""generate_workflow tool handler — two-pass workflow generation from a goal."""
from __future__ import annotations

import json
import re


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)\n```", re.DOTALL)


def _coerce_json(raw: str) -> dict:
    """Pull the first JSON object out of potentially-messy LLM output.

    Some cloud models wrap JSON in prose or markdown fences even when told
    not to. We try:
      1. raw JSON
      2. markdown fenced block
      3. first `{...}` span (greedy outer-brace match)
    """
    s = (raw or "").strip()
    # Happy path — parses directly.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fenced code block.
    m = _FENCE_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Greedy outer-brace span.
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(s[first:last + 1])
        except json.JSONDecodeError:
            pass
    # Give up — re-raise the original exception with the raw text prefix so
    # the caller can see what the model actually said.
    raise json.JSONDecodeError(
        f"could not extract JSON from model output (first 200 chars): {s[:200]!r}",
        s, 0,
    )


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
        force_overwrite = bool((tool_input or {}).get("force_overwrite"))

        # Guard: refuse on a populated canvas unless the caller explicitly
        # overrides. Building a fresh standalone pipeline when the user
        # already has one typically means duplicated nodes and wasted
        # cached evaluations — extension via modify_workflow is the right
        # path. The LLM sees this error, then follows the prompt's
        # "Extending an existing canvas" rubric.
        if len(list(graph.all_nodes())) > 0 and not force_overwrite:
            return {"error": (
                "Canvas is not empty. Do NOT call generate_workflow to add to "
                "an existing pipeline — that would create a disconnected or "
                "duplicated workflow. Instead: call inspect_canvas to see the "
                "existing nodes, then call modify_workflow ONCE with all "
                "add_node + set_prop + connect ops needed, using the real "
                "existing node ids as src/dst for the attachment connect op. "
                "If the user explicitly asks for a completely separate "
                "pipeline from scratch, retry with force_overwrite=true."
            )}

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
            selection = _coerce_json(raw1)
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
            workflow = _coerce_json(raw2)
        except Exception as e:
            return {"error": f"Pass 2 (workflow JSON) failed: {type(e).__name__}: {e}"}

        nodes = workflow.get("nodes") or []
        edges = workflow.get("edges") or []
        canvas_was_empty = len(list(graph.all_nodes())) == 0
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "preview_types": [n.get("type", "?") for n in nodes],
            "workflow": workflow,
            "canvas_was_empty": canvas_was_empty,
            # Tell the model *unambiguously* whether the workflow is now on the
            # canvas. True means the UI will auto-apply it (empty canvas);
            # False means it's waiting on a user Apply/Discard confirm.
            "applied": canvas_was_empty,
            "hint": (
                "Workflow has been applied to the canvas. Do NOT call "
                "modify_workflow to add the same nodes again — they are "
                "already there. If you want to tweak or extend the workflow "
                "(add/remove/wire nodes, set properties), use modify_workflow."
                if canvas_was_empty else
                "User will be prompted to Apply or Discard this workflow. "
                "Do NOT call modify_workflow now; wait for confirmation."
            ),
        }

    return _handler
