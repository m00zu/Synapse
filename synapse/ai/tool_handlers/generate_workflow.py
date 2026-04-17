"""generate_workflow tool handler — two-pass workflow generation from a goal."""
from __future__ import annotations

import json
import re
from pathlib import Path


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)\n```", re.DOTALL)

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "llm_node_schema.json"


def _load_schema() -> dict:
    try:
        return json.loads(_SCHEMA_PATH.read_text()).get("node_catalog", {})
    except Exception:
        return {}


def _port_names(port_list: list) -> list[str]:
    """Extract names from schema port lists (each entry is {name, type, ...})."""
    out = []
    for p in port_list or []:
        if isinstance(p, dict) and p.get("name"):
            out.append(p["name"])
        elif isinstance(p, str):
            out.append(p)
    return out


def _validate_port_hints(workflow: dict) -> list[str]:
    """Return a list of human-readable warnings for edges whose port hints do
    not match the source/target node's schema.

    Only fires when the hint is non-empty — the loader auto-wires by type when
    no hint is given, and that's fine. The warning value is to catch cases
    like SplitRGBNode with src_port='image' (hallucinated) that the loader
    silently auto-resolves to the first compatible output (e.g. 'red'),
    producing a workflow that applies cleanly but connects to the wrong
    channel.
    """
    schema = _load_schema()
    if not schema:
        return []
    nodes = workflow.get("nodes") or []
    id_to_type = {}
    for n in nodes:
        nid = n.get("id")
        ntype = n.get("type")
        if nid is not None and ntype:
            id_to_type[nid] = ntype
            id_to_type[str(nid)] = ntype
    warnings: list[str] = []
    for edge in workflow.get("edges") or []:
        if not isinstance(edge, list) or len(edge) < 3:
            continue
        src_id = edge[0]
        dst_id = edge[1]
        src_hint = edge[2] if len(edge) >= 3 else ""
        dst_hint = edge[3] if len(edge) >= 4 else ""
        src_type = id_to_type.get(src_id) or id_to_type.get(str(src_id))
        dst_type = id_to_type.get(dst_id) or id_to_type.get(str(dst_id))
        if src_hint and src_type and src_type in schema:
            outs = _port_names(schema[src_type].get("outputs"))
            if outs and src_hint not in outs:
                warnings.append(
                    f"Edge {src_id}->{dst_id}: source port {src_hint!r} not found on "
                    f"{src_type}. Valid outputs: {outs}. The loader auto-wired to "
                    f"'{outs[0]}' — if that's wrong, call modify_workflow to "
                    f"disconnect and reconnect with the right src_port."
                )
        if dst_hint and dst_type and dst_type in schema:
            ins = _port_names(schema[dst_type].get("inputs"))
            if ins and dst_hint not in ins:
                warnings.append(
                    f"Edge {src_id}->{dst_id}: target port {dst_hint!r} not found on "
                    f"{dst_type}. Valid inputs: {ins}. Loader auto-wired by type; "
                    f"if wrong, fix via modify_workflow."
                )
    return warnings


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
        warnings = _validate_port_hints(workflow)
        result = {
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
        if warnings:
            result["warnings"] = warnings
        return result

    return _handler
