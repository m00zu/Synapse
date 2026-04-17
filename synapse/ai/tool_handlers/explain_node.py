"""explain_node tool handler — return the catalog card for a node class."""
from __future__ import annotations


def explain_node_handler(tool_input: dict) -> dict:
    node_type = (tool_input or {}).get("node_type")
    if not node_type:
        return {"error": "explain_node requires 'node_type' (string)."}

    # build_detailed_cards lives in llm_assistant and reads the schema JSON.
    from synapse.llm_assistant import build_detailed_cards

    try:
        card = build_detailed_cards([node_type])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if not card or not card.strip():
        return {"error": f"No catalog entry for node type: {node_type}"}

    return {"node_type": node_type, "card": card}
