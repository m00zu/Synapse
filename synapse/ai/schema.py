"""Shared structured-output schemas for LLM clients.

`RESPONSE_SCHEMA` is Ollama's JSON schema for the workflow response format.
Lives here (not in llm_assistant.py) so ai.clients modules can import it
without a back-reference to llm_assistant.
"""

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
