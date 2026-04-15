"""Tool schemas and dispatcher for the Synapse AI orchestrator.

Phase 2a: schemas + dispatcher + pure tool handlers (no LLM client calls).
Phase 2b: ChatOrchestrator + client tool-calling wire them together.
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = ["TOOLS", "TOOL_NAMES", "ToolDispatcher"]


TOOL_NAMES = (
    "generate_workflow",
    "modify_workflow",
    "write_python_script",
    "inspect_canvas",
    "explain_node",
    "read_node_output",
)


TOOLS: list[dict] = [
    {
        "name": "generate_workflow",
        "description": (
            "Generate a complete Synapse workflow from a natural-language goal. "
            "Use this ONLY when the canvas is empty, or the user explicitly "
            "asks for a new, separate pipeline. On an empty canvas the "
            "workflow is AUTO-APPLIED and the call returns with applied=true. "
            "On a non-empty canvas the tool returns an error asking you to use "
            "inspect_canvas + modify_workflow instead (so you don't duplicate "
            "existing nodes or lose cached evaluations). If the user really "
            "wants a from-scratch pipeline alongside existing work, retry with "
            "force_overwrite=true and the UI will prompt for Apply/Discard."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "User's goal in plain English."},
                "constraints": {"type": "string", "description": "Optional extra constraints."},
                "force_overwrite": {"type": "boolean",
                                    "description": "Allow generation on a non-empty canvas. "
                                                   "Only set when the user explicitly wants a "
                                                   "new separate pipeline."},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "modify_workflow",
        "description": (
            "Apply a list of graph operations to the current canvas in a single undo group. "
            "Partial success is allowed — failed ops are reported, successful ones stay."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "description": "Ordered list of graph operations.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["add_node", "remove_node", "connect",
                                         "disconnect", "set_prop"],
                            },
                            "type": {"type": "string", "description": "Node class name (add_node)."},
                            "id":   {"type": "string", "description": "Node id to target (most ops)."},
                            "src":  {"type": "string", "description": "Source node id (connect/disconnect)."},
                            "dst":  {"type": "string", "description": "Destination node id (connect/disconnect)."},
                            "src_port": {"type": "string"},
                            "dst_port": {"type": "string"},
                            "prop": {"type": "string", "description": "Property name (set_prop)."},
                            "value": {"description": "Property value (set_prop). Any JSON type."},
                        },
                        "required": ["op"],
                    },
                }
            },
            "required": ["operations"],
        },
    },
    {
        "name": "write_python_script",
        "description": (
            "Generate Python code for a PythonScriptNode. If node_id is given, the code "
            "is written directly into the node's script_code property (undoable). "
            "The node's input/output port counts are resized to match n_inputs/n_outputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Target PythonScriptNode id."},
                "description": {"type": "string", "description": "What the code should do."},
                "n_inputs":  {"type": "integer", "minimum": 1, "maximum": 8},
                "n_outputs": {"type": "integer", "minimum": 1, "maximum": 8},
                "input_hints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "port": {"type": "string"},
                            "kind": {"type": "string", "enum": ["table", "image", "mask", "scalar"]},
                            "schema": {"type": "string"},
                        },
                    },
                },
                "output_hints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "port": {"type": "string"},
                            "kind": {"type": "string", "enum": ["table", "image", "mask", "figure"]},
                        },
                    },
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "inspect_canvas",
        "description": (
            "Return nodes + edges + properties for the current canvas, capped at ~2000 "
            "output tokens. Use when you need full details about what is on the graph."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "If omitted, returns all nodes.",
                },
                "include_props": {"type": "boolean"},
            },
        },
    },
    {
        "name": "explain_node",
        "description": (
            "Return the authoritative documentation card (ports, props, docstring) "
            "for a Synapse node class by exact class name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_type": {"type": "string", "description": "Node class name, e.g. 'ParticlePropsNode'."},
            },
            "required": ["node_type"],
        },
    },
    {
        "name": "read_node_output",
        "description": (
            "Peek at the last-evaluated output(s) of one or more nodes. Nodes "
            "with multiple output ports (e.g. `OutlierDetectionNode` which "
            "emits BOTH `kept` and `removed` tables) return per-port data in "
            "a `ports` dict — you see each port's shape, columns, head, "
            "and type individually. Works on ANY evaluated node, not just "
            "terminal ones. "
            "Pass `node_id` for one node, or `node_ids` (up to 8) for a batch. "
            "Optionally pass `port` to target a specific output port name "
            "(e.g. `port: 'removed'`); without it you get all ports. "
            "Thumbnails attached only when vision-capable AND fewer than 3 "
            "images across the batch. Per-call token budget truncates oversized "
            "replies with `truncated: true`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string",
                            "description": "Id of a single node to read."},
                "node_ids": {"type": "array",
                             "items": {"type": "string"},
                             "maxItems": 8,
                             "description": "Ids of multiple nodes to read (max 8)."},
                "port": {"type": "string",
                         "description": "Optional: read only this output port name. "
                                        "Omit to get all ports keyed by name under `ports`."},
            },
        },
    },
]


class ToolDispatcher:
    """Register handlers and dispatch by name."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict], Any]] = {}

    def register(self, name: str, handler: Callable[[dict], Any]) -> None:
        if name not in TOOL_NAMES:
            raise ValueError(f"Unknown tool name: {name}")
        if name in self._handlers:
            raise ValueError(f"Tool already registered: {name}")
        self._handlers[name] = handler

    def registered_names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def dispatch(self, name: str, tool_input: dict) -> Any:
        """Call the registered handler and return its result.

        Failure contract: any error — unknown tool name, exception inside the
        handler, or a non-dict return — is converted to ``{"error": "..."}``.
        Handlers should signal failure by returning a dict containing an
        ``"error"`` key rather than raising; both are accepted but returning
        a dict is preferred so tool output stays uniform for the orchestrator.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"No handler registered for tool: {name}"}
        try:
            result = handler(tool_input)
            if not isinstance(result, dict):
                return {"error": f"Tool {name} returned non-dict: {type(result).__name__}"}
            return result
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
