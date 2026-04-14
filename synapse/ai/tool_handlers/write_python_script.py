"""write_python_script tool handler — generate Python for a PythonScriptNode."""
from __future__ import annotations

import re

from synapse.ai.prompts import WRITE_PYTHON_SCRIPT_SUBPROMPT


_FENCE_RE = re.compile(r"^```(?:[a-zA-Z]*)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text.strip()


def make_write_python_script_handler(graph, client):
    """Bind a handler to (graph, client). Client must expose chat_multi()."""

    def _handler(tool_input: dict) -> dict:
        inp = tool_input or {}
        node_id = inp.get("node_id")
        description = inp.get("description")
        if not node_id:
            return {"error": "write_python_script requires 'node_id'."}
        if not description:
            return {"error": "write_python_script requires 'description'."}

        node = next((n for n in graph.all_nodes() if n.id == node_id), None)
        if node is None:
            return {"error": f"No node with id: {node_id}"}
        type_name = getattr(node, "type_name", type(node).__name__)
        if type_name != "PythonScriptNode":
            return {"error": (
                "write_python_script only targets PythonScriptNode; "
                f"node {node_id} is {type_name}."
            )}

        n_in = int(inp.get("n_inputs") or 1)
        n_out = int(inp.get("n_outputs") or 1)
        try:
            node.set_property("n_inputs", n_in)
            node.set_property("n_outputs", n_out)
        except Exception as e:
            return {"error": f"Failed to resize ports: {type(e).__name__}: {e}"}

        user_msg = (
            f"{description}\n\n"
            f"Ports: n_inputs={n_in}, n_outputs={n_out}.\n"
            f"input_hints={inp.get('input_hints') or []}\n"
            f"output_hints={inp.get('output_hints') or []}\n"
            "Return ONLY the code body."
        )
        try:
            raw = client.chat_multi(
                system=WRITE_PYTHON_SCRIPT_SUBPROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:
            return {"error": f"sub-LLM call failed: {type(e).__name__}: {e}"}

        code = _strip_fences(raw)
        try:
            node.set_property("script_code", code, push_undo=True)
        except TypeError:
            node.set_property("script_code", code)
        except Exception as e:
            return {"error": f"Failed to write script_code: {type(e).__name__}: {e}"}

        assigned = sorted(set(re.findall(r"\b(out_\d+)\b", code)))
        return {
            "target_node_id": node_id,
            "line_count": len(code.splitlines()),
            "assigned_outputs": assigned,
        }

    return _handler
