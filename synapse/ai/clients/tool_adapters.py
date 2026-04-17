"""Convert provider-neutral tool schemas to each LLM provider's native shape.

Anthropic:  [{name, description, input_schema}]   (matches our internal format)
OpenAI:     [{type: "function", function: {name, description, parameters}}]
Gemini:     [{functionDeclarations: [{name, description, parameters}]}]
Fallback:   a system-prompt addendum describing the <tool_call> protocol.
"""
from __future__ import annotations

import json


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Anthropic's input_schema matches our internal shape; pass through."""
    return [
        {"name": t["name"], "description": t["description"],
         "input_schema": t["input_schema"]}
        for t in tools
    ]


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def to_gemini_tools(tools: list[dict]) -> list[dict]:
    return [{
        "functionDeclarations": [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
            for t in tools
        ]
    }]


_FALLBACK_HEADER = """\
You have access to the following tools. To call one, output exactly one line
starting with `<tool_call>` and ending with `</tool_call>` containing a JSON
object with keys "name" and "input", then STOP generating.

Example:
<tool_call>{"name": "inspect_canvas", "input": {}}</tool_call>

After a tool call, wait for the user message containing the result, then
continue. If you do not need to call a tool, just reply normally in markdown.
Never emit <tool_call>...</tool_call> as part of a larger explanation — the
line must stand alone.

Available tools:
"""


def build_fallback_prompt(tools: list[dict]) -> str:
    lines = [_FALLBACK_HEADER]
    for t in tools:
        lines.append(f"- name: {t['name']}")
        lines.append(f"  description: {t['description']}")
        lines.append(f"  input_schema: {json.dumps(t['input_schema'])}")
    return "\n".join(lines)
