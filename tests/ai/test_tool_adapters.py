import json

from synapse.ai.tools import TOOLS
from synapse.ai.clients.tool_adapters import (
    to_anthropic_tools, to_openai_tools, to_gemini_tools, build_fallback_prompt,
)


def test_anthropic_is_identity_shape():
    out = to_anthropic_tools(TOOLS)
    assert len(out) == 6
    for t in out:
        assert set(t.keys()) >= {"name", "description", "input_schema"}


def test_openai_wraps_in_function_envelope():
    out = to_openai_tools(TOOLS)
    assert len(out) == 6
    for t in out:
        assert t["type"] == "function"
        fn = t["function"]
        assert {"name", "description", "parameters"} <= set(fn)
        assert fn["parameters"]["type"] == "object"


def test_gemini_wraps_in_function_declarations_envelope():
    out = to_gemini_tools(TOOLS)
    assert len(out) == 1
    decls = out[0]["functionDeclarations"]
    assert len(decls) == 6
    for d in decls:
        assert {"name", "description", "parameters"} <= set(d)


def test_fallback_prompt_contains_protocol_and_schemas():
    txt = build_fallback_prompt(TOOLS)
    assert "<tool_call>" in txt and "</tool_call>" in txt
    for t in TOOLS:
        assert t["name"] in txt
    assert "input_schema" in txt or "parameters" in txt


def test_openai_stripped_default_keys_stay_stripped():
    out = to_openai_tools(TOOLS)
    txt = json.dumps(out)
    assert '"default"' not in txt
