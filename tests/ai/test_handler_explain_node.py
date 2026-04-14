from synapse.ai.tool_handlers.explain_node import explain_node_handler


def test_explain_known_node():
    out = explain_node_handler({"node_type": "ParticlePropsNode"})
    # The detailed card is stringified; we just assert it's non-empty with the name.
    assert "ParticleProps" in out.get("card", "")
    assert out.get("node_type") == "ParticlePropsNode"


def test_explain_unknown_node_returns_error():
    out = explain_node_handler({"node_type": "DoesNotExistNode"})
    assert "error" in out


def test_explain_missing_input_returns_error():
    out = explain_node_handler({})
    assert "error" in out
    assert "node_type" in out["error"]
