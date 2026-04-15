from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph
from synapse.ai.tool_handlers.generate_workflow import make_generate_workflow_handler


def _mock_client_yielding(pass1_json: str, pass2_json: str):
    """A minimal stand-in for the two sequential chat_multi calls the handler makes."""
    client = MagicMock()
    client.chat_multi.side_effect = [pass1_json, pass2_json]
    return client


def test_generate_workflow_returns_preview_payload():
    g = FakeGraph()
    client = _mock_client_yielding(
        pass1_json='{"nodes": ["CSVLoader", "SortTable"]}',
        pass2_json='{"nodes":[{"id":1,"type":"CSVLoader","props":{"path":"x.csv"}},'
                   '{"id":2,"type":"SortTable","props":{"column":"value"}}],'
                   '"edges":[[1,2]]}',
    )
    handler = make_generate_workflow_handler(graph=g, client=client)
    out = handler({"goal": "Load a CSV and sort it"})
    assert "error" not in out
    assert out["node_count"] == 2
    assert out["edge_count"] == 1
    assert out["preview_types"] == ["CSVLoader", "SortTable"]
    assert "workflow" in out
    assert out["canvas_was_empty"] is True  # FakeGraph is empty


def test_generate_workflow_missing_goal_returns_error():
    handler = make_generate_workflow_handler(graph=FakeGraph(), client=MagicMock())
    out = handler({})
    assert "error" in out and "goal" in out["error"]


def test_generate_workflow_bad_json_returns_error():
    client = _mock_client_yielding("not json", "also not json")
    handler = make_generate_workflow_handler(graph=FakeGraph(), client=client)
    out = handler({"goal": "x"})
    assert "error" in out


def test_generate_workflow_refuses_on_non_empty_canvas():
    """Guard against duplicate pipelines: refuse generate_workflow when the
    canvas already has nodes, pointing the model at modify_workflow."""
    from tests.ai.fakes import FakeNode
    g = FakeGraph(); g.add_node(FakeNode("pre", "Existing"))
    handler = make_generate_workflow_handler(graph=g, client=MagicMock())
    out = handler({"goal": "build something"})
    assert "error" in out
    assert "modify_workflow" in out["error"]
    assert "inspect_canvas" in out["error"]
    assert "force_overwrite" in out["error"]


def test_generate_workflow_force_overwrite_bypasses_non_empty_guard():
    """force_overwrite=true allows a from-scratch pipeline even with
    existing nodes — used when the user explicitly wants a separate one."""
    from tests.ai.fakes import FakeNode
    g = FakeGraph(); g.add_node(FakeNode("pre", "Existing"))
    client = _mock_client_yielding(
        pass1_json='{"nodes": ["CSVLoader"]}',
        pass2_json='{"nodes":[{"id":1,"type":"CSVLoader"}],"edges":[]}',
    )
    handler = make_generate_workflow_handler(graph=g, client=client)
    out = handler({"goal": "x", "force_overwrite": True})
    assert "error" not in out
    assert out["canvas_was_empty"] is False
    assert out["applied"] is False


def test_generate_workflow_tolerates_prose_before_json():
    """Cloud models often wrap JSON with a friendly preamble."""
    client = _mock_client_yielding(
        pass1_json='Sure! Here are the nodes I picked:\n\n{"nodes": ["CSVLoader"]}',
        pass2_json='Here is the workflow:\n{"nodes":[{"id":1,"type":"CSVLoader"}],"edges":[]}',
    )
    handler = make_generate_workflow_handler(graph=FakeGraph(), client=client)
    out = handler({"goal": "x"})
    assert "error" not in out
    assert out["node_count"] == 1


def test_generate_workflow_tolerates_markdown_fenced_json():
    client = _mock_client_yielding(
        pass1_json='```json\n{"nodes": ["CSVLoader"]}\n```',
        pass2_json='```json\n{"nodes":[{"id":1,"type":"CSVLoader"}],"edges":[]}\n```',
    )
    handler = make_generate_workflow_handler(graph=FakeGraph(), client=client)
    out = handler({"goal": "x"})
    assert "error" not in out
    assert out["preview_types"] == ["CSVLoader"]


def test_generate_workflow_rejects_fully_non_json_output():
    """When the coercer can't find any JSON, surface the failure with context."""
    client = _mock_client_yielding(
        pass1_json="I cannot help with that request.",
        pass2_json="",
    )
    handler = make_generate_workflow_handler(graph=FakeGraph(), client=client)
    out = handler({"goal": "x"})
    assert "error" in out
    assert "Pass 1" in out["error"]
