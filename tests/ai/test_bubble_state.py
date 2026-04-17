"""Tests for synapse.ai.bubble_state — pure-Python bubble rendering."""
from synapse.ai.bubble_state import (
    _BubbleState, ToolChip, WorkflowProposal,
    render_bubble_html, parse_anchor,
)

COLORS = {
    "user_bg": "#1f6feb", "user_fg": "#fff",
    "ai_bg": "#161b22", "ai_fg": "#c9d1d9", "ai_label": "#58a6ff",
    "ai_border": "#30363d", "err_bg": "#3d1214", "err_fg": "#f85149",
    "sys_fg": "#8b949e",
}


def test_user_bubble_has_text_and_tail():
    s = _BubbleState(bubble_id="b1", role="user", text="hi")
    html = render_bubble_html(s, COLORS)
    assert "hi" in html
    assert "&#9698;" in html  # tail glyph


def test_assistant_bubble_streaming_shows_cursor():
    s = _BubbleState(bubble_id="b2", role="assistant", text="Thinking", streaming=True)
    html = render_bubble_html(s, COLORS)
    assert "Thinking" in html
    assert "●" in html  # streaming cursor


def test_assistant_bubble_with_chip_renders_pill_with_status():
    s = _BubbleState(bubble_id="b3", role="assistant")
    s.chips.append(ToolChip(
        chip_id="c1", name="inspect_canvas",
        input_preview="{}", status="running",
    ))
    html = render_bubble_html(s, COLORS)
    assert "inspect_canvas" in html
    assert "chip://b3/c1" in html
    assert "⋯" in html  # running glyph


def test_expanded_chip_renders_full_payload_block():
    s = _BubbleState(bubble_id="b4", role="assistant")
    s.chips.append(ToolChip(
        chip_id="c1", name="inspect_canvas",
        input_preview="{}", status="ok", result_summary="4 nodes",
        full_input={"foo": 1}, full_result={"nodes": [1, 2]},
    ))
    s.expanded_chips.add("c1")
    html = render_bubble_html(s, COLORS)
    assert "<pre" in html
    assert "&quot;foo&quot;" in html or '"foo"' in html
    assert "nodes" in html


def test_pending_workflow_proposal_shows_apply_and_discard_links():
    s = _BubbleState(bubble_id="b5", role="assistant")
    s.workflow = WorkflowProposal(
        node_count=3, edge_count=2,
        preview_types=["CSVReader", "SortTable", "TopN"],
    )
    html = render_bubble_html(s, COLORS)
    assert "apply://b5" in html
    assert "discard://b5" in html
    assert "3" in html and "2" in html


def test_applied_workflow_proposal_shows_only_applied_marker():
    s = _BubbleState(bubble_id="b6", role="assistant")
    s.workflow = WorkflowProposal(
        node_count=1, edge_count=0, preview_types=["X"], state="applied",
    )
    html = render_bubble_html(s, COLORS)
    assert "Applied" in html
    assert "apply://b6" not in html
    assert "discard://b6" not in html


def test_discarded_workflow_proposal_shows_only_discarded_marker():
    s = _BubbleState(bubble_id="b8", role="assistant")
    s.workflow = WorkflowProposal(
        node_count=1, edge_count=0, preview_types=["X"], state="discarded",
    )
    html = render_bubble_html(s, COLORS)
    assert "Discarded" in html
    assert "apply://b8" not in html
    assert "discard://b8" not in html


def test_parse_anchor_chip():
    assert parse_anchor("chip://b3/c1") == ("chip", "b3", "c1")


def test_parse_anchor_apply_has_no_chip():
    assert parse_anchor("apply://b3") == ("apply", "b3", None)


def test_parse_anchor_discard_has_no_chip():
    assert parse_anchor("discard://b3") == ("discard", "b3", None)


def test_parse_anchor_rejects_unknown_scheme():
    import pytest
    with pytest.raises(ValueError):
        parse_anchor("http://evil.com")


def test_parse_anchor_rejects_trailing_path_segments():
    import pytest
    with pytest.raises(ValueError):
        parse_anchor("chip://b/c/extra")


def test_parse_anchor_rejects_bare_string():
    import pytest
    with pytest.raises(ValueError):
        parse_anchor("foo")


def test_error_bubble_uses_error_colors_and_no_tail_for_text_only_role():
    s = _BubbleState(bubble_id="b7", role="error", text="network down")
    html = render_bubble_html(s, COLORS)
    assert "network down" in html
    assert COLORS["err_bg"] in html
