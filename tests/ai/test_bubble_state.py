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


def test_assistant_code_block_uses_dark_theme_background():
    s = _BubbleState(
        bubble_id="bc", role="assistant",
        text="Here:\n```\nFoo -> Bar\n```\n",
    )
    colors = {**COLORS, "code_bg": "#0d1117", "code_fg": "#c9d1d9"}
    html_out = render_bubble_html(s, colors)
    # Pre tag carries our injected dark background, not Pygments' white.
    assert "<pre style=" in html_out
    assert "background:#0d1117" in html_out
    assert "color:#c9d1d9" in html_out


def test_assistant_inline_code_also_styled():
    s = _BubbleState(
        bubble_id="bi", role="assistant",
        text="Use `render_markdown` for text.",
    )
    colors = {**COLORS, "code_bg": "#0d1117", "code_fg": "#c9d1d9"}
    html_out = render_bubble_html(s, colors)
    assert "<code style=" in html_out
    assert "background:#0d1117" in html_out


def test_error_bubble_uses_error_colors_and_no_tail_for_text_only_role():
    s = _BubbleState(bubble_id="b7", role="error", text="network down")
    html = render_bubble_html(s, COLORS)
    assert "network down" in html
    assert COLORS["err_bg"] in html


# ---------------------------------------------------------------------------
# _BubbleLog tests (using a pure-Python fake browser — no Qt required)
# ---------------------------------------------------------------------------

class _FakeBrowser:
    def __init__(self):
        self.segments: list[str] = []
        self._pinned = True

    def append_html(self, html: str) -> None:
        self.segments.append(html)

    def clear(self) -> None:
        self.segments.clear()

    def snapshot_position(self) -> int:
        return len(self.segments)

    def rewrite_from(self, pos: int) -> None:
        del self.segments[pos:]

    def scroll_to_bottom(self) -> None:
        pass

    def is_pinned_to_bottom(self) -> bool:
        return self._pinned


def test_bubble_log_add_appends_one_segment_per_call():
    from synapse.llm_assistant import _BubbleLog
    fake = _FakeBrowser()
    log = _BubbleLog(fake, colors_getter=lambda: COLORS)
    log.add(_BubbleState(bubble_id="", role="user", text="hi"))
    log.add(_BubbleState(bubble_id="", role="assistant", text="hello"))
    assert len(fake.segments) == 2


def test_bubble_log_update_rewrites_tail_from_bubble_position():
    from synapse.llm_assistant import _BubbleLog
    fake = _FakeBrowser()
    log = _BubbleLog(fake, colors_getter=lambda: COLORS)
    bid1 = log.add(_BubbleState(bubble_id="", role="assistant", text="first"))
    bid2 = log.add(_BubbleState(bubble_id="", role="assistant", text="second"))
    log.update(bid1, lambda s: setattr(s, "text", "first (edited)"))
    # After update, segment 0 reflects the edit, segment 1 is still the second bubble (re-rendered).
    assert "first (edited)" in fake.segments[0]
    assert "second" in fake.segments[1]
    assert len(fake.segments) == 2


def test_bubble_log_clear_empties_state_and_browser():
    from synapse.llm_assistant import _BubbleLog
    fake = _FakeBrowser()
    log = _BubbleLog(fake, colors_getter=lambda: COLORS)
    log.add(_BubbleState(bubble_id="", role="user", text="x"))
    log.clear()
    assert fake.segments == []
