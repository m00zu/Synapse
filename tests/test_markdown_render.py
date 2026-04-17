from synapse.markdown_render import render_markdown


def test_renders_plain_paragraph():
    html = render_markdown("Hello **world**.")
    assert "<strong>world</strong>" in html


def test_renders_heading():
    html = render_markdown("# Title")
    assert "<h1" in html
    assert "Title" in html


def test_renders_fenced_code_with_language():
    src = "```python\nprint('hi')\n```"
    html = render_markdown(src)
    # Pygments emits inline style spans for syntax highlighting
    assert "style=" in html
    assert "print" in html


def test_renders_table():
    md = "| a | b |\n| - | - |\n| 1 | 2 |\n"
    html = render_markdown(md)
    assert "<table" in html
    assert "<td>1</td>" in html


def test_escapes_raw_html():
    # XSS guard: raw <script> in source markdown should not produce a live
    # script tag. Default python-markdown escapes unsafe HTML.
    html = render_markdown("<script>alert(1)</script>")
    assert "<script>" not in html.lower()


def test_empty_input_returns_empty_string():
    assert render_markdown("") == ""
    assert render_markdown(None) == ""  # tolerant of None
