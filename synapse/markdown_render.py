"""Markdown -> HTML renderer for chat bubbles.

Uses python-markdown with fenced_code, tables, codehilite (Pygments inline
styles). Output is safe for QTextBrowser which handles a subset of HTML.

**Thread-safety**: the shared ``_RENDERER`` instance uses ``.reset()`` between
calls to avoid accumulating state; this is NOT thread-safe. All calls must
come from the Qt main thread. Phase 2's ``ChatStreamWorker`` must marshal
rendering to the main thread OR construct a fresh ``markdown.Markdown`` per
call before invoking from a worker.
"""
from __future__ import annotations

import re

import markdown as _md
from markdown.postprocessors import Postprocessor

# Tags that should never appear in chat bubble output.
_DANGEROUS = re.compile(
    r'<(script|style|iframe|object|embed|form|input|button)[\s\S]*?</\1>',
    re.IGNORECASE,
)
_DANGEROUS_OPEN = re.compile(
    r'<(script|style|iframe|object|embed|form|input|button)(\s[^>]*)?>',
    re.IGNORECASE,
)


class _SanitizePostprocessor(Postprocessor):
    """Strip dangerous tags that python-markdown passes through by default."""

    def run(self, text: str) -> str:
        text = _DANGEROUS.sub('', text)
        text = _DANGEROUS_OPEN.sub('', text)
        return text


# Reusable Markdown instance with the extensions we need. codehilite with
# inline_css=True embeds Pygments styles directly into generated <span>s, so
# no external stylesheet is needed — important for QTextBrowser.
_RENDERER = _md.Markdown(
    extensions=["fenced_code", "tables", "codehilite"],
    extension_configs={
        "codehilite": {
            "guess_lang": False,
            "noclasses": True,       # inline styles via Pygments
            "pygments_style": "default",
        },
    },
    output_format="html",
)

# Register sanitizer at priority 5 (runs after RawHtmlPostprocessor at 30,
# AndSubstitutePostprocessor at 20 — so this is last).
_RENDERER.postprocessors.register(_SanitizePostprocessor(_RENDERER), 'sanitize', 5)


def render_markdown(text: str | None) -> str:
    """Render *text* from markdown to an HTML snippet. Empty/None -> ''."""
    if not text:
        return ""
    # Reset so internal state doesn't accumulate across calls.
    _RENDERER.reset()
    return _RENDERER.convert(text)
