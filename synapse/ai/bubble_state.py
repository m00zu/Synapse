"""Pure-Python bubble state dataclasses + HTML renderer + anchor-URL parser.

No Qt imports — this module can be imported and tested without a display.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Literal, Optional

from synapse.markdown_render import render_markdown


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolChip:
    chip_id: str                  # stable id for anchor routing, e.g. "c3"
    name: str                     # tool name, e.g. "inspect_canvas"
    input_preview: str            # short one-line JSON preview for chip label
    status: Literal["running", "ok", "error"] = "running"
    result_summary: str = ""      # short one-line result for chip label
    full_input: dict = field(default_factory=dict)    # rendered when expanded
    full_result: Optional[dict] = None                # rendered when expanded


@dataclass
class WorkflowProposal:
    node_count: int
    edge_count: int
    preview_types: list[str]
    state: Literal["pending", "applied", "discarded"] = "pending"


@dataclass
class _BubbleState:
    bubble_id: str                        # e.g. "b7"
    role: Literal["user", "assistant", "system", "error"]
    text: str = ""                        # raw markdown for assistant; plain for others
    chips: list[ToolChip] = field(default_factory=list)
    expanded_chips: set[str] = field(default_factory=set)
    workflow: Optional[WorkflowProposal] = None
    streaming: bool = False               # True while tokens still arriving


# ---------------------------------------------------------------------------
# Status glyphs
# ---------------------------------------------------------------------------

_STATUS_GLYPH: dict[str, str] = {
    "running": "⋯",
    "ok": "✓",
    "error": "⚠",
}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_bubble_html(state: _BubbleState, colors: dict) -> str:
    """Return an HTML fragment for the given bubble state.

    Parameters
    ----------
    state:
        The current state of the bubble to render.
    colors:
        Dict of CSS colour values.  Required keys:
        ``user_bg``, ``user_fg``, ``ai_bg``, ``ai_fg``, ``ai_label``,
        ``ai_border``, ``err_bg``, ``err_fg``, ``sys_fg``.
    """
    c = colors
    role = state.role

    # Pre-escape plain text used in user/system/error roles
    text_escaped = html.escape(state.text)
    text_html = text_escaped.replace("\n", "<br>")

    if role == "user":
        return (
            f"<div style='margin:0; padding:0; line-height:6px;'>&nbsp;</div>"
            f"<table width='100%' cellpadding='0' cellspacing='0'>"
            f"<tr><td width='15%'></td>"
            f"<td style='background:{c['user_bg']}; color:{c['user_fg']}; "
            f"padding:8px 14px; border-radius:14px 14px 4px 14px;'>"
            f"<span style='font-size:13px; line-height:1.5;'>"
            f"{text_html}</span></td></tr>"
            f"<tr><td></td>"
            f"<td align='right' style='padding:0; line-height:0; font-size:0;'>"
            f"<span style='color:{c['user_bg']}; font-size:14px; "
            f"line-height:0;'>&#9698;</span></td></tr>"
            f"</table>"
        )

    elif role == "error":
        return (
            f"<div style='margin:0; padding:0; line-height:6px;'>&nbsp;</div>"
            f"<table width='100%' cellpadding='0' cellspacing='0'>"
            f"<tr><td style='background:{c['err_bg']}; color:{c['err_fg']}; "
            f"padding:8px 14px; border-radius:12px;'>"
            f"<span style='font-size:10px; font-weight:600;'>Error</span><br>"
            f"<span style='font-size:13px;'>{text_html}</span></td>"
            f"<td width='15%'></td></tr>"
            f"<tr><td style='padding:0; line-height:0; font-size:0;'>"
            f"<span style='color:{c['err_bg']}; font-size:14px; "
            f"line-height:0;'>&#9699;</span></td>"
            f"<td></td></tr>"
            f"</table>"
        )

    elif role != "assistant":
        # system (and any unknown role treated as system)
        return (
            f"<div style='margin:0; padding:0; line-height:6px;'>&nbsp;</div>"
            f"<table width='100%' cellpadding='0' cellspacing='0'><tr>"
            f"<td width='10%'></td>"
            f"<td align='center' style='color:{c['sys_fg']}; "
            f"font-size:11px; font-style:italic; padding:4px 0;'>"
            f"{text_html}</td>"
            f"<td width='10%'></td></tr></table>"
        )

    # ------------------------------------------------------------------
    # Assistant bubble — new layout
    # ------------------------------------------------------------------
    parts: list[str] = []

    # Header: "AI" label + optional streaming cursor
    cursor = " <span style='color:#58a6ff;'>●</span>" if state.streaming else ""
    parts.append(
        f"<span style='color:{c['ai_label']}; font-size:10px; "
        f"font-weight:600;'>AI</span>{cursor}<br>"
    )

    # Chips row
    if state.chips:
        bid_attr = html.escape(state.bubble_id, quote=True)
        chip_parts: list[str] = []
        for chip in state.chips:
            glyph = _STATUS_GLYPH.get(chip.status, "?")
            label_name = html.escape(chip.name)
            label_preview = html.escape(chip.input_preview)
            label_result = html.escape(chip.result_summary) if chip.result_summary else "…"
            pill_style = (
                "display:inline-block; padding:2px 8px; margin:2px 4px 2px 0; "
                "border-radius:10px; font-size:11px; text-decoration:none; "
                "background:#21262d; color:#c9d1d9; border:1px solid #30363d;"
            )
            cid_attr = html.escape(chip.chip_id, quote=True)
            url = f"chip://{bid_attr}/{cid_attr}"
            chip_parts.append(
                f"<a href='{url}' style='{pill_style}'>"
                f"&#128295; {label_name} &rarr; {glyph} {label_result}"
                f"</a>"
            )
        parts.append(
            f"<div style='margin:4px 0;'>{''.join(chip_parts)}</div>"
        )

        # Expanded chip detail blocks
        for chip in state.chips:
            if chip.chip_id not in state.expanded_chips:
                continue
            input_json = html.escape(json.dumps(chip.full_input, indent=2))
            result_json = ""
            if chip.full_result is not None:
                result_json = html.escape(json.dumps(chip.full_result, indent=2))
            pre_style = (
                "background:#0d1117; color:#c9d1d9; font-size:11px; "
                "padding:6px 10px; border-radius:6px; overflow-x:auto; "
                "border:1px solid #30363d; margin:4px 0;"
            )
            block = f"<pre style='{pre_style}'><b>Input:</b>\n{input_json}"
            if result_json:
                block += f"\n\n<b>Result:</b>\n{result_json}"
            block += "</pre>"
            parts.append(block)

    # Markdown body
    body_html = render_markdown(state.text)
    if not body_html and state.text:
        body_html = html.escape(state.text).replace("\n", "<br>")
    if body_html:
        parts.append(
            f"<div style='font-size:13px; line-height:1.5;'>{body_html}</div>"
        )

    # Workflow proposal block
    if state.workflow is not None:
        wf = state.workflow
        bid = html.escape(state.bubble_id, quote=True)
        if wf.state == "pending":
            types_str = html.escape(", ".join(wf.preview_types))
            summary = (
                f"Proposed: {wf.node_count} nodes, {wf.edge_count} edges"
                f" &mdash; {types_str}"
            )
            apply_style = (
                "display:inline-block; padding:4px 12px; margin:4px 6px 4px 0; "
                "border-radius:6px; font-size:12px; text-decoration:none; "
                "background:#238636; color:#fff; border:1px solid #2ea043;"
            )
            discard_style = (
                "display:inline-block; padding:4px 12px; margin:4px 0; "
                "border-radius:6px; font-size:12px; text-decoration:none; "
                "background:#21262d; color:#d0d7de; border:1px solid #30363d;"
            )
            parts.append(
                f"<div style='margin:6px 0; font-size:12px; color:{c['ai_fg']};'>"
                f"{summary}</div>"
                f"<div style='margin:4px 0;'>"
                f"<a href='apply://{bid}' style='{apply_style}'>Apply</a>"
                f"<a href='discard://{bid}' style='{discard_style}'>Discard</a>"
                f"</div>"
            )
        elif wf.state == "applied":
            parts.append(
                f"<div style='margin:6px 0; font-size:12px; "
                f"color:#8b949e;'>&#10003; Applied</div>"
            )
        else:  # discarded
            parts.append(
                f"<div style='margin:6px 0; font-size:12px; "
                f"color:#8b949e;'>Discarded</div>"
            )

    inner = "\n".join(parts)
    return (
        f"<div style='margin:0; padding:0; line-height:6px;'>&nbsp;</div>"
        f"<table width='100%' cellpadding='0' cellspacing='0'>"
        f"<tr><td style='background:{c['ai_bg']}; color:{c['ai_fg']}; "
        f"padding:8px 14px; border-radius:4px 14px 14px 14px; "
        f"border:1px solid {c['ai_border']};'>"
        f"{inner}</td>"
        f"<td width='15%'></td></tr>"
        f"<tr><td style='padding:0; line-height:0; font-size:0;'>"
        f"<span style='color:{c['ai_bg']}; font-size:14px; "
        f"line-height:0;'>&#9699;</span></td>"
        f"<td></td></tr>"
        f"</table>"
    )


# ---------------------------------------------------------------------------
# Anchor URL parsing
# ---------------------------------------------------------------------------

def parse_anchor(url: str) -> tuple[str, str, Optional[str]]:
    """Parse a custom-scheme anchor URL used in chat bubbles.

    Returns
    -------
    (action, bubble_id, chip_id)
        ``action`` is one of ``"chip"``, ``"apply"``, ``"discard"``.
        ``chip_id`` is ``None`` for ``apply``/``discard`` schemes.

    Raises
    ------
    ValueError
        For any unknown scheme or malformed URL.
    """
    _KNOWN = {"chip", "apply", "discard"}

    if "://" not in url:
        raise ValueError(f"Not a recognised anchor URL: {url!r}")

    scheme, rest = url.split("://", 1)
    if scheme not in _KNOWN:
        raise ValueError(
            f"Unknown anchor scheme {scheme!r}. Expected one of: {sorted(_KNOWN)}"
        )

    if scheme == "chip":
        # chip://<bubble_id>/<chip_id>  — exactly two non-empty segments
        segments = rest.split("/")
        if len(segments) != 2:
            raise ValueError(
                f"Malformed chip URL (expected exactly 2 path segments): {url!r}"
            )
        bubble_id, chip_id = segments
        if not bubble_id or not chip_id:
            raise ValueError(f"Malformed chip URL (empty segment): {url!r}")
        return ("chip", bubble_id, chip_id)

    # apply://<bubble_id>  or  discard://<bubble_id>
    bubble_id = rest.rstrip("/")
    if not bubble_id:
        raise ValueError(f"Malformed {scheme} URL (empty bubble_id): {url!r}")
    return (scheme, bubble_id, None)
