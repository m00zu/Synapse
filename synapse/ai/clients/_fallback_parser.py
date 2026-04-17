"""Streaming parser for the prompt-based tool-call protocol.

Watches the model's streamed text for a ``<tool_call>{...}</tool_call>`` span
that contains a JSON object. Emits ``("text", chunk)`` for everything outside
the markers, ``("tool_call", dict)`` when a complete span is parsed, or
``("error", message)`` when JSON decoding fails.

Once a tool_call is emitted, subsequent input is discarded — the protocol says
the model stops after the marker. The orchestrator will dispatch the tool,
send the result back, and start a fresh parser for the next turn.
"""
from __future__ import annotations

import json
from typing import List, Tuple

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"


class StreamedToolCallParser:
    """Incrementally consume streamed text and split text / tool_call events."""

    def __init__(self) -> None:
        self._buf = ""
        self._in_marker = False
        self._done = False

    def feed(self, chunk: str) -> List[Tuple[str, object]]:
        """Consume *chunk*. Returns a list of (kind, payload) tuples where
        kind is "text" | "tool_call" | "error"."""
        if self._done:
            return []
        out: List[Tuple[str, object]] = []
        self._buf += chunk

        while self._buf and not self._done:
            if not self._in_marker:
                idx = self._buf.find(_OPEN)
                if idx == -1:
                    # No opener in buffer. Flush text except a trailing
                    # partial-opener prefix (might be split across chunks).
                    prefix_hold = _longest_possible_opener_prefix(self._buf)
                    keep_start = len(self._buf) - prefix_hold
                    emit = self._buf[:keep_start]
                    if emit:
                        out.append(("text", emit))
                    self._buf = self._buf[keep_start:]
                    break  # need more input
                else:
                    if idx > 0:
                        out.append(("text", self._buf[:idx]))
                    self._buf = self._buf[idx + len(_OPEN):]
                    self._in_marker = True
            else:
                idx = self._buf.find(_CLOSE)
                if idx == -1:
                    break  # waiting for closer
                body = self._buf[:idx]
                self._buf = self._buf[idx + len(_CLOSE):]
                self._in_marker = False
                self._done = True  # discard everything after the tool_call
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError as e:
                    out.append(("error", f"tool_call JSON decode failed: {e}"))
                else:
                    out.append(("tool_call", parsed))
        return out

    def finish(self) -> List[Tuple[str, object]]:
        """Called at end-of-stream. Returns any remaining buffered events.
        Currently returns an empty list — partial tool_calls are abandoned."""
        return []


def _longest_possible_opener_prefix(buf: str) -> int:
    """Return the length of the longest suffix of *buf* that could be the
    start of ``<tool_call>``."""
    for n in range(len(_OPEN) - 1, 0, -1):
        if buf.endswith(_OPEN[:n]):
            return n
    return 0
