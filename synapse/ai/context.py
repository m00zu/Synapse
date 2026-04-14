"""Per-turn context helpers for the AI chat orchestrator.

Phase 2a implements:
  * graph_summary(graph) -> str   — cheap one-line canvas description
  * estimate_tokens(text) -> int  — conservative char/4 heuristic
  * HistoryRoller                 — window + tool-result truncation (no LLM yet)
"""
from __future__ import annotations

__all__ = ["graph_summary", "estimate_tokens", "HistoryRoller"]


def _node_type(node) -> str:
    """Return the node's class name. Works for both FakeNode (has type_name
    attribute) and real NodeGraphQt nodes (use Python class name)."""
    return getattr(node, "type_name", None) or type(node).__name__


def _chain_types_if_linear(nodes: list) -> list[str] | None:
    """Return the type chain if the graph is a single linear path, else None."""
    if not nodes:
        return []
    heads = []
    for n in nodes:
        in_connected = any(
            p.connected_ports() for p in n.inputs().values()
        )
        if not in_connected:
            heads.append(n)
    if len(heads) != 1:
        return None
    visited: list = []
    cur = heads[0]
    while cur is not None:
        if cur in visited:
            return None  # cycle
        visited.append(cur)
        next_node = None
        for port in cur.outputs().values():
            peers = port.connected_ports()
            if peers:
                if next_node is not None:
                    return None  # branching
                next_node = peers[0].node()
        cur = next_node
    if len(visited) != len(nodes):
        return None  # disconnected sub-graph
    return [_node_type(n) for n in visited]


def graph_summary(graph) -> str:
    """Compact one-line description of the current canvas."""
    nodes = list(graph.all_nodes())
    if not nodes:
        return "Canvas: empty."
    n = len(nodes)

    chain = _chain_types_if_linear(nodes)
    if chain is not None and 1 <= n <= 8:
        if n == 1:
            return f"Canvas: 1 node ({chain[0]})."
        return f"Canvas: {n} nodes — " + " → ".join(chain) + "."

    from collections import Counter
    type_counts = Counter(_node_type(node) for node in nodes)
    parts = [
        (f"{t}×{c}" if c > 1 else t)
        for t, c in type_counts.most_common(5)
    ]
    tail = ", ..." if len(type_counts) > 5 else ""
    connected = _count_connected_components(nodes)
    suffix = f"; {connected} disconnected branches" if connected > 1 else ""
    return f"Canvas: {n} nodes ({', '.join(parts)}{tail}){suffix}."


def _count_connected_components(nodes: list) -> int:
    """Count weakly-connected components (ignoring edge direction)."""
    seen: set = set()
    count = 0
    for start in nodes:
        if id(start) in seen:
            continue
        count += 1
        stack = [start]
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            for port in list(cur.inputs().values()) + list(cur.outputs().values()):
                for peer in port.connected_ports():
                    stack.append(peer.node())
    return count


def estimate_tokens(text: str | None) -> int:
    """Conservative char/4 approximation — good enough for budget checks."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class HistoryRoller:
    """Window + tool-result truncation. No LLM-based summarization in 2a."""

    KEEP_TURNS = 8
    TOOL_RESULT_TRUNCATE_AFTER_TURNS = 2

    def __init__(self, keep_turns: int = KEEP_TURNS):
        self.keep_turns = keep_turns

    def roll(self, messages: list[dict]) -> list[dict]:
        """Trim to the last ``keep_turns`` user+assistant pairs, and replace
        old tool results with a short stub.
        """
        turn_starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if len(turn_starts) <= self.keep_turns:
            out = list(messages)
        else:
            first_kept = turn_starts[-self.keep_turns]
            dropped_count = first_kept
            out = [{
                "role": "system",
                "content": (
                    f"[{dropped_count} earlier messages trimmed to conserve context]"
                ),
            }] + messages[first_kept:]

        # Truncate tool results that are at least TOOL_RESULT_TRUNCATE_AFTER_TURNS
        # user turns in the past. Walking backwards, we increment the counter
        # when we cross a user message. A tool whose counter is already >= the
        # threshold lived that many user turns ago — it's stale enough to drop.
        user_turns_so_far = 0
        out_reversed = list(reversed(out))
        for idx, m in enumerate(out_reversed):
            if m.get("role") == "user":
                user_turns_so_far += 1
            if (
                m.get("role") == "tool"
                and user_turns_so_far >= self.TOOL_RESULT_TRUNCATE_AFTER_TURNS
            ):
                original = m.get("content", "")
                summary = f"[truncated tool result, orig {estimate_tokens(str(original))} tokens]"
                out_reversed[idx] = {**m, "content": summary}
        return list(reversed(out_reversed))
