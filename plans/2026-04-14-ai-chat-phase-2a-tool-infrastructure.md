# AI Chat Phase 2a — Tool Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the server-side tool machinery the Phase 2b orchestrator will call — tool schemas, a dispatcher, six tool handlers, a base system prompt, a canvas-summary helper, and a history roller. Everything is testable against an in-memory fake graph; no LLM is invoked and no UI is touched.

**Architecture:**
- `synapse/ai/prompts.py` hosts two string constants — `BASE_SYSTEM_PROMPT` (role + markdown + tool-use rules) and `WRITE_PYTHON_SCRIPT_SUBPROMPT` (specialized for the Python-code sub-LLM call).
- `synapse/ai/context.py` owns `graph_summary(graph) -> str`, `estimate_tokens(text) -> int`, and a `HistoryRoller` class that trims conversation history + old tool results (no LLM-based summarization — that is deferred to Phase 2b, where we have an `LLMClient` in scope).
- `synapse/ai/tools.py` owns the `TOOLS` JSON schema list and a `ToolDispatcher` class. Tool handlers are separate callables registered with the dispatcher so each handler is testable in isolation.
- Feature flag `USE_ORCHESTRATOR` is read from `~/.synapse_llm_config.json` (default False). Phase 2b will gate UI routing on it; Phase 2a only introduces the flag.
- Phase 2a does **not** add an orchestrator class, does **not** change any LLMClient, and does **not** modify `AIChatPanel`. Tests exercise `ToolDispatcher` directly.

**Tech Stack:** Python 3.13+, existing helpers from `synapse/llm_assistant.py` (`serialize_graph`, `build_detailed_cards`, `TwoPassLLMWorker`, `WorkflowLoader`, `build_condensed_catalog`, `build_system_prompt`), NodeGraphQt duck-typed through a minimal graph interface.

---

## File Structure

**New files:**

```
synapse/
  ai/
    prompts.py           (BASE_SYSTEM_PROMPT + WRITE_PYTHON_SCRIPT_SUBPROMPT)
    context.py           (graph_summary, estimate_tokens, HistoryRoller)
    tools.py             (TOOLS schema list + ToolDispatcher)
    tool_handlers/
      __init__.py        (re-exports all handlers)
      inspect_canvas.py
      explain_node.py
      read_node_output.py
      generate_workflow.py
      modify_workflow.py
      write_python_script.py
    feature_flags.py     (get_use_orchestrator() + feature-flag loader)
tests/
  ai/
    test_prompts.py
    test_context.py
    test_tools_schema.py
    test_tool_dispatcher.py
    test_handler_inspect_canvas.py
    test_handler_explain_node.py
    test_handler_read_node_output.py
    test_handler_generate_workflow.py
    test_handler_modify_workflow.py
    test_handler_write_python_script.py
  ai/
    fakes.py             (FakeGraph / FakeNode / FakePort for handler tests)
```

**Modified files:**

```
synapse/ai/__init__.py   (add convenience re-exports for Phase 2a modules)
synapse/ai/clients/__init__.py  (no change — left from Phase 1)
```

No modification to `synapse/llm_assistant.py` in Phase 2a.

---

## Task 1: Scaffolding — empty module stubs + fake-graph helper

**Files:**
- Create: `synapse/ai/prompts.py` (with `__all__ = []`)
- Create: `synapse/ai/context.py` (with `__all__ = []`)
- Create: `synapse/ai/tools.py` (with `__all__ = []`)
- Create: `synapse/ai/feature_flags.py`
- Create: `synapse/ai/tool_handlers/__init__.py` (empty)
- Create: `tests/ai/fakes.py` (minimal `FakeNode` / `FakeGraph`)
- Create: `tests/ai/test_fakes.py`

- [ ] **Step 1: Create empty module stubs**

Create each of `synapse/ai/prompts.py`, `synapse/ai/context.py`, `synapse/ai/tools.py`, `synapse/ai/tool_handlers/__init__.py` with exactly:

```python
"""<one-line description — filled in by later tasks>"""
__all__: list[str] = []
```

- [ ] **Step 2: Create feature-flag loader**

Write `synapse/ai/feature_flags.py`:

```python
"""Read-only accessors for Synapse AI feature flags.

Flags live in ``~/.synapse_llm_config.json`` under the ``ai`` key, e.g.

    {
      "ai": {
        "use_orchestrator": false
      }
    }

Missing/unreadable config → defaults (all flags False).
"""
from __future__ import annotations

import json
from pathlib import Path

_CONFIG_PATH = Path.home() / ".synapse_llm_config.json"


def _load() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text())
    except Exception:
        return {}


def get_use_orchestrator() -> bool:
    """True when the ChatOrchestrator path is enabled (Phase 2b)."""
    return bool(_load().get("ai", {}).get("use_orchestrator", False))
```

- [ ] **Step 3: Write `tests/ai/fakes.py`**

A minimal graph/node/port fake that subsequent handler tests can reuse. It mirrors the duck-types NodeGraphQt exposes to the existing `serialize_graph()` function (`all_nodes()`, `node.id`, `node.model.custom_properties`, `node.inputs()`, `node.outputs()`, `port.connected_ports()`, `port.name()`, `port.node()`).

```python
"""Lightweight fakes for handler unit tests. No Qt / NodeGraphQt required."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakePort:
    _name: str
    parent: "FakeNode"
    peers: list["FakePort"] = field(default_factory=list)

    def name(self) -> str:
        return self._name

    def node(self) -> "FakeNode":
        return self.parent

    def connected_ports(self) -> list["FakePort"]:
        return list(self.peers)

    def connect_to(self, other: "FakePort") -> None:
        if other not in self.peers:
            self.peers.append(other)
            other.peers.append(self)


@dataclass
class _Model:
    custom_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeNode:
    id: str
    type_name: str
    _inputs: dict[str, FakePort] = field(default_factory=dict)
    _outputs: dict[str, FakePort] = field(default_factory=dict)
    model: _Model = field(default_factory=_Model)
    output_values: dict[str, Any] = field(default_factory=dict)
    _last_error: str | None = None

    def __init__(self, node_id: str, type_name: str, props: dict | None = None):
        self.id = node_id
        self.type_name = type_name
        self._inputs = {}
        self._outputs = {}
        self.model = _Model(custom_properties=dict(props or {}))
        self.output_values = {}
        self._last_error = None

    def add_input(self, name: str) -> FakePort:
        p = FakePort(_name=name, parent=self)
        self._inputs[name] = p
        return p

    def add_output(self, name: str) -> FakePort:
        p = FakePort(_name=name, parent=self)
        self._outputs[name] = p
        return p

    def inputs(self) -> dict[str, FakePort]:
        return self._inputs

    def outputs(self) -> dict[str, FakePort]:
        return self._outputs

    def get_input(self, name: str) -> FakePort | None:
        return self._inputs.get(name)

    def get_output(self, name: str) -> FakePort | None:
        return self._outputs.get(name)

    def name(self) -> str:
        return self.model.custom_properties.get("name", self.id)

    def set_property(self, key: str, value: Any, push_undo: bool = False) -> None:
        self.model.custom_properties[key] = value


class FakeGraph:
    """Minimal stand-in for NodeGraphQt's NodeGraph used by handler tests."""

    def __init__(self) -> None:
        self._nodes: list[FakeNode] = []

    def add_node(self, node: FakeNode) -> FakeNode:
        self._nodes.append(node)
        return node

    def remove_node(self, node: FakeNode) -> None:
        self._nodes = [n for n in self._nodes if n.id != node.id]

    def all_nodes(self) -> list[FakeNode]:
        return list(self._nodes)

    def get_node_by_id(self, node_id: str) -> FakeNode | None:
        for n in self._nodes:
            if n.id == node_id:
                return n
        return None
```

- [ ] **Step 4: Write `tests/ai/test_fakes.py`**

Sanity tests for the fakes themselves so later handler tests can rely on them:

```python
from tests.ai.fakes import FakeGraph, FakeNode


def test_node_basic_identity():
    n = FakeNode("n1", "CSVLoader", {"path": "x.csv"})
    assert n.id == "n1"
    assert n.type_name == "CSVLoader"
    assert n.model.custom_properties["path"] == "x.csv"


def test_node_ports_and_connection():
    a = FakeNode("a", "A"); b = FakeNode("b", "B")
    out = a.add_output("out_1")
    in_ = b.add_input("in_1")
    out.connect_to(in_)
    assert b.inputs()["in_1"].connected_ports()[0].node().id == "a"
    assert a.outputs()["out_1"].connected_ports()[0].node().id == "b"


def test_graph_add_remove_lookup():
    g = FakeGraph()
    n = FakeNode("n1", "X")
    g.add_node(n)
    assert g.get_node_by_id("n1") is n
    assert len(g.all_nodes()) == 1
    g.remove_node(n)
    assert g.get_node_by_id("n1") is None


def test_set_property_updates_custom_properties():
    n = FakeNode("n", "Script")
    n.set_property("script_code", "out_1 = in_1")
    assert n.model.custom_properties["script_code"] == "out_1 = in_1"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/ai/test_fakes.py -v`
Expected: 4 pass.

Also confirm the full suite still passes: `pytest tests/ -q`
Expected: 37 + 4 = 41 pass.

- [ ] **Step 6: Commit**

```bash
git add synapse/ai/prompts.py synapse/ai/context.py synapse/ai/tools.py synapse/ai/feature_flags.py synapse/ai/tool_handlers/__init__.py tests/ai/fakes.py tests/ai/test_fakes.py
git commit -m "chore(ai): scaffold Phase 2a tool-infrastructure modules + test fakes"
```

---

## Task 2: `prompts.py::BASE_SYSTEM_PROMPT`

**Files:**
- Modify: `synapse/ai/prompts.py`
- Create: `tests/ai/test_prompts.py`

The base prompt is a multi-line string that Phase 2b's orchestrator will prepend to every turn. Required contents per the spec:
- Role: *You are a Synapse workflow assistant…*
- Tools: *prefer chatting and asking clarifying questions; use tools only when useful.*
- Rule: *Never dump raw JSON to the user — use tools instead.*
- Markdown guidance: *Reply in GitHub-flavoured markdown. Use fenced code blocks for code, tables for comparisons.*

- [ ] **Step 1: Write failing tests in `tests/ai/test_prompts.py`**

```python
from synapse.ai.prompts import BASE_SYSTEM_PROMPT, WRITE_PYTHON_SCRIPT_SUBPROMPT


def test_base_prompt_mentions_synapse_and_markdown():
    assert "Synapse" in BASE_SYSTEM_PROMPT
    assert "markdown" in BASE_SYSTEM_PROMPT.lower()


def test_base_prompt_forbids_raw_json_dumps():
    assert "raw JSON" in BASE_SYSTEM_PROMPT


def test_base_prompt_mentions_tools_and_clarifying_questions():
    low = BASE_SYSTEM_PROMPT.lower()
    assert "tool" in low
    assert "clarifying" in low or "clarify" in low


def test_write_python_script_subprompt_references_in_out_vars():
    # Must tell the sub-LLM the PythonScriptNode conventions
    assert "in_1" in WRITE_PYTHON_SCRIPT_SUBPROMPT
    assert "out_1" in WRITE_PYTHON_SCRIPT_SUBPROMPT


def test_write_python_script_subprompt_forbids_fences():
    # The handler parses the output directly — no markdown fences allowed.
    low = WRITE_PYTHON_SCRIPT_SUBPROMPT.lower()
    assert "fence" in low or "no markdown" in low or "```" in WRITE_PYTHON_SCRIPT_SUBPROMPT
```

- [ ] **Step 2: Run, expect ImportError / AttributeError**

Run: `pytest tests/ai/test_prompts.py -v`
Expected: tests fail because the constants don't exist yet.

- [ ] **Step 3: Implement `synapse/ai/prompts.py`**

```python
"""System and sub-prompts for the Synapse AI chat orchestrator."""
from __future__ import annotations

__all__ = ["BASE_SYSTEM_PROMPT", "WRITE_PYTHON_SCRIPT_SUBPROMPT"]


BASE_SYSTEM_PROMPT = """\
You are the AI assistant for Synapse — a scientific node-graph workflow editor.

Your job is to help the user analyse their data by:
  * explaining how Synapse nodes work,
  * building or editing workflows in the node graph,
  * writing Python code for PythonScriptNode when no dedicated node exists,
  * inspecting the current canvas and debugging it.

How to respond:
  * Reply in GitHub-flavoured markdown (headings, lists, fenced code blocks, tables).
  * Prefer short, direct answers. Ask clarifying questions when the request is ambiguous.
  * Use the tools provided to actually do things — do not paste raw JSON workflow
    templates into your reply. When you need to build or modify the workflow, call
    the appropriate tool. When you need to read the user's canvas, call inspect_canvas.
  * Never invent tool names, node class names, or node properties that the user has
    not mentioned. If you are unsure, call explain_node or inspect_canvas first.
"""


WRITE_PYTHON_SCRIPT_SUBPROMPT = """\
You are a focused code generator for Synapse's PythonScriptNode. Output exactly the
Python code body — no markdown fences, no prose, no comments unless required for
correctness.

Available in the execution environment:
  * Inputs: in_1, in_2, ... (DataFrame / ndarray / raw value; unconnected = None).
  * Outputs: out_1, out_2, ... — your code MUST assign to every declared output.
  * Pre-imported aliases: pd, np, scipy, skimage, cv2, PIL, plt.
  * Type wrappers: TableData, ImageData, MaskData, FigureData, StatData
    (use e.g. `out_1 = MaskData(payload=arr)` to force the output kind).
  * Progress: `set_progress(percent)` updates the node's progress bar (0-100).

Rules:
  * Do NOT wrap output in ``` fences or backticks.
  * Do NOT print except for diagnostics; assign results to out_N instead.
  * If the user asks for something impossible with the declared ports, stop and
    assign `out_1` to a pandas DataFrame containing a clear error message so the
    reviewer sees the failure downstream.
"""
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_prompts.py -v`
Expected: 5 pass.

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/prompts.py tests/ai/test_prompts.py
git commit -m "feat(ai/prompts): add base system prompt and python-script sub-prompt"
```

---

## Task 3: `context.py::graph_summary`

**Files:**
- Modify: `synapse/ai/context.py`
- Create: `tests/ai/test_context.py`

Input: a graph object with `all_nodes()`. Output: a single short line summarizing node count, types, and linear chain hints. Examples:

```
Canvas: empty.
Canvas: 1 node (CSVLoader).
Canvas: 4 nodes — CSVLoader → ParticleProps → SortTable → TopN.
Canvas: 6 nodes (CSVLoader×2, ParticleProps, ...); 2 disconnected.
```

For Phase 2a we keep this deterministic and cheap — no error-state detection (that comes later when we wire `BaseExecutionNode.mark_error()` flags through).

- [ ] **Step 1: Write tests in `tests/ai/test_context.py`**

```python
from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.context import graph_summary


def _chain(types: list[str]) -> FakeGraph:
    g = FakeGraph()
    nodes = [FakeNode(f"n{i+1}", t) for i, t in enumerate(types)]
    for n in nodes:
        g.add_node(n)
    for i in range(len(nodes) - 1):
        out = nodes[i].add_output("out_1")
        inn = nodes[i+1].add_input("in_1")
        out.connect_to(inn)
    return g


def test_graph_summary_empty():
    assert graph_summary(FakeGraph()) == "Canvas: empty."


def test_graph_summary_single_node():
    g = FakeGraph(); g.add_node(FakeNode("a", "CSVLoader"))
    s = graph_summary(g)
    assert s.startswith("Canvas: 1 node")
    assert "CSVLoader" in s


def test_graph_summary_linear_chain_shows_arrows():
    g = _chain(["CSVLoader", "ParticleProps", "SortTable", "TopN"])
    s = graph_summary(g)
    assert "4 nodes" in s
    assert "CSVLoader" in s and "TopN" in s
    # arrow notation between adjacent types
    assert "→" in s or "->" in s


def test_graph_summary_nonlinear_drops_arrow_notation():
    # Two disconnected sub-chains — no meaningful single chain to render.
    g = FakeGraph()
    a = FakeNode("a", "A"); b = FakeNode("b", "B")
    c = FakeNode("c", "C"); d = FakeNode("d", "D")
    for n in (a, b, c, d):
        g.add_node(n)
    a.add_output("o").connect_to(b.add_input("i"))
    c.add_output("o").connect_to(d.add_input("i"))
    s = graph_summary(g)
    assert "4 nodes" in s
    assert "disconnected" in s.lower() or "branches" in s.lower()


def test_graph_summary_under_200_chars():
    # Big graphs must still produce a compact summary (spec: 50-150 tokens).
    g = _chain(["Node"] * 50)
    assert len(graph_summary(g)) < 200
```

- [ ] **Step 2: Run — expect ImportError**

Run: `pytest tests/ai/test_context.py -v`

- [ ] **Step 3: Implement `graph_summary` in `synapse/ai/context.py`**

Replace the module contents with:

```python
"""Per-turn context helpers for the AI chat orchestrator.

Phase 2a implements:
  * graph_summary(graph) -> str   — cheap one-line canvas description
  * estimate_tokens(text) -> int  — conservative char/4 heuristic
  * HistoryRoller                 — window + tool-result truncation (no LLM yet)
"""
from __future__ import annotations

from typing import Iterable

__all__ = ["graph_summary", "estimate_tokens", "HistoryRoller"]


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
    return [n.type_name for n in visited]


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

    # Non-linear or large — type summary.
    from collections import Counter
    type_counts = Counter(node.type_name for node in nodes)
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
        """Return a new messages list trimmed to at most ``keep_turns`` user+assistant
        pairs, with tool results older than ``TOOL_RESULT_TRUNCATE_AFTER_TURNS``
        replaced by a short stub.
        """
        # Identify turn boundaries by user-role messages.
        turn_starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if len(turn_starts) <= self.keep_turns:
            out = list(messages)
        else:
            first_kept = turn_starts[-self.keep_turns]
            dropped_count = first_kept
            # Replace dropped prefix with a single synthetic system message.
            out = [{
                "role": "system",
                "content": (
                    f"[{dropped_count} earlier messages trimmed to conserve context]"
                ),
            }] + messages[first_kept:]

        # Truncate old tool results. "Old" = more than
        # TOOL_RESULT_TRUNCATE_AFTER_TURNS user turns ago.
        user_turns_so_far = 0
        # Walk backwards: each time we pass a user role, increment the counter.
        out_reversed = list(reversed(out))
        for idx, m in enumerate(out_reversed):
            if m.get("role") == "user":
                user_turns_so_far += 1
            if (
                m.get("role") == "tool"
                and user_turns_so_far > self.TOOL_RESULT_TRUNCATE_AFTER_TURNS
            ):
                original = m.get("content", "")
                summary = f"[truncated tool result, orig {estimate_tokens(str(original))} tokens]"
                out_reversed[idx] = {**m, "content": summary}
        return list(reversed(out_reversed))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_context.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/context.py tests/ai/test_context.py
git commit -m "feat(ai/context): add graph_summary helper"
```

---

## Task 4: `context.py::estimate_tokens` + `HistoryRoller` tests

`estimate_tokens` and `HistoryRoller` were already implemented in Task 3's full module write. This task only adds focused tests for them.

**Files:**
- Modify: `tests/ai/test_context.py`

- [ ] **Step 1: Append tests to `tests/ai/test_context.py`**

```python
# --- estimate_tokens ------------------------------------------------------

def test_estimate_tokens_empty():
    from synapse.ai.context import estimate_tokens
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_short_text():
    from synapse.ai.context import estimate_tokens
    # 12-char string → 12//4 = 3
    assert estimate_tokens("hello world!") == 3


def test_estimate_tokens_monotonic():
    from synapse.ai.context import estimate_tokens
    short = estimate_tokens("hi")
    long_ = estimate_tokens("hi " * 1000)
    assert long_ > short


# --- HistoryRoller --------------------------------------------------------

def _msgs(pairs: list[tuple[str, str]]) -> list[dict]:
    return [{"role": r, "content": c} for r, c in pairs]


def test_history_roller_keeps_short_history_unchanged():
    from synapse.ai.context import HistoryRoller
    roller = HistoryRoller(keep_turns=8)
    msgs = _msgs([
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "what's a ROIMask?"),
        ("assistant", "a node that defines a region of interest."),
    ])
    assert roller.roll(msgs) == msgs


def test_history_roller_trims_beyond_window():
    from synapse.ai.context import HistoryRoller
    roller = HistoryRoller(keep_turns=2)
    # 4 user turns → older 2 should be dropped.
    msgs = []
    for i in range(4):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    out = roller.roll(msgs)
    # First entry is the synthetic trim system message.
    assert out[0]["role"] == "system"
    assert "trimmed" in out[0]["content"].lower()
    # The last 2 user+assistant pairs survive intact.
    assert {"role": "user", "content": "q2"} in out
    assert {"role": "user", "content": "q3"} in out


def test_history_roller_truncates_old_tool_results():
    from synapse.ai.context import HistoryRoller
    roller = HistoryRoller(keep_turns=8)
    msgs = [
        {"role": "user", "content": "turn 1"},
        {"role": "tool", "content": "BIG RESULT " * 100},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "turn 3"},  # current turn
    ]
    out = roller.roll(msgs)
    tool_msg = next(m for m in out if m["role"] == "tool")
    assert "truncated" in tool_msg["content"].lower()
    assert len(tool_msg["content"]) < 100


def test_history_roller_keeps_recent_tool_results():
    from synapse.ai.context import HistoryRoller
    roller = HistoryRoller(keep_turns=8)
    msgs = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "turn 2"},
        {"role": "tool", "content": "RECENT RESULT " * 100},
        {"role": "assistant", "content": "ok"},
    ]
    out = roller.roll(msgs)
    tool_msg = next(m for m in out if m["role"] == "tool")
    assert "RECENT RESULT" in tool_msg["content"]
    assert "truncated" not in tool_msg["content"].lower()
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/ai/test_context.py -v`
Expected: 5 prior + 7 new = 12 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/ai/test_context.py
git commit -m "test(ai/context): cover estimate_tokens and HistoryRoller"
```

---

## Task 5: `tools.py::TOOLS` schema list

**Files:**
- Modify: `synapse/ai/tools.py`
- Create: `tests/ai/test_tools_schema.py`

Define the 6 tool JSON schemas as a module-level list. Each entry matches the shape the Anthropic / OpenAI / Gemini tool-calling APIs expect:

```python
{"name": "...", "description": "...", "input_schema": {"type": "object", ...}}
```

- [ ] **Step 1: Write tests**

```python
from synapse.ai.tools import TOOLS, TOOL_NAMES


def test_all_six_tools_present():
    assert TOOL_NAMES == (
        "generate_workflow",
        "modify_workflow",
        "write_python_script",
        "inspect_canvas",
        "explain_node",
        "read_node_output",
    )
    assert [t["name"] for t in TOOLS] == list(TOOL_NAMES)


def test_each_tool_has_required_keys():
    for t in TOOLS:
        assert isinstance(t.get("name"), str) and t["name"]
        assert isinstance(t.get("description"), str) and t["description"]
        schema = t.get("input_schema")
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_write_python_script_requires_description():
    wps = next(t for t in TOOLS if t["name"] == "write_python_script")
    required = wps["input_schema"].get("required") or []
    assert "description" in required


def test_modify_workflow_operations_is_array():
    mw = next(t for t in TOOLS if t["name"] == "modify_workflow")
    ops = mw["input_schema"]["properties"]["operations"]
    assert ops["type"] == "array"
    # item schema should list the supported op kinds
    item = ops.get("items", {})
    ops_enum = (item.get("properties", {}).get("op", {}).get("enum")
                or item.get("oneOf"))
    assert ops_enum is not None, "modify_workflow ops must be enumerated"


def test_read_node_output_requires_node_id():
    rno = next(t for t in TOOLS if t["name"] == "read_node_output")
    assert "node_id" in rno["input_schema"].get("required", [])
```

- [ ] **Step 2: Implement `synapse/ai/tools.py` TOOLS section**

Replace the module contents with:

```python
"""Tool schemas and dispatcher for the Synapse AI orchestrator.

Phase 2a: schemas + dispatcher + pure tool handlers (no LLM client calls).
Phase 2b: ChatOrchestrator + client tool-calling wire them together.
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = ["TOOLS", "TOOL_NAMES", "ToolDispatcher", "ToolResult"]


TOOL_NAMES = (
    "generate_workflow",
    "modify_workflow",
    "write_python_script",
    "inspect_canvas",
    "explain_node",
    "read_node_output",
)


TOOLS: list[dict] = [
    {
        "name": "generate_workflow",
        "description": (
            "Generate a full Synapse workflow from scratch given a natural-language goal. "
            "Runs the two-pass node-selection + JSON-generation pipeline. "
            "If the canvas is empty the workflow is applied silently; otherwise a "
            "preview payload is returned for the UI to confirm."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "User's goal in plain English."},
                "constraints": {"type": "string", "description": "Optional extra constraints."},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "modify_workflow",
        "description": (
            "Apply a list of graph operations to the current canvas in a single undo group. "
            "Partial success is allowed — failed ops are reported, successful ones stay."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "description": "Ordered list of graph operations.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["add_node", "remove_node", "connect",
                                         "disconnect", "set_prop"],
                            },
                            "type": {"type": "string", "description": "Node class name (add_node)."},
                            "id":   {"type": "string", "description": "Node id to target (most ops)."},
                            "src":  {"type": "string", "description": "Source node id (connect/disconnect)."},
                            "dst":  {"type": "string", "description": "Destination node id (connect/disconnect)."},
                            "src_port": {"type": "string"},
                            "dst_port": {"type": "string"},
                            "prop": {"type": "string", "description": "Property name (set_prop)."},
                            "value": {"description": "Property value (set_prop). Any JSON type."},
                        },
                        "required": ["op"],
                    },
                }
            },
            "required": ["operations"],
        },
    },
    {
        "name": "write_python_script",
        "description": (
            "Generate Python code for a PythonScriptNode. If node_id is given, the code "
            "is written directly into the node's script_code property (undoable). "
            "The node's input/output port counts are resized to match n_inputs/n_outputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Target PythonScriptNode id."},
                "description": {"type": "string", "description": "What the code should do."},
                "n_inputs":  {"type": "integer", "minimum": 1, "maximum": 8, "default": 1},
                "n_outputs": {"type": "integer", "minimum": 1, "maximum": 8, "default": 1},
                "input_hints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "port": {"type": "string"},
                            "kind": {"type": "string", "enum": ["table", "image", "mask", "scalar"]},
                            "schema": {"type": "string"},
                        },
                    },
                },
                "output_hints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "port": {"type": "string"},
                            "kind": {"type": "string", "enum": ["table", "image", "mask", "figure"]},
                        },
                    },
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "inspect_canvas",
        "description": (
            "Return nodes + edges + properties for the current canvas, capped at ~2000 "
            "output tokens. Use when you need full details about what is on the graph."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "If omitted, returns all nodes.",
                },
                "include_props": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "explain_node",
        "description": (
            "Return the authoritative documentation card (ports, props, docstring) "
            "for a Synapse node class by exact class name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_type": {"type": "string", "description": "Node class name, e.g. 'ParticlePropsNode'."},
            },
            "required": ["node_type"],
        },
    },
    {
        "name": "read_node_output",
        "description": (
            "Peek at the last-evaluated output of a node: shape, dtype, "
            "head-of-table, min/max/NaN-count for images, or error string. "
            "Includes a small thumbnail only when the active client supports vision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Id of the node to read."},
            },
            "required": ["node_id"],
        },
    },
]


class ToolResult(dict):
    """Dict subclass with an ``ok`` property so tool handlers can return either
    ``ToolResult(...)`` or a regular dict with an ``"error"`` key."""

    @property
    def ok(self) -> bool:
        return "error" not in self


class ToolDispatcher:
    """Register handlers and dispatch by name. Subclasses may override
    ``dispatch`` for caching / logging; Phase 2a uses the base class as-is."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict], Any]] = {}

    def register(self, name: str, handler: Callable[[dict], Any]) -> None:
        if name not in TOOL_NAMES:
            raise ValueError(f"Unknown tool name: {name}")
        if name in self._handlers:
            raise ValueError(f"Tool already registered: {name}")
        self._handlers[name] = handler

    def registered_names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def dispatch(self, name: str, tool_input: dict) -> Any:
        """Call the registered handler. Errors are wrapped into ``{"error": "..."}``
        so the orchestrator always gets a dict-like result to feed back to the LLM.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"No handler registered for tool: {name}"}
        try:
            result = handler(tool_input)
            if not isinstance(result, dict):
                return {"error": f"Tool {name} returned non-dict: {type(result).__name__}"}
            return result
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ai/test_tools_schema.py -v`
Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add synapse/ai/tools.py tests/ai/test_tools_schema.py
git commit -m "feat(ai/tools): add 6-tool schema list and result type"
```

---

## Task 6: `tools.py::ToolDispatcher` tests

The dispatcher was implemented in Task 5's full file write. This task adds focused tests.

**Files:**
- Create: `tests/ai/test_tool_dispatcher.py`

- [ ] **Step 1: Write tests**

```python
import pytest
from synapse.ai.tools import ToolDispatcher, TOOL_NAMES


def test_register_and_dispatch_roundtrip():
    d = ToolDispatcher()
    d.register("inspect_canvas", lambda inp: {"echo": inp})
    assert d.registered_names() == ("inspect_canvas",)
    assert d.dispatch("inspect_canvas", {"node_ids": ["x"]}) == {"echo": {"node_ids": ["x"]}}


def test_register_rejects_unknown_name():
    d = ToolDispatcher()
    with pytest.raises(ValueError):
        d.register("not_a_tool", lambda _: {})


def test_register_rejects_duplicates():
    d = ToolDispatcher()
    d.register("explain_node", lambda _: {})
    with pytest.raises(ValueError):
        d.register("explain_node", lambda _: {})


def test_dispatch_unknown_name_returns_error():
    d = ToolDispatcher()
    out = d.dispatch("explain_node", {"node_type": "X"})
    assert "error" in out and "No handler" in out["error"]


def test_dispatch_wraps_handler_exception_into_error():
    d = ToolDispatcher()
    def boom(_inp):
        raise RuntimeError("kaboom")
    d.register("explain_node", boom)
    out = d.dispatch("explain_node", {"node_type": "X"})
    assert out == {"error": "RuntimeError: kaboom"}


def test_dispatch_rejects_non_dict_handler_return():
    d = ToolDispatcher()
    d.register("explain_node", lambda _: "oops, not a dict")
    out = d.dispatch("explain_node", {"node_type": "X"})
    assert "error" in out and "non-dict" in out["error"]


def test_all_six_tool_names_are_registrable():
    d = ToolDispatcher()
    for name in TOOL_NAMES:
        d.register(name, lambda _inp, n=name: {"name": n})
    assert set(d.registered_names()) == set(TOOL_NAMES)
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/ai/test_tool_dispatcher.py -v`
Expected: 7 pass.

- [ ] **Step 3: Commit**

```bash
git add tests/ai/test_tool_dispatcher.py
git commit -m "test(ai/tools): cover ToolDispatcher registration + dispatch"
```

---

## Task 7: `tool_handlers/inspect_canvas.py`

**Files:**
- Create: `synapse/ai/tool_handlers/inspect_canvas.py`
- Create: `tests/ai/test_handler_inspect_canvas.py`

Returns `{nodes: [...], edges: [...], truncated: bool}`. Reuses the existing `serialize_graph()` helper in `synapse/llm_assistant.py` and applies a token cap using `estimate_tokens` from `ai/context`.

- [ ] **Step 1: Write tests**

```python
from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tool_handlers.inspect_canvas import make_inspect_canvas_handler


def _graph_with_two_nodes():
    g = FakeGraph()
    a = FakeNode("a", "CSVLoader", {"path": "/x.csv"})
    b = FakeNode("b", "SortTable", {"column": "value"})
    g.add_node(a); g.add_node(b)
    a.add_output("out_1").connect_to(b.add_input("in_1"))
    return g


def test_inspect_returns_all_nodes_when_node_ids_empty():
    g = _graph_with_two_nodes()
    handler = make_inspect_canvas_handler(g)
    out = handler({})
    assert set(n["type"] for n in out["nodes"]) == {"CSVLoader", "SortTable"}
    assert len(out["edges"]) == 1


def test_inspect_filters_to_requested_nodes():
    g = _graph_with_two_nodes()
    handler = make_inspect_canvas_handler(g)
    out = handler({"node_ids": ["a"]})
    assert [n["type"] for n in out["nodes"]] == ["CSVLoader"]
    # edges filtered to only those touching requested nodes
    assert out["edges"] == [] or all(
        e_src in {"a"} or e_dst in {"a"}
        for e_src, e_dst in (tuple(e) for e in out["edges"])
    )


def test_inspect_omits_props_when_include_props_false():
    g = _graph_with_two_nodes()
    handler = make_inspect_canvas_handler(g)
    out = handler({"include_props": False})
    for n in out["nodes"]:
        assert "props" not in n


def test_inspect_sets_truncated_flag_when_over_budget():
    # Build a wide graph — 50 nodes with long property values — to exceed cap.
    g = FakeGraph()
    for i in range(50):
        g.add_node(FakeNode(f"n{i}", "BigNode", {"blob": "x" * 500}))
    handler = make_inspect_canvas_handler(g, token_cap=200)
    out = handler({})
    assert out["truncated"] is True
    assert isinstance(out["nodes"], list)  # still returns a (shortened) list
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Implement `synapse/ai/tool_handlers/inspect_canvas.py`**

```python
"""inspect_canvas tool handler — read-only canvas dump with a token cap."""
from __future__ import annotations

import json
from typing import Callable

from synapse.ai.context import estimate_tokens


DEFAULT_TOKEN_CAP = 2000


def make_inspect_canvas_handler(graph, token_cap: int = DEFAULT_TOKEN_CAP) -> Callable[[dict], dict]:
    """Bind a handler to this graph + cap so it can be registered with ToolDispatcher."""

    def _handler(tool_input: dict) -> dict:
        node_id_filter: set[str] | None = None
        if tool_input.get("node_ids"):
            node_id_filter = set(tool_input["node_ids"])
        include_props = tool_input.get("include_props", True)

        nodes_out: list[dict] = []
        for node in graph.all_nodes():
            if node_id_filter is not None and node.id not in node_id_filter:
                continue
            entry: dict = {"id": node.id, "type": getattr(node, "type_name", type(node).__name__)}
            if include_props:
                try:
                    props = {
                        k: v for k, v in node.model.custom_properties.items()
                        if not k.startswith("_")
                    }
                except Exception:
                    props = {}
                if props:
                    entry["props"] = props
            nodes_out.append(entry)

        edges_out: list[list[str]] = []
        keep = {n["id"] for n in nodes_out}
        for node in graph.all_nodes():
            src_id = node.id
            for _, port in node.outputs().items():
                for connected in port.connected_ports():
                    dst_id = connected.node().id
                    if src_id in keep or dst_id in keep:
                        if (src_id in keep and dst_id in keep) or node_id_filter is None:
                            edges_out.append([src_id, dst_id])

        # Token cap — serialise, if over budget trim nodes until under.
        result = {"nodes": nodes_out, "edges": edges_out, "truncated": False}
        approx = estimate_tokens(json.dumps(result))
        while approx > token_cap and nodes_out:
            nodes_out.pop()
            edges_out[:] = [
                e for e in edges_out
                if e[0] in {n["id"] for n in nodes_out} and e[1] in {n["id"] for n in nodes_out}
            ]
            result = {"nodes": nodes_out, "edges": edges_out, "truncated": True}
            approx = estimate_tokens(json.dumps(result))
        return result

    return _handler
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_handler_inspect_canvas.py -v`
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/tool_handlers/inspect_canvas.py tests/ai/test_handler_inspect_canvas.py
git commit -m "feat(ai/handlers): add inspect_canvas handler"
```

---

## Task 8: `tool_handlers/explain_node.py`

**Files:**
- Create: `synapse/ai/tool_handlers/explain_node.py`
- Create: `tests/ai/test_handler_explain_node.py`

Looks up a node class name and returns its detailed card. Reuses `build_detailed_cards` from `synapse/llm_assistant.py`.

- [ ] **Step 1: Write tests**

```python
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
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Implement `synapse/ai/tool_handlers/explain_node.py`**

```python
"""explain_node tool handler — return the catalog card for a node class."""
from __future__ import annotations


def explain_node_handler(tool_input: dict) -> dict:
    node_type = (tool_input or {}).get("node_type")
    if not node_type:
        return {"error": "explain_node requires 'node_type' (string)."}

    # build_detailed_cards lives in llm_assistant and reads the schema JSON.
    from synapse.llm_assistant import build_detailed_cards

    try:
        card = build_detailed_cards([node_type])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if not card or not card.strip():
        return {"error": f"No catalog entry for node type: {node_type}"}

    return {"node_type": node_type, "card": card}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_handler_explain_node.py -v`
Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/tool_handlers/explain_node.py tests/ai/test_handler_explain_node.py
git commit -m "feat(ai/handlers): add explain_node handler"
```

---

## Task 9: `tool_handlers/read_node_output.py`

**Files:**
- Create: `synapse/ai/tool_handlers/read_node_output.py`
- Create: `tests/ai/test_handler_read_node_output.py`

Inspects a node's `output_values` (populated by `evaluate()`) and returns kind/metadata/text-preview/error. Thumbnail support is declared via a `supports_vision` callable injected by the caller so tests stay deterministic. When a thumbnail would be added it's base64 of the PNG.

- [ ] **Step 1: Write tests**

```python
import numpy as np
import pandas as pd

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tool_handlers.read_node_output import make_read_node_output_handler


class _TableStub:
    def __init__(self, df):
        self.df = df


class _ImageStub:
    def __init__(self, arr):
        self.payload = arr


def test_read_table_output_returns_head_and_shape():
    g = FakeGraph()
    n = FakeNode("n1", "SomeNode")
    n.output_values["out_1"] = _TableStub(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    n.add_output("out_1")
    g.add_node(n)
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_id": "n1"})
    assert out["kind"] == "table"
    assert out["metadata"]["shape"] == [3, 2]
    assert "a" in out["text_preview"] and "b" in out["text_preview"]
    assert "thumbnail" not in out


def test_read_image_output_reports_shape_dtype_range_no_vision():
    g = FakeGraph()
    n = FakeNode("n1", "SomeNode")
    n.output_values["out_1"] = _ImageStub(np.zeros((10, 10), dtype=np.uint8))
    n.add_output("out_1")
    g.add_node(n)
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_id": "n1"})
    assert out["kind"] == "image"
    assert out["metadata"]["shape"] == [10, 10]
    assert out["metadata"]["dtype"] == "uint8"
    assert "thumbnail" not in out


def test_read_image_output_includes_thumbnail_when_vision_supported():
    pil = __import__("PIL.Image", fromlist=["Image"])  # require PIL at runtime
    g = FakeGraph()
    n = FakeNode("n1", "SomeNode")
    n.output_values["out_1"] = _ImageStub(np.zeros((10, 10), dtype=np.uint8))
    n.add_output("out_1")
    g.add_node(n)
    handler = make_read_node_output_handler(g, supports_vision=lambda: True)
    out = handler({"node_id": "n1"})
    assert "thumbnail" in out
    # Thumbnail is base64-encoded PNG
    import base64
    assert base64.b64decode(out["thumbnail"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_read_unknown_node_returns_error():
    g = FakeGraph()
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_id": "does_not_exist"})
    assert "error" in out


def test_read_node_with_no_output_values_returns_empty_marker():
    g = FakeGraph()
    n = FakeNode("n1", "Empty")
    n.add_output("out_1")
    g.add_node(n)
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_id": "n1"})
    assert out.get("kind") == "empty"
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Implement `synapse/ai/tool_handlers/read_node_output.py`**

```python
"""read_node_output tool handler — compact peek at a node's last evaluated output."""
from __future__ import annotations

import base64
import io
from typing import Callable

THUMB_SIZE = 256


def _unwrap(value):
    """Unwrap NodeData-like objects (payload / df attributes)."""
    if value is None:
        return None
    if hasattr(value, "df"):
        return value.df
    if hasattr(value, "payload"):
        return value.payload
    return value


def _image_thumbnail_b64(arr) -> str | None:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    a = arr
    if a.ndim == 2:
        img = Image.fromarray(_to_uint8(a))
    elif a.ndim == 3 and a.shape[-1] in (3, 4):
        img = Image.fromarray(_to_uint8(a))
    else:
        return None
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _to_uint8(a):
    import numpy as np
    if a.dtype == np.uint8:
        return a
    mn, mx = float(a.min()), float(a.max())
    if mx == mn:
        return (a * 0).astype(np.uint8)
    scaled = (a - mn) / (mx - mn) * 255.0
    return scaled.astype(np.uint8)


def make_read_node_output_handler(
    graph,
    supports_vision: Callable[[], bool],
):
    """Build a handler bound to a graph + a vision-capability probe."""

    def _handler(tool_input: dict) -> dict:
        node_id = (tool_input or {}).get("node_id")
        if not node_id:
            return {"error": "read_node_output requires 'node_id'."}
        node = next((n for n in graph.all_nodes() if n.id == node_id), None)
        if node is None:
            return {"error": f"No node with id: {node_id}"}
        if not getattr(node, "output_values", None):
            return {"kind": "empty", "node_id": node_id}
        # Pick the first declared output port's value.
        first_port = next(iter(node.outputs()), None)
        if first_port is None:
            return {"kind": "empty", "node_id": node_id}
        raw = node.output_values.get(first_port)
        unwrapped = _unwrap(raw)
        if unwrapped is None:
            return {"kind": "empty", "node_id": node_id}
        # Dispatch by type.
        try:
            import pandas as pd
            import numpy as np
        except Exception:
            pd = None; np = None

        if pd is not None and isinstance(unwrapped, pd.DataFrame):
            df = unwrapped
            return {
                "kind": "table",
                "node_id": node_id,
                "metadata": {"shape": list(df.shape), "columns": list(df.columns.astype(str))},
                "text_preview": df.head(10).to_markdown(index=False),
            }
        if np is not None and isinstance(unwrapped, np.ndarray):
            arr = unwrapped
            meta = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(arr.min()) if arr.size else None,
                "max": float(arr.max()) if arr.size else None,
                "nan_count": int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0,
            }
            out = {"kind": "image", "node_id": node_id, "metadata": meta, "text_preview": ""}
            if supports_vision():
                thumb = _image_thumbnail_b64(arr)
                if thumb:
                    out["thumbnail"] = thumb
            return out

        # Fallback — treat as scalar / other.
        return {
            "kind": "other",
            "node_id": node_id,
            "metadata": {"type": type(unwrapped).__name__},
            "text_preview": repr(unwrapped)[:500],
        }

    return _handler
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_handler_read_node_output.py -v`
Expected: 5 pass.

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/tool_handlers/read_node_output.py tests/ai/test_handler_read_node_output.py
git commit -m "feat(ai/handlers): add read_node_output handler with optional thumbnail"
```

---

## Task 10: `tool_handlers/generate_workflow.py`

**Files:**
- Create: `synapse/ai/tool_handlers/generate_workflow.py`
- Create: `tests/ai/test_handler_generate_workflow.py`

Wraps the existing two-pass flow. We do NOT reuse the `TwoPassLLMWorker` Qt wrapper directly — it's `QObject`-based and async. Instead we call the two underlying functions sequentially. The handler takes an `LLMClient` dependency so tests can mock it.

The handler does NOT auto-apply to the canvas. It returns a preview payload the caller (Phase 2b orchestrator) uses to either apply silently (canvas empty) or route through the UI Apply/Discard flow.

- [ ] **Step 1: Write tests**

```python
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


def test_generate_workflow_canvas_was_empty_false_when_graph_has_nodes():
    from tests.ai.fakes import FakeNode
    g = FakeGraph(); g.add_node(FakeNode("pre", "Existing"))
    client = _mock_client_yielding(
        pass1_json='{"nodes": ["CSVLoader"]}',
        pass2_json='{"nodes":[{"id":1,"type":"CSVLoader"}],"edges":[]}',
    )
    handler = make_generate_workflow_handler(graph=g, client=client)
    out = handler({"goal": "x"})
    assert out["canvas_was_empty"] is False
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Implement `synapse/ai/tool_handlers/generate_workflow.py`**

```python
"""generate_workflow tool handler — two-pass workflow generation from a goal."""
from __future__ import annotations

import json
from typing import Any


def make_generate_workflow_handler(graph, client):
    """Build a handler bound to (graph, client).

    ``client`` must expose ``chat_multi(system: str, messages: list[dict]) -> str``.
    Phase 1's OllamaClient / OpenAIClient / ClaudeClient etc. all satisfy this.
    """

    def _handler(tool_input: dict) -> dict:
        goal = (tool_input or {}).get("goal")
        if not goal:
            return {"error": "generate_workflow requires 'goal' (string)."}
        constraints = (tool_input or {}).get("constraints") or ""

        # Import lazily — llm_assistant is heavy and we want tests isolated.
        from synapse.llm_assistant import (
            build_condensed_catalog,
            build_selection_prompt,
            build_detailed_cards,
            build_system_prompt,
        )

        catalog = build_condensed_catalog()
        user_message = goal + (("\n\nConstraints: " + constraints) if constraints else "")

        # Pass 1: select relevant node class names.
        sel_prompt = build_selection_prompt(catalog)
        try:
            raw1 = client.chat_multi(
                system=sel_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            selection = json.loads(raw1)
            selected_names: list[str] = list(selection.get("nodes", []))
        except Exception as e:
            return {"error": f"Pass 1 (node selection) failed: {type(e).__name__}: {e}"}

        # Pass 2: generate full workflow with detailed cards.
        detail_sys = build_system_prompt(build_detailed_cards(selected_names) or catalog)
        try:
            raw2 = client.chat_multi(
                system=detail_sys,
                messages=[{"role": "user", "content": user_message}],
            )
            workflow = json.loads(raw2)
        except Exception as e:
            return {"error": f"Pass 2 (workflow JSON) failed: {type(e).__name__}: {e}"}

        nodes = workflow.get("nodes") or []
        edges = workflow.get("edges") or []
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "preview_types": [n.get("type", "?") for n in nodes],
            "workflow": workflow,
            "canvas_was_empty": len(list(graph.all_nodes())) == 0,
        }

    return _handler
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_handler_generate_workflow.py -v`
Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/tool_handlers/generate_workflow.py tests/ai/test_handler_generate_workflow.py
git commit -m "feat(ai/handlers): add generate_workflow handler wrapping two-pass flow"
```

---

## Task 11: `tool_handlers/modify_workflow.py`

**Files:**
- Create: `synapse/ai/tool_handlers/modify_workflow.py`
- Create: `tests/ai/test_handler_modify_workflow.py`

Applies a list of ops (`add_node`, `remove_node`, `connect`, `disconnect`, `set_prop`) to the graph. Partial success is allowed; failed ops are recorded.

For Phase 2a we bind to the same duck-typed graph the other handlers use. Real NodeGraphQt integration (undo groups, node factory lookup) is deferred to Phase 2b when we have the real graph in scope. The handler dispatches through small inline helpers that the `FakeGraph` can satisfy.

- [ ] **Step 1: Write tests**

```python
from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tool_handlers.modify_workflow import make_modify_workflow_handler


class _Factory:
    """Tiny stand-in for NodeGraphQt's create_node. Tests record each call."""
    def __init__(self):
        self.created = []

    def __call__(self, type_name: str, node_id: str) -> FakeNode:
        n = FakeNode(node_id, type_name)
        self.created.append(n)
        return n


def _setup():
    g = FakeGraph(); factory = _Factory()
    handler = make_modify_workflow_handler(graph=g, node_factory=factory)
    return g, factory, handler


def test_add_node_applies_and_reports_success():
    g, factory, handler = _setup()
    out = handler({"operations": [
        {"op": "add_node", "type": "CSVLoader", "id": "n1"},
    ]})
    assert out["applied"] == [{"op": "add_node", "id": "n1"}]
    assert out["failed"] == []
    assert g.get_node_by_id("n1") is not None


def test_connect_disconnect_roundtrip():
    g, factory, handler = _setup()
    out = handler({"operations": [
        {"op": "add_node", "type": "CSVLoader", "id": "a"},
        {"op": "add_node", "type": "SortTable", "id": "b"},
        {"op": "connect", "src": "a", "src_port": "out_1",
                          "dst": "b", "dst_port": "in_1"},
    ]})
    assert out["failed"] == []
    a = g.get_node_by_id("a"); b = g.get_node_by_id("b")
    assert a.outputs()["out_1"].connected_ports()[0].node().id == "b"

    out2 = handler({"operations": [
        {"op": "disconnect", "src": "a", "src_port": "out_1",
                             "dst": "b", "dst_port": "in_1"},
    ]})
    assert out2["failed"] == []
    assert a.outputs()["out_1"].connected_ports() == []


def test_set_prop_updates_custom_property():
    g, factory, handler = _setup()
    handler({"operations": [{"op": "add_node", "type": "CSVLoader", "id": "n1"}]})
    out = handler({"operations": [
        {"op": "set_prop", "id": "n1", "prop": "path", "value": "/data.csv"},
    ]})
    assert out["failed"] == []
    assert g.get_node_by_id("n1").model.custom_properties["path"] == "/data.csv"


def test_remove_node():
    g, factory, handler = _setup()
    handler({"operations": [{"op": "add_node", "type": "X", "id": "n1"}]})
    out = handler({"operations": [{"op": "remove_node", "id": "n1"}]})
    assert out["failed"] == []
    assert g.get_node_by_id("n1") is None


def test_partial_success_reports_failures_individually():
    g, factory, handler = _setup()
    out = handler({"operations": [
        {"op": "add_node", "type": "CSVLoader", "id": "a"},   # ok
        {"op": "set_prop", "id": "nonexistent", "prop": "x", "value": 1},  # fail
        {"op": "add_node", "type": "SortTable", "id": "b"},   # ok
    ]})
    applied_ids = [a["id"] for a in out["applied"]]
    assert "a" in applied_ids and "b" in applied_ids
    assert len(out["failed"]) == 1
    assert out["failed"][0]["op"]["id"] == "nonexistent"


def test_unknown_op_kind_reports_failure():
    g, factory, handler = _setup()
    out = handler({"operations": [{"op": "do_magic"}]})
    assert out["applied"] == []
    assert len(out["failed"]) == 1
    assert "unknown op" in out["failed"][0]["reason"].lower()
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Implement `synapse/ai/tool_handlers/modify_workflow.py`**

```python
"""modify_workflow tool handler — apply a batch of graph operations."""
from __future__ import annotations

from typing import Callable


def make_modify_workflow_handler(graph, node_factory: Callable[[str, str], object]):
    """Bind a handler to (graph, node_factory).

    ``node_factory(type_name: str, node_id: str)`` creates a new node of the
    requested type and returns it. Phase 2b will supply a factory backed by
    NodeGraphQt's ``create_node`` + the class registry; Phase 2a tests supply
    a trivial callable that yields FakeNode instances.
    """

    def _lookup(node_id: str):
        for n in graph.all_nodes():
            if n.id == node_id:
                return n
        return None

    def _apply_one(op: dict) -> tuple[bool, str, dict | None]:
        kind = op.get("op")
        if kind == "add_node":
            if not op.get("type") or not op.get("id"):
                return False, "add_node requires 'type' and 'id'", None
            if _lookup(op["id"]) is not None:
                return False, f"node id already exists: {op['id']}", None
            node = node_factory(op["type"], op["id"])
            graph.add_node(node)
            return True, "", {"op": "add_node", "id": op["id"]}
        if kind == "remove_node":
            nid = op.get("id")
            node = _lookup(nid or "")
            if node is None:
                return False, f"no such node: {nid}", None
            graph.remove_node(node)
            return True, "", {"op": "remove_node", "id": nid}
        if kind == "set_prop":
            node = _lookup(op.get("id") or "")
            if node is None:
                return False, f"no such node: {op.get('id')}", None
            prop = op.get("prop")
            if not prop:
                return False, "set_prop requires 'prop'", None
            try:
                node.set_property(prop, op.get("value"))
            except Exception as e:
                return False, f"{type(e).__name__}: {e}", None
            return True, "", {"op": "set_prop", "id": node.id, "prop": prop}
        if kind == "connect":
            src = _lookup(op.get("src") or ""); dst = _lookup(op.get("dst") or "")
            if not src or not dst:
                return False, "connect requires existing src and dst node ids", None
            sport = src.outputs().get(op.get("src_port") or "")
            dport = dst.inputs().get(op.get("dst_port") or "")
            if not sport:
                sport = src.add_output(op.get("src_port") or "out_1")
            if not dport:
                dport = dst.add_input(op.get("dst_port") or "in_1")
            sport.connect_to(dport)
            return True, "", {"op": "connect", "src": src.id, "dst": dst.id}
        if kind == "disconnect":
            src = _lookup(op.get("src") or ""); dst = _lookup(op.get("dst") or "")
            if not src or not dst:
                return False, "disconnect requires existing src and dst node ids", None
            sport = src.outputs().get(op.get("src_port") or "")
            dport = dst.inputs().get(op.get("dst_port") or "")
            if sport and dport and dport in sport.connected_ports():
                sport.peers.remove(dport); dport.peers.remove(sport)
                return True, "", {"op": "disconnect", "src": src.id, "dst": dst.id}
            return False, "no such connection", None
        return False, f"unknown op kind: {kind}", None

    def _handler(tool_input: dict) -> dict:
        ops = (tool_input or {}).get("operations")
        if not isinstance(ops, list):
            return {"error": "modify_workflow requires 'operations' (array)."}
        applied: list[dict] = []
        failed: list[dict] = []
        for op in ops:
            ok, reason, record = _apply_one(op)
            if ok and record is not None:
                applied.append(record)
            else:
                failed.append({"op": op, "reason": reason})
        return {"applied": applied, "failed": failed}

    return _handler
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_handler_modify_workflow.py -v`
Expected: 6 pass.

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/tool_handlers/modify_workflow.py tests/ai/test_handler_modify_workflow.py
git commit -m "feat(ai/handlers): add modify_workflow handler with partial-success reporting"
```

---

## Task 12: `tool_handlers/write_python_script.py`

**Files:**
- Create: `synapse/ai/tool_handlers/write_python_script.py`
- Create: `tests/ai/test_handler_write_python_script.py`

Handler flow:
1. Locate target `PythonScriptNode` by id. Refuse if the node isn't a `PythonScriptNode` (`type_name != "PythonScriptNode"` in tests; `type(node).__name__` in real use).
2. Resize input/output port counts via `set_property("n_inputs", ...)` / `set_property("n_outputs", ...)`.
3. Call a sub-LLM with `WRITE_PYTHON_SCRIPT_SUBPROMPT` + the user description + port hints.
4. Set the `script_code` property to the result.
5. Return `{target_node_id, line_count, assigned_outputs}`.

- [ ] **Step 1: Write tests**

```python
from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tool_handlers.write_python_script import make_write_python_script_handler


def _python_node(node_id="n1") -> FakeNode:
    n = FakeNode(node_id, "PythonScriptNode")
    return n


def test_writes_code_to_existing_python_script_node():
    g = FakeGraph(); n = _python_node(); g.add_node(n)
    client = MagicMock()
    client.chat_multi.return_value = "out_1 = in_1.copy()\nout_1['log_a'] = np.log2(in_1['a'])"
    handler = make_write_python_script_handler(graph=g, client=client)
    out = handler({
        "node_id": "n1",
        "description": "log2 of column a",
        "n_inputs": 1,
        "n_outputs": 1,
    })
    assert "error" not in out
    assert out["target_node_id"] == "n1"
    assert n.model.custom_properties["script_code"].startswith("out_1 = in_1.copy()")
    assert out["line_count"] == 2
    assert out["assigned_outputs"] == ["out_1"]


def test_refuses_wrong_node_type():
    g = FakeGraph(); n = FakeNode("n1", "CSVLoader"); g.add_node(n)
    handler = make_write_python_script_handler(graph=g, client=MagicMock())
    out = handler({"node_id": "n1", "description": "x"})
    assert "error" in out and "PythonScriptNode" in out["error"]


def test_refuses_missing_node():
    handler = make_write_python_script_handler(graph=FakeGraph(), client=MagicMock())
    out = handler({"node_id": "ghost", "description": "x"})
    assert "error" in out


def test_missing_description_returns_error():
    g = FakeGraph(); g.add_node(_python_node())
    handler = make_write_python_script_handler(graph=g, client=MagicMock())
    out = handler({"node_id": "n1"})
    assert "error" in out and "description" in out["error"]


def test_strips_markdown_fences_from_llm_output():
    g = FakeGraph(); n = _python_node(); g.add_node(n)
    client = MagicMock()
    client.chat_multi.return_value = "```python\nout_1 = in_1\n```"
    handler = make_write_python_script_handler(graph=g, client=client)
    out = handler({"node_id": "n1", "description": "passthrough"})
    assert "error" not in out
    assert n.model.custom_properties["script_code"] == "out_1 = in_1"


def test_resizes_ports_via_set_property():
    g = FakeGraph(); n = _python_node(); g.add_node(n)
    client = MagicMock()
    client.chat_multi.return_value = "out_1 = in_1\nout_2 = in_2"
    handler = make_write_python_script_handler(graph=g, client=client)
    handler({
        "node_id": "n1",
        "description": "passthrough two inputs",
        "n_inputs": 2,
        "n_outputs": 2,
    })
    assert n.model.custom_properties.get("n_inputs") == 2
    assert n.model.custom_properties.get("n_outputs") == 2
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Implement `synapse/ai/tool_handlers/write_python_script.py`**

```python
"""write_python_script tool handler — generate Python for a PythonScriptNode."""
from __future__ import annotations

import re

from synapse.ai.prompts import WRITE_PYTHON_SCRIPT_SUBPROMPT


_FENCE_RE = re.compile(r"^```(?:[a-zA-Z]*)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text.strip()


def make_write_python_script_handler(graph, client):
    """Bind a handler to (graph, client). Client must expose chat_multi()."""

    def _handler(tool_input: dict) -> dict:
        inp = tool_input or {}
        node_id = inp.get("node_id")
        description = inp.get("description")
        if not node_id:
            return {"error": "write_python_script requires 'node_id'."}
        if not description:
            return {"error": "write_python_script requires 'description'."}

        node = next((n for n in graph.all_nodes() if n.id == node_id), None)
        if node is None:
            return {"error": f"No node with id: {node_id}"}
        type_name = getattr(node, "type_name", type(node).__name__)
        if type_name != "PythonScriptNode":
            return {"error": (
                "write_python_script only targets PythonScriptNode; "
                f"node {node_id} is {type_name}."
            )}

        n_in = int(inp.get("n_inputs") or 1)
        n_out = int(inp.get("n_outputs") or 1)
        try:
            node.set_property("n_inputs", n_in)
            node.set_property("n_outputs", n_out)
        except Exception as e:
            return {"error": f"Failed to resize ports: {type(e).__name__}: {e}"}

        user_msg = (
            f"{description}\n\n"
            f"Ports: n_inputs={n_in}, n_outputs={n_out}.\n"
            f"input_hints={inp.get('input_hints') or []}\n"
            f"output_hints={inp.get('output_hints') or []}\n"
            "Return ONLY the code body."
        )
        try:
            raw = client.chat_multi(
                system=WRITE_PYTHON_SCRIPT_SUBPROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:
            return {"error": f"sub-LLM call failed: {type(e).__name__}: {e}"}

        code = _strip_fences(raw)
        try:
            node.set_property("script_code", code, push_undo=True)
        except TypeError:
            node.set_property("script_code", code)
        except Exception as e:
            return {"error": f"Failed to write script_code: {type(e).__name__}: {e}"}

        assigned = sorted(set(re.findall(r"\b(out_\d+)\b", code)))
        return {
            "target_node_id": node_id,
            "line_count": len(code.splitlines()),
            "assigned_outputs": assigned,
        }

    return _handler
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_handler_write_python_script.py -v`
Expected: 6 pass.

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/tool_handlers/write_python_script.py tests/ai/test_handler_write_python_script.py
git commit -m "feat(ai/handlers): add write_python_script handler with sub-LLM + fence stripping"
```

---

## Task 13: Integration test — dispatcher wiring end-to-end

**Files:**
- Create: `tests/ai/test_dispatcher_wiring.py`

Locks in that all 6 handlers can be registered on a single `ToolDispatcher` instance and produce the right shape of result.

- [ ] **Step 1: Write integration test**

```python
from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tools import ToolDispatcher, TOOL_NAMES
from synapse.ai.tool_handlers.inspect_canvas import make_inspect_canvas_handler
from synapse.ai.tool_handlers.explain_node import explain_node_handler
from synapse.ai.tool_handlers.read_node_output import make_read_node_output_handler
from synapse.ai.tool_handlers.generate_workflow import make_generate_workflow_handler
from synapse.ai.tool_handlers.modify_workflow import make_modify_workflow_handler
from synapse.ai.tool_handlers.write_python_script import make_write_python_script_handler


def _all_tool_dispatcher():
    graph = FakeGraph()
    client = MagicMock()
    client.chat_multi.side_effect = [
        '{"nodes":["CSVLoader"]}',
        '{"nodes":[{"id":1,"type":"CSVLoader"}],"edges":[]}',
        "out_1 = in_1",  # for write_python_script
    ]
    d = ToolDispatcher()
    d.register("inspect_canvas", make_inspect_canvas_handler(graph))
    d.register("explain_node", explain_node_handler)
    d.register("read_node_output", make_read_node_output_handler(graph, lambda: False))
    d.register("generate_workflow", make_generate_workflow_handler(graph, client))
    d.register("modify_workflow", make_modify_workflow_handler(
        graph, node_factory=lambda t, i: FakeNode(i, t)))
    d.register("write_python_script", make_write_python_script_handler(graph, client))
    return d, graph


def test_dispatcher_knows_all_six_tools():
    d, _ = _all_tool_dispatcher()
    assert set(d.registered_names()) == set(TOOL_NAMES)


def test_inspect_canvas_empty_graph():
    d, _ = _all_tool_dispatcher()
    out = d.dispatch("inspect_canvas", {})
    assert out == {"nodes": [], "edges": [], "truncated": False}


def test_explain_node_unknown():
    d, _ = _all_tool_dispatcher()
    out = d.dispatch("explain_node", {"node_type": "NoSuchNode"})
    assert "error" in out


def test_end_to_end_modify_then_read():
    d, graph = _all_tool_dispatcher()
    # Add a PythonScriptNode, write code into it, inspect canvas.
    d.dispatch("modify_workflow", {"operations": [
        {"op": "add_node", "type": "PythonScriptNode", "id": "py1"},
    ]})
    out_write = d.dispatch("write_python_script", {
        "node_id": "py1", "description": "passthrough",
        "n_inputs": 1, "n_outputs": 1,
    })
    assert "error" not in out_write
    out_inspect = d.dispatch("inspect_canvas", {})
    ids = [n["id"] for n in out_inspect["nodes"]]
    assert "py1" in ids


def test_bad_tool_name_returns_wrapped_error():
    d, _ = _all_tool_dispatcher()
    out = d.dispatch("summon_demon", {})
    assert "error" in out
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/ai/test_dispatcher_wiring.py -v`
Expected: 5 pass.

Also run the full suite: `pytest tests/ -v`
Expected: all tests pass (Phase 1's 37 + Phase 2a's new tests ≈ 80+).

- [ ] **Step 3: Commit**

```bash
git add tests/ai/test_dispatcher_wiring.py
git commit -m "test(ai): integration test for full 6-tool dispatcher wiring"
```

---

## Task 14: `ai/__init__.py` convenience re-exports

**Files:**
- Modify: `synapse/ai/__init__.py`

- [ ] **Step 1: Add re-exports**

Replace the empty `synapse/ai/__init__.py` with:

```python
"""Synapse AI package — clients, tools, orchestrator (Phase 2b)."""
from synapse.ai.feature_flags import get_use_orchestrator
from synapse.ai.prompts import BASE_SYSTEM_PROMPT, WRITE_PYTHON_SCRIPT_SUBPROMPT
from synapse.ai.context import graph_summary, estimate_tokens, HistoryRoller
from synapse.ai.tools import TOOLS, TOOL_NAMES, ToolDispatcher

__all__ = [
    "get_use_orchestrator",
    "BASE_SYSTEM_PROMPT", "WRITE_PYTHON_SCRIPT_SUBPROMPT",
    "graph_summary", "estimate_tokens", "HistoryRoller",
    "TOOLS", "TOOL_NAMES", "ToolDispatcher",
]
```

- [ ] **Step 2: Smoke test**

Run: `python -c "from synapse.ai import TOOLS, ToolDispatcher, graph_summary; print(len(TOOLS))"`
Expected: `6`

Also: `pytest tests/ -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add synapse/ai/__init__.py
git commit -m "refactor(ai): add package convenience re-exports"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ BASE_SYSTEM_PROMPT with markdown + tool-use + no-raw-JSON rules — Task 2
- ✅ WRITE_PYTHON_SCRIPT_SUBPROMPT with PythonScriptNode API — Task 2
- ✅ graph_summary one-liner — Task 3
- ✅ HistoryRoller windowing + tool-result truncation — Task 3/4
- ✅ estimate_tokens budget helper — Task 3/4
- ✅ All 6 TOOLS schemas — Task 5
- ✅ ToolDispatcher register/dispatch with error wrapping — Task 5/6
- ✅ inspect_canvas handler with token cap — Task 7
- ✅ explain_node handler — Task 8
- ✅ read_node_output handler with optional thumbnail — Task 9
- ✅ generate_workflow handler wrapping two-pass flow — Task 10
- ✅ modify_workflow handler with partial success — Task 11
- ✅ write_python_script handler with sub-LLM + fence strip — Task 12
- ✅ Feature flag USE_ORCHESTRATOR — Task 1 (flag exists; UI routing is Phase 2b)

**Deferred to Phase 2b (explicitly out of scope for 2a):**
- ChatOrchestrator agent loop + tool-call budget enforcement.
- ChatStreamWorker Qt thread.
- Native tool-calling wiring into Claude/OpenAI/Gemini `chat_with_tools_stream`.
- Prompt-based fallback protocol parser for Ollama/Groq.
- AIChatPanel feature-flag routing and tool-call chip UI.
- LLM-based history rollup (summary of dropped turns).
- Vision capability detection integrated into `read_node_output` (currently a callable passed in).
- `generate_workflow` Apply/Discard preview — handler only returns the payload; UI wiring is 2b.
- `modify_workflow` NodeGraphQt undo group + real-node factory — handler uses duck-typed graph; 2b wires the real NodeGraphQt instance + NodeGraphQt `create_node` for factory.

**Placeholder scan:** no `TBD`/`TODO`/"fill in later". Every handler has a complete implementation with tests.

**Type consistency:** every handler takes `tool_input: dict` and returns a `dict`. Handlers that need state (graph, client, factory) are factories (`make_..._handler(...)`) returning the real callable. Consistent signature across the six.
