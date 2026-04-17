# AI Chat Phase 3 — UI Polish

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the working Phase 2b orchestrator path and dress it up for human use — stream tokens into a live bubble instead of dumping them on turn end, render tool calls as inline chips with expandable JSON, replace the `QMessageBox` workflow-Apply modal with inline Apply/Discard controls, surface a token-budget meter under the input, give the user a dedicated Stop button, and show a vision-capability badge next to the model dropdown. No orchestrator or client behavior changes — this is strictly rendering, interaction, and Qt wiring.

**Scope is deliberately cosmetic.** History rollup + the real token-estimator-with-compression are punted to Phase 4. The meter in Phase 3 uses a dumb `chars ÷ 4` approximation so we can land the UI without the compression plumbing.

**Architecture / approach:**
- `AIChatPanel._chat_display` stays a single `QTextBrowser`. The bubble log is HTML-in-a-text-document; we track the **text-document position of the current assistant bubble** and rewrite its HTML block in place as tokens/chips/apply-state change. This avoids rearchitecting to a `QScrollArea`+widget-per-bubble model, which would be Phase 4+ work.
- Interactivity (chip expand/collapse, inline Apply/Discard, cancel from a chip) is routed through `QTextBrowser.anchorClicked` with `setOpenLinks(False)`. Links use custom schemes (`chip://{bubble_id}/{chip_id}`, `apply://{bubble_id}`, `discard://{bubble_id}`) parsed by a new `_on_anchor_clicked` slot.
- A per-bubble `_BubbleState` dataclass (text buffer, list of chips, pending workflow preview, expanded-chip set, document block position) is held in an `OrderedDict[bubble_id → _BubbleState]`. Each `_on_orch_*` slot updates the state and re-renders the one affected bubble. A 50 ms `QTimer` coalesces text deltas so we don't re-render per token on fast models.
- `ChatStreamWorker` signals are already present (`token_received`, `tool_call_started`, `tool_call_finished`, `workflow_preview`, `cap_exceeded`, `error`, `cancelled`, `turn_finished`). No changes needed in `chat_worker.py` or `orchestrator.py`.
- Stop button: new `QPushButton` next to Send; visible only while a turn is in flight. Replaces the current pattern of repurposing the Send button's label to "⏹".
- Vision badge: small `QLabel` next to the model dropdown that reads `👁 vision` or `text-only`, updated in `_on_chat_model_changed` and `_on_chat_provider_changed`.
- Token meter: small `QLabel` under the input reading `this turn: ~3.2k · session: ~24k · limit: 200k`. Uses a naive char-based estimator (`len(text) // 4`) plus a hard-coded per-model context-window map in `synapse/ai/context.py`.
- Tests: pure-logic tests for the new bubble-state renderer (input → HTML substrings) and the anchor-URL parser. Qt widget tests remain out of scope per existing repo convention (no `pytest-qt`).

**Tech Stack:** Python 3.13+, PySide6 (`QTextBrowser`, `QTextCursor`, `QTimer`), existing markdown renderer (`synapse/markdown_render.py`), existing `ChatStreamWorker`.

---

## File Structure

**New files:**

```
synapse/
  ai/
    bubble_state.py              (NEW — _BubbleState + _BubbleLog HTML renderer)
    token_estimate.py            (NEW — chars→tokens approximation + per-model window map)
tests/
  ai/
    test_bubble_state.py
    test_token_estimate.py
```

**Modified files:**

```
synapse/
  llm_assistant.py               (AIChatPanel: bubble-state wiring, Stop btn, vision badge,
                                  token meter, anchor routing, remove QMessageBox modal)
  ai/context.py                  (expose per-model context window lookup; small helper)
```

No change to `synapse/ai/orchestrator.py`, `synapse/ai/chat_worker.py`, or any client.

---

## Task 1: Bubble state + renderer

**Files:**
- Create: `synapse/ai/bubble_state.py`
- Create: `tests/ai/test_bubble_state.py`

Goal: extract every bubble's HTML into a pure function of a `_BubbleState` so we can re-render a bubble when state changes. Two data classes + one render fn. No Qt imports here — keeps tests lightweight.

### Types

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

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
```

### Renderer

`render_bubble_html(state: _BubbleState, colors: dict) -> str` emits one HTML fragment:

1. **User / system / error** bubbles — same HTML as today's `_append_bubble`, just moved here verbatim.
2. **Assistant** bubble — new layout, top-to-bottom inside one rounded card:
   - Header: small `AI` label, plus a blinking `●` cursor when `streaming=True`.
   - Chips row: each chip rendered as `<a href="chip://{bubble_id}/{chip_id}" style="...pill...">🔧 {name} → {status_glyph} {result_summary or '…'}</a>`. Status glyphs: `⋯` running, `✓` ok, `⚠` error.
   - Expanded chips: for each `chip_id` in `expanded_chips`, render a `<pre style="...">` block under the chips row showing `full_input` and `full_result` as indented JSON.
   - Markdown body: `render_markdown(state.text)`. Empty while tool-only turns run.
   - Workflow proposal block (if `state.workflow` is not None):
     - `pending` → summary line `"Proposed: {node_count} nodes, {edge_count} edges — {preview_types}"` + two anchor "buttons" `<a href="apply://{bubble_id}" style="...primary...">Apply</a> <a href="discard://{bubble_id}" style="...secondary...">Discard</a>`.
     - `applied` → muted `"✓ Applied"` line.
     - `discarded` → muted `"Discarded"` line.

The anchor-button styling is inline CSS borrowed from the existing Send-button colors (`#238636` for primary, `#21262d`/`#d0d7de` for secondary) so they read as buttons inside the bubble.

### Anchor URL parsing

Helper: `parse_anchor(url: str) -> tuple[action, bubble_id, chip_id|None]` where `action ∈ {"chip","apply","discard"}`. Unknown scheme → raises `ValueError`. Tests cover all three plus malformed input.

### Step 1: Write tests

```python
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

def test_error_bubble_uses_error_colors_and_no_tail_for_text_only_role():
    s = _BubbleState(bubble_id="b7", role="error", text="network down")
    html = render_bubble_html(s, COLORS)
    assert "network down" in html
    assert COLORS["err_bg"] in html
```

### Step 2: Run — ImportError expected

### Step 3: Implement `synapse/ai/bubble_state.py`

Copy the `role == "user"` / `role == "system"` / `role == "error"` HTML verbatim from `AIChatPanel._append_bubble` (llm_assistant.py:2710) into matching branches. Rewrite the `assistant` branch to build chips + expanded panels + markdown body + workflow block. `render_markdown` is imported from `synapse.markdown_render`.

HTML-escape user-provided strings (`input_preview`, `result_summary`, JSON payloads) via `html.escape`. Tool names are from a fixed allowlist and do not need escaping, but escape them anyway to keep the renderer defensively pure.

### Step 4: Run tests — 10 pass. Full suite: current 170 + 10 = 180.

### Step 5: Commit

```bash
git add synapse/ai/bubble_state.py tests/ai/test_bubble_state.py
git commit -m "feat(ai/chat): extract bubble rendering into a pure state+renderer"
```

---

## Task 2: Token-budget estimator

**Files:**
- Create: `synapse/ai/token_estimate.py`
- Create: `tests/ai/test_token_estimate.py`
- Modify: `synapse/ai/context.py` (re-export for discoverability; no logic here)

Goal: a dumb-but-useful estimator for the meter label. Explicitly not a Phase 4 compressor.

### API

```python
def estimate_tokens(text: str) -> int:
    """~4 chars per token. Empty string → 0."""

def estimate_messages_tokens(messages: list[dict]) -> int:
    """Sum over each message's role+content. Ignores tool-call structured fields
    beyond a +10 token fudge per message for overhead."""

def model_context_window(provider: str, model: str) -> int:
    """Return a best-guess context window (tokens) for a given provider+model.
    Falls back to 8192 when unknown. Table is hard-coded — it is a display hint,
    not a contract."""
```

The model table covers current defaults: Claude 4.x (200k), GPT-4.x / o-series (128k–1M), Gemini 2.5 (1M), Ollama Cloud Gemma 3 (128k), Ollama local / Groq (32k default). Unknown → 8192. When in doubt, err low — an underestimate inconveniences the user with a false "near budget" flag; an overestimate hides a real OOM.

### Step 1: Write tests

```python
from synapse.ai.token_estimate import (
    estimate_tokens, estimate_messages_tokens, model_context_window,
)

def test_empty_string_zero_tokens():
    assert estimate_tokens("") == 0

def test_four_chars_roughly_one_token():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100

def test_messages_sums_role_and_content_with_per_message_overhead():
    msgs = [
        {"role": "user", "content": "a" * 400},       # ~100 + 1 (role) + 10
        {"role": "assistant", "content": "b" * 400},  # ~100 + 2 + 10
    ]
    n = estimate_messages_tokens(msgs)
    assert 200 <= n <= 250

def test_messages_tolerates_missing_content():
    # Orchestrator's tool_result blocks may have no `content` key.
    msgs = [{"role": "tool", "tool_call_id": "abc"}]
    assert estimate_messages_tokens(msgs) >= 0

def test_context_window_known_models():
    assert model_context_window("Claude", "claude-sonnet-4-6") >= 200_000
    assert model_context_window("Gemini", "gemini-2.5-pro") >= 1_000_000

def test_context_window_unknown_falls_back_to_8k():
    assert model_context_window("Whatever", "unheard-of-3b") == 8192
```

### Step 2: Run — ImportError expected

### Step 3: Implement

Trivial. Keep the model table ≤30 entries; pattern-match on prefixes (`model.startswith("claude-")`, etc.) so new minor versions map correctly without table churn.

### Step 4: Run tests — 6 pass. Full suite: 180 + 6 = 186.

### Step 5: Commit

```bash
git add synapse/ai/token_estimate.py tests/ai/test_token_estimate.py
git commit -m "feat(ai): add token estimator and per-model context-window lookup"
```

---

## Task 3: Bubble log that rewrites in place

**Files:**
- Modify: `synapse/llm_assistant.py`
- Add to `tests/ai/test_bubble_state.py`: two small tests for `_BubbleLog` ordering (no Qt needed since `_BubbleLog` delegates to a pluggable `set_html(position, html)` callback).

Goal: replace today's "always append" behavior in `AIChatPanel` with a `_BubbleLog` helper that can **update a prior bubble's HTML in place**. This is what lets tokens stream into an existing bubble and chips flip from `⋯` to `✓` without appending a new copy each time.

### Approach

Introduce `_BubbleLog` inside `llm_assistant.py` (not in `bubble_state.py` — Qt-dependent):

```python
class _BubbleLog:
    def __init__(self, text_browser: QtWidgets.QTextBrowser, colors_getter):
        self._tb = text_browser
        self._colors = colors_getter            # callable for theme changes
        self._states: OrderedDict[str, _BubbleState] = OrderedDict()
        self._block_positions: dict[str, int] = {}   # bubble_id → start position

    def add(self, state: _BubbleState) -> str: ...
    def update(self, bubble_id: str, mutator: Callable[[_BubbleState], None]): ...
    def get(self, bubble_id: str) -> _BubbleState: ...
    def clear(self): ...
```

Implementation notes:
- `add`: remember current `document().characterCount()` as this bubble's start position, then append `render_bubble_html(state)`. Return a generated `bubble_id` (`f"b{self._next}"`).
- `update`: run the mutator on the state, then re-render just that bubble. Because arbitrary HTML blocks don't have stable widths in a text document, the practical approach is: **select from this bubble's start position to end-of-document, remove, and re-append this bubble + any later bubbles** in order. Re-appending later bubbles is cheap (a few KB of HTML per turn) and sidesteps fighting `QTextDocument` frame boundaries. The scroll position is preserved iff the user was pinned to the bottom (check `verticalScrollBar().value() == maximum()` before mutation, restore if true).
- `clear`: reset state + `self._tb.clear()`.

### Panel wiring

In `AIChatPanel._build_ui`, replace direct `self._chat_display.append(...)` usages with `self._bubble_log.add(_BubbleState(...))`. The existing `_append_bubble` method becomes a thin wrapper that delegates (preserves the legacy `_ChatWorker` path during Phase 3).

### Orchestrator slot rewrites (live-streaming behavior)

Today's flow appends a fresh system bubble per tool event. New flow: one assistant bubble owns the entire turn.

```python
def _run_with_orchestrator(self, user_text):
    # Create ONE assistant bubble up front.
    self._current_bubble_id = self._bubble_log.add(
        _BubbleState(bubble_id="", role="assistant", streaming=True)
    )
    # ...spawn worker as before...

def _on_orch_token(self, piece):
    self._pending_token_buffer += piece
    if not self._token_flush_timer.isActive():
        self._token_flush_timer.start(50)  # coalesce

def _flush_tokens(self):
    buf, self._pending_token_buffer = self._pending_token_buffer, ""
    if not buf: return
    self._bubble_log.update(
        self._current_bubble_id,
        lambda s: setattr(s, "text", s.text + buf),
    )

def _on_orch_tool_started(self, name, inp):
    chip_id = f"c{len(self._bubble_log.get(self._current_bubble_id).chips)}"
    self._bubble_log.update(self._current_bubble_id, lambda s: s.chips.append(
        ToolChip(chip_id=chip_id, name=name,
                 input_preview=_short_json(inp), status="running", full_input=inp)
    ))
    self._last_chip_id = chip_id

def _on_orch_tool_finished(self, name, result):
    cid = self._last_chip_id
    self._bubble_log.update(self._current_bubble_id, lambda s: _finish_chip(
        s, cid, result,
    ))

def _on_orch_workflow_preview(self, result):
    wf = WorkflowProposal(
        node_count=result["node_count"], edge_count=result["edge_count"],
        preview_types=list(result.get("preview_types", [])),
        state="applied" if result.get("canvas_was_empty") else "pending",
    )
    self._bubble_log.update(
        self._current_bubble_id, lambda s: setattr(s, "workflow", wf),
    )
    self._last_workflow = result.get("workflow") or {}
    if wf.state == "applied":
        self._apply_workflow_from_orchestrator(replace=True)

def _on_orch_turn_finished(self, final):
    self._token_flush_timer.stop()
    self._flush_tokens()
    self._bubble_log.update(
        self._current_bubble_id, lambda s: setattr(s, "streaming", False),
    )
```

`_finish_chip` finds the chip by id and sets `status`, `result_summary` (e.g. `"4 nodes"` derived from result keys), and `full_result=result`.

### Anchor routing

Connect once in `_build_ui`:

```python
self._chat_display.setOpenLinks(False)
self._chat_display.anchorClicked.connect(self._on_anchor_clicked)
```

```python
def _on_anchor_clicked(self, url: QtCore.QUrl):
    try:
        action, bubble_id, chip_id = parse_anchor(url.toString())
    except ValueError:
        return
    if action == "chip":
        self._bubble_log.update(bubble_id, lambda s: (
            s.expanded_chips.discard(chip_id) if chip_id in s.expanded_chips
            else s.expanded_chips.add(chip_id)
        ))
    elif action == "apply":
        state = self._bubble_log.get(bubble_id)
        if state.workflow and state.workflow.state == "pending":
            self._apply_workflow_from_orchestrator(replace=False)
            self._bubble_log.update(bubble_id, lambda s: setattr(s.workflow, "state", "applied"))
    elif action == "discard":
        self._bubble_log.update(bubble_id, lambda s: setattr(s.workflow, "state", "discarded"))
```

**Delete** `_on_orch_workflow_preview`'s `QMessageBox.question(...)` block — the inline buttons replace it.

### Step 1: Write tests for `_BubbleLog` ordering (no Qt)

Inject a fake `QTextBrowser` stand-in that records `append(html)` and `clear()` calls. Verify:

- `add` appends once per call.
- `update(id)` results in a final document content equal to the re-rendered sequence of all bubbles from `id` onward.
- `clear` empties state.

(We will not test the anchor slot — it needs a real QTextBrowser + `QUrl`, which requires `pytest-qt`. The parser is already covered in Task 1.)

### Step 2: Run — AttributeError/ImportError expected

### Step 3: Implement `_BubbleLog` + rewire slots

Non-trivial. Key gotchas:
- `QTextCursor.setPosition` is in characters, which matches `document().characterCount()`. Save start on `add`, restore on `update`.
- On `update`, use a cursor to select from the bubble's start to document end, `removeSelectedText()`, then rebuild.
- Some bubble HTML uses `<table>` — `QTextDocument` sometimes inserts a paragraph separator after a table that bumps character counts by 1. Use `self._tb.toHtml()` / `setHtml` path only as a fallback if positions drift in practice.
- Keep the legacy non-orchestrator path alive (still calls `_append_bubble` → `_bubble_log.add`). No change to how the old `_ChatWorker` emits results; it just becomes one bubble per emit.

### Step 4: Run tests — 3 new pass. Full suite: 186 + 3 = 189.

### Step 5: Manual smoke (not automated)

- Ask "Say hi" → one assistant bubble, tokens appear live, bubble ends with no chips.
- Ask "What's in the canvas?" on empty canvas → one bubble with `inspect_canvas ⋯ → ✓ 0 nodes` chip; clicking chip expands; clicking again collapses.
- Ask "Build me a sort-then-top-N on this CSV" on empty canvas → workflow applies silently, chip shows ✓ applied.
- Same prompt on a non-empty canvas → inline Apply / Discard buttons; clicking Apply merges and flips the bubble to `✓ Applied`; clicking Discard flips to `Discarded`, no mutation.

### Step 6: Commit

```bash
git add synapse/llm_assistant.py tests/ai/test_bubble_state.py
git commit -m "feat(ai/chat): stream tokens + chips + inline Apply/Discard into one bubble"
```

---

## Task 4: Dedicated Stop button

**Files:**
- Modify: `synapse/llm_assistant.py`

Goal: remove the Send-button-label-swap pattern and add a purpose-built Stop button.

### Changes

- In `_build_ui` (around llm_assistant.py:2208), add right after the Send button:
  ```python
  self._stop_btn = QtWidgets.QPushButton("Stop")
  self._stop_btn.setObjectName("stopBtn")
  self._stop_btn.setEnabled(False)
  self._stop_btn.setVisible(False)
  self._stop_btn.clicked.connect(self._on_stop_orchestrator)
  btn_row.addWidget(self._stop_btn)
  ```
- In `_build_style`, add `QPushButton#stopBtn { background: #da3633; color: #fff; border: 1px solid #f85149; font-weight: 600; }` + hover rule.
- `_run_with_orchestrator` (llm_assistant.py:2463): stop mutating `_send_btn.setText("⏹")` and stop swapping `.clicked` connections. Instead:
  ```python
  self._send_btn.setEnabled(False)
  self._stop_btn.setVisible(True)
  self._stop_btn.setEnabled(True)
  ```
- In all `_on_orch_*` turn-end slots (llm_assistant.py:2615 and siblings), restore via:
  ```python
  self._send_btn.setEnabled(True)
  self._stop_btn.setEnabled(False)
  self._stop_btn.setVisible(False)
  ```
- Remove the now-dead `_send_btn.clicked.disconnect()` dance and the manual `setText("Send")` calls.

### No tests

This is a pure Qt UI change; not testable without `pytest-qt`. Verify manually: send a prompt, confirm Stop appears and Send is disabled; click Stop and confirm cancellation works as before.

### Commit

```bash
git add synapse/llm_assistant.py
git commit -m "feat(ai/chat): dedicated Stop button instead of repurposed Send"
```

---

## Task 5: Vision badge

**Files:**
- Modify: `synapse/llm_assistant.py`

Goal: show the active client's vision capability at a glance.

### Changes

- Add a `QLabel` next to the model dropdown in `_build_ui`:
  ```python
  self._vision_badge = QtWidgets.QLabel()
  self._vision_badge.setObjectName("visionBadge")
  row.addWidget(self._vision_badge)
  ```
- New method:
  ```python
  def _refresh_vision_badge(self):
      ok = bool(self._client and getattr(self._client, "supports_vision", False))
      self._vision_badge.setText("👁 vision" if ok else "text-only")
      self._vision_badge.setStyleSheet(
          "color:#3fb950; font-size:11px; font-weight:600;"
          if ok else
          "color:#8b949e; font-size:11px; font-style:italic;"
      )
  ```
- Call `_refresh_vision_badge()` from:
  - `_load_config` (initial load)
  - `_on_chat_provider_changed`
  - `_on_chat_model_changed` (llm_assistant.py:2502)

### No tests (trivial formatting).

### Commit

```bash
git add synapse/llm_assistant.py
git commit -m "feat(ai/chat): show vision capability badge next to model dropdown"
```

---

## Task 6: Token meter

**Files:**
- Modify: `synapse/llm_assistant.py`

Goal: small always-visible meter under the input reading `this turn: ~3.2k · session: ~24k · limit: 200k`.

### Changes

- Add a `QLabel` below the input row:
  ```python
  self._token_meter = QtWidgets.QLabel()
  self._token_meter.setStyleSheet("color:#8b949e; font-size:10px; padding:2px 4px;")
  layout.addWidget(self._token_meter)
  ```
- Helper method:
  ```python
  def _refresh_token_meter(self, per_turn: int | None = None):
      from synapse.ai.token_estimate import estimate_messages_tokens, model_context_window
      session = estimate_messages_tokens(self._messages) + estimate_tokens(self._system)
      window = model_context_window(
          self._provider_combo.currentText(),
          self._chat_model_combo.currentText() if hasattr(self, "_chat_model_combo") else "",
      )
      parts = []
      if per_turn is not None:
          parts.append(f"this turn: ~{per_turn/1000:.1f}k")
      parts.append(f"session: ~{session/1000:.1f}k")
      parts.append(f"limit: {window//1000}k")
      self._token_meter.setText(" · ".join(parts))
  ```
- Call `_refresh_token_meter()` from:
  - `_build_ui` (end)
  - `_on_orch_turn_finished` (after the bubble closes) — pass `per_turn` from `_pending_token_buffer` length or a running counter maintained in `_on_orch_token`.
  - `_on_chat_model_changed` and `_on_chat_provider_changed` (to refresh the `limit:` number).
  - `_on_clear` (session resets).

### No tests

Estimator already covered in Task 2. The widget wiring is a formatting + signal wire-up.

### Commit

```bash
git add synapse/llm_assistant.py
git commit -m "feat(ai/chat): token-budget meter under input"
```

---

## Task 7: Full-suite + end-of-branch review

### Step 1: Run full suite

```bash
cd /Users/s/Desktop/demo/PySide_Node/.worktrees/ai-chat-phase2b
pytest -x
```

Expect 189 green tests.

### Step 2: Manual exercise matrix

One smoke pass per provider (Ollama local, Ollama Cloud, OpenRouter, OpenAI, Claude, Groq, Gemini). For each:

1. "Say hi" → streaming tokens, no chips, vision badge matches model, meter updates.
2. "What's in my canvas?" on empty canvas → one `inspect_canvas` chip, expand/collapse works.
3. "Build a CSV → sort → top-N workflow" on empty canvas → chip ✓, workflow applies silently, bubble says Applied.
4. Same prompt on a non-empty canvas → inline Apply/Discard buttons; both paths update bubble state correctly.
5. Long prompt → Stop mid-turn → cancellation clean; no lingering state.

### Step 3: End-of-branch review

```
Use superpowers:code-reviewer on the Phase 3 branch against plans/2026-04-17-ai-chat-phase-3-ui-polish.md
```

Address any findings as normal fix-up commits.

### Step 4: Merge decision

Phase 3 lives on branch `feat/ai-chat-phase-3` (stacked on `feat/ai-chat-phase2b`). Do **not** land to `main` — user prefers stacked worktrees per prior guidance. Leave branch in place for Phase 4.

---

## Out of scope (Phase 4 or later)

- **Real history rollup + compression.** The estimator here is a display gauge, not a budget enforcer. Phase 4 adds cheap-model summarization for turns older than the visible window plus thumbnail eviction.
- **QScrollArea + widget-per-bubble rewrite.** Today's re-render-tail approach is fine for chats ≤ ~100 bubbles. If users report sluggishness past that, this is the Phase 4+ escape hatch.
- **Drop the `USE_ORCHESTRATOR` feature flag and the legacy `_ChatWorker` path.** Also Phase 4.
- **Per-chip cancel links.** Nice-to-have once users complain; Stop cancels the whole turn for now.
- **Token meter hover with per-message breakdown.** Phase 4+, if anyone asks.

## Open questions

None at time of writing. Call out during code review if one emerges.
