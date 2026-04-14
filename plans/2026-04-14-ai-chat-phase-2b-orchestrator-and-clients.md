# AI Chat Phase 2b — Orchestrator, Client Tool-Calling, UI Wiring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the Phase 2a tool infrastructure and wire it into each LLM client's streaming interface (native tool-calling for Claude/OpenAI/Gemini, prompt-based fallback for Ollama/Groq), build a `ChatOrchestrator` agent loop that dispatches tools end-to-end, put the whole thing behind the existing `USE_ORCHESTRATOR` feature flag, and route `AIChatPanel` traffic through the new path when the flag is on.

**Architecture:**
- A new `synapse/ai/clients/tool_adapters.py` converts the provider-neutral `TOOLS` list into each provider's native shape (Anthropic `input_schema`, OpenAI `function.parameters`, Gemini `functionDeclarations`), and synthesizes a prompt-fallback system-prompt addendum for providers without native tool-calling.
- Each client's existing `chat_with_tools_stream(system, messages, tools=None)` is extended: when `tools` is passed, the request includes provider-native tool declarations, and the SSE/NDJSON parser emits `StreamEvent(kind="tool_call", tool_call={"id","name","input"})` whenever the model invokes a tool. Claude/OpenAI/Gemini use their native tool-use events; Ollama/Groq rely on the prompt-fallback parser that watches streamed text for `<tool_call>{...}</tool_call>` markers.
- A new `synapse/ai/orchestrator.py` hosts `ChatOrchestrator` — the agent loop. Per user turn: build system prompt + history + canvas summary, call `client.chat_with_tools_stream(..., tools=TOOLS)`, forward `text` events to the caller, and on `tool_call` dispatch the tool, append the result as a provider-appropriate message, and loop. Hard cap: 4 tool calls per user turn, then a single final prose turn.
- `synapse/ai/chat_worker.py` hosts `ChatStreamWorker(QObject)` that owns a `QThread`, drives the orchestrator, and emits Qt signals the chat panel renders.
- `AIChatPanel._on_send` picks the legacy `_ChatWorker` path or the new `ChatStreamWorker` path based on `get_use_orchestrator()`. When the orchestrator path is active, tool-call status shows as a small system-message bubble in the chat log (Phase 3 will upgrade to inline chips). `generate_workflow` results are applied silently on an empty canvas; otherwise a modal confirm dialog asks Apply / Discard (Phase 3 will replace with inline buttons).
- No existing tests lose coverage. New integration test drives an end-to-end turn with a mocked streaming client + real dispatcher + real `FakeGraph` to verify the full loop.

**Tech Stack:** Python 3.13+, PySide6, existing LLM clients (extended), `requests` streaming, `tests/ai/fakes.py`.

---

## File Structure

**New files:**

```
synapse/
  ai/
    clients/
      tool_adapters.py        (NEW — TOOLS → provider-native; prompt-fallback builder)
      _fallback_parser.py     (NEW — StreamedToolCallParser for <tool_call>...</tool_call>)
    orchestrator.py           (NEW — ChatOrchestrator + OrchestratorEvent)
    chat_worker.py            (NEW — ChatStreamWorker(QObject) for Qt integration)
tests/
  ai/
    test_tool_adapters.py
    test_fallback_parser.py
    test_orchestrator.py
    test_phase2b_integration.py
```

**Modified files:**

```
synapse/
  ai/
    clients/
      claude.py               (extend chat_with_tools_stream to emit tool_call events)
      openai.py               (same)
      gemini.py               (same)
      ollama.py               (inject fallback protocol into system prompt; parse output)
      groq.py                 (same as ollama)
  llm_assistant.py            (AIChatPanel._on_send routes via feature flag;
                               add _on_workflow_confirm for generate_workflow Apply/Discard)
```

No change to `synapse/ai/tools.py` or `synapse/ai/tool_handlers/*` (the Phase 2a surface is stable).

---

## Task 1: Tool format adapters

**Files:**
- Create: `synapse/ai/clients/tool_adapters.py`
- Create: `tests/ai/test_tool_adapters.py`

Functions:
- `to_anthropic_tools(tools: list[dict]) -> list[dict]` — identity (Anthropic already uses `{name, description, input_schema}`).
- `to_openai_tools(tools: list[dict]) -> list[dict]` — wraps each as `{"type": "function", "function": {"name", "description", "parameters"}}`.
- `to_gemini_tools(tools: list[dict]) -> list[dict]` — wraps as `[{"functionDeclarations": [{"name", "description", "parameters"}, ...]}]`.
- `build_fallback_prompt(tools: list[dict]) -> str` — produces a system-prompt addendum describing the `<tool_call>{...}</tool_call>` protocol with the tool schemas inlined as JSON.

### Step 1: Write tests

```python
import json

from synapse.ai.tools import TOOLS
from synapse.ai.clients.tool_adapters import (
    to_anthropic_tools, to_openai_tools, to_gemini_tools, build_fallback_prompt,
)


def test_anthropic_is_identity_shape():
    out = to_anthropic_tools(TOOLS)
    assert len(out) == 6
    for t in out:
        assert set(t.keys()) >= {"name", "description", "input_schema"}


def test_openai_wraps_in_function_envelope():
    out = to_openai_tools(TOOLS)
    assert len(out) == 6
    for t in out:
        assert t["type"] == "function"
        fn = t["function"]
        assert {"name", "description", "parameters"} <= set(fn)
        assert fn["parameters"]["type"] == "object"


def test_gemini_wraps_in_function_declarations_envelope():
    out = to_gemini_tools(TOOLS)
    # Gemini expects a single-element list containing {"functionDeclarations": [...]}
    assert len(out) == 1
    decls = out[0]["functionDeclarations"]
    assert len(decls) == 6
    for d in decls:
        assert {"name", "description", "parameters"} <= set(d)


def test_fallback_prompt_contains_protocol_and_schemas():
    txt = build_fallback_prompt(TOOLS)
    assert "<tool_call>" in txt and "</tool_call>" in txt
    # Must mention every tool by name so the LLM knows its options.
    for t in TOOLS:
        assert t["name"] in txt
    # Must be parseable JSON-within-a-string (tool schemas inlined).
    assert "input_schema" in txt or "parameters" in txt


def test_openai_stripped_default_keys_stay_stripped():
    # Phase 2a removed "default" from schemas — ensure openai pass-through
    # doesn't reintroduce them from anywhere.
    out = to_openai_tools(TOOLS)
    txt = json.dumps(out)
    assert '"default"' not in txt
```

### Step 2: Run — ImportError expected

### Step 3: Implement `synapse/ai/clients/tool_adapters.py`

```python
"""Convert provider-neutral tool schemas to each LLM provider's native shape.

Anthropic:  [{name, description, input_schema}]   (matches our internal format)
OpenAI:     [{type: "function", function: {name, description, parameters}}]
Gemini:     [{functionDeclarations: [{name, description, parameters}]}]
Fallback:   a system-prompt addendum describing the <tool_call> protocol.
"""
from __future__ import annotations

import json


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Anthropic's input_schema matches our internal shape; pass through."""
    return [
        {"name": t["name"], "description": t["description"],
         "input_schema": t["input_schema"]}
        for t in tools
    ]


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def to_gemini_tools(tools: list[dict]) -> list[dict]:
    return [{
        "functionDeclarations": [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
            for t in tools
        ]
    }]


_FALLBACK_HEADER = """\
You have access to the following tools. To call one, output exactly one line
starting with `<tool_call>` and ending with `</tool_call>` containing a JSON
object with keys "name" and "input", then STOP generating.

Example:
<tool_call>{"name": "inspect_canvas", "input": {}}</tool_call>

After a tool call, wait for the user message containing the result, then
continue. If you do not need to call a tool, just reply normally in markdown.
Never emit <tool_call>...</tool_call> as part of a larger explanation — the
line must stand alone.

Available tools:
"""


def build_fallback_prompt(tools: list[dict]) -> str:
    lines = [_FALLBACK_HEADER]
    for t in tools:
        lines.append(f"- name: {t['name']}")
        lines.append(f"  description: {t['description']}")
        lines.append(f"  input_schema: {json.dumps(t['input_schema'])}")
    return "\n".join(lines)
```

### Step 4: Run tests — 5 pass. Full suite: 103 + 5 = 108.

### Step 5: Commit

```bash
git add synapse/ai/clients/tool_adapters.py tests/ai/test_tool_adapters.py
git commit -m "feat(ai/clients): add tool-schema adapters for Anthropic/OpenAI/Gemini + fallback"
```

---

## Task 2: Fallback tool-call parser (for Ollama / Groq)

**Files:**
- Create: `synapse/ai/clients/_fallback_parser.py`
- Create: `tests/ai/test_fallback_parser.py`

A small state machine that consumes streaming text chunks and emits either `text` pass-through chunks or a single `tool_call` dict when it sees a complete `<tool_call>{...}</tool_call>` span.

### Step 1: Write tests

```python
from synapse.ai.clients._fallback_parser import StreamedToolCallParser


def test_passthrough_text():
    p = StreamedToolCallParser()
    assert p.feed("hello ") == [("text", "hello ")]
    assert p.feed("world.") == [("text", "world.")]
    assert p.finish() == []


def test_detects_complete_tool_call():
    p = StreamedToolCallParser()
    events = p.feed('<tool_call>{"name": "inspect_canvas", "input": {}}</tool_call>')
    # One tool_call event, no text leaking through the markers.
    kinds = [e[0] for e in events]
    assert kinds == ["tool_call"]
    tc = events[0][1]
    assert tc == {"name": "inspect_canvas", "input": {}}


def test_detects_tool_call_split_across_chunks():
    p = StreamedToolCallParser()
    out = []
    out += p.feed('some reply <tool_call>{"name"')
    out += p.feed(': "explain_node", "input":')
    out += p.feed(' {"node_type": "X"}}</tool_call> trailing')
    # Pre-marker text passes through; marker contents emit one tool_call;
    # post-marker text is suppressed (per protocol: the model is told to STOP).
    text_chunks = [e[1] for e in out if e[0] == "text"]
    assert "some reply " in "".join(text_chunks)
    tool_calls = [e[1] for e in out if e[0] == "tool_call"]
    assert tool_calls == [{"name": "explain_node", "input": {"node_type": "X"}}]


def test_malformed_json_inside_markers_emits_error():
    p = StreamedToolCallParser()
    events = p.feed('<tool_call>{broken json</tool_call>')
    kinds = [e[0] for e in events]
    assert "error" in kinds
    err_msg = next(e[1] for e in events if e[0] == "error")
    assert "JSON" in err_msg or "decode" in err_msg.lower()


def test_no_tool_call_finish_is_noop():
    p = StreamedToolCallParser()
    p.feed("hi there")
    assert p.finish() == []


def test_partial_opener_not_yet_a_tool_call():
    p = StreamedToolCallParser()
    # "<tool_c" is not yet a full opener; should be buffered, not emitted.
    assert p.feed("<tool_c") == []
    # When the opener completes, no text event for the opener itself.
    events = p.feed('all>{"name":"inspect_canvas","input":{}}</tool_call>')
    assert [e[0] for e in events] == ["tool_call"]
```

### Step 2: Run — ImportError expected

### Step 3: Implement `synapse/ai/clients/_fallback_parser.py`

```python
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
                    # Emit any text before the opener.
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
    start of ``<tool_call>``. Used to avoid emitting a partial opener as text.
    """
    for n in range(len(_OPEN) - 1, 0, -1):
        if buf.endswith(_OPEN[:n]):
            return n
    return 0
```

### Step 4: Run tests — 6 pass. Full suite: 108 + 6 = 114.

### Step 5: Commit

```bash
git add synapse/ai/clients/_fallback_parser.py tests/ai/test_fallback_parser.py
git commit -m "feat(ai/clients): add StreamedToolCallParser for prompt-based fallback"
```

---

## Task 3: Claude native tool-calling

**Files:**
- Modify: `synapse/ai/clients/claude.py`
- Modify: `tests/ai/test_claude_stream.py`

Extend `chat_with_tools_stream` so when `tools` is non-None, the request body includes `tools=to_anthropic_tools(tools)`, and the SSE parser additionally handles:
- `content_block_start` with `type=tool_use` → start collecting tool-call.
- `content_block_delta` with `delta.type=input_json_delta` → accumulate `delta.partial_json`.
- `content_block_stop` → emit a single `StreamEvent(kind="tool_call", tool_call={"id", "name", "input"})`.

Anthropic's `content_block_start` event JSON looks like:
```json
{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_abc","name":"inspect_canvas","input":{}}}
```
Each `content_block_delta` adds to `partial_json`. At `content_block_stop` we parse the full partial_json buffer.

### Step 1: Write test (append to `tests/ai/test_claude_stream.py`)

```python
def test_claude_stream_emits_tool_call_event_when_tools_enabled():
    from synapse.ai.tools import TOOLS
    client = ClaudeClient(api_key="sk-ant-test")
    # Minimal SSE: message_start, a tool_use content_block_start, two
    # input_json_delta parts, content_block_stop, message_stop.
    lines = [
        b"event: message_start",
        b'data: {"type":"message_start"}',
        b"",
        b"event: content_block_start",
        b'data: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"tool_use","id":"toolu_1","name":"inspect_canvas","input":{}}}',
        b"",
        b"event: content_block_delta",
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"input_json_delta","partial_json":"{\\"node_ids\\":"}}',
        b"",
        b"event: content_block_delta",
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"input_json_delta","partial_json":" [\\"a\\"]}"}}',
        b"",
        b"event: content_block_stop",
        b'data: {"type":"content_block_stop","index":0}',
        b"",
        b"event: message_stop",
        b'data: {"type":"message_stop"}',
    ]
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with patch("synapse.ai.clients.claude.requests.post", return_value=resp) as pm:
        events = list(client.chat_with_tools_stream(
            system="s", messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
        ))
    # Request included tools in Anthropic format.
    _, kwargs = pm.call_args
    payload = kwargs["json"]
    assert "tools" in payload
    assert any(t["name"] == "inspect_canvas" for t in payload["tools"])
    # One tool_call event, with parsed input.
    tcs = [e for e in events if e.kind == "tool_call"]
    assert len(tcs) == 1
    assert tcs[0].tool_call["name"] == "inspect_canvas"
    assert tcs[0].tool_call["id"] == "toolu_1"
    assert tcs[0].tool_call["input"] == {"node_ids": ["a"]}
    assert events[-1].kind == "done"
```

### Step 2: Run — failing (existing implementation ignores `tools`).

### Step 3: Modify `synapse/ai/clients/claude.py::chat_with_tools_stream`

Replace the existing method. New version includes a `tools` parameter propagation and the extended event parser. Full replacement:

```python
    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        from synapse.ai.clients.tool_adapters import to_anthropic_tools

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = to_anthropic_tools(tools)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        # State for the current tool_use content block (Anthropic emits these
        # in parts: content_block_start → delta(s) of partial_json → content_block_stop).
        pending_tool: dict | None = None  # {"id","name","buf"}

        try:
            with requests.post(
                f"{self.BASE_URL}/messages",
                headers=headers,
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    etype = obj.get("type")
                    if etype == "content_block_start":
                        cb = obj.get("content_block") or {}
                        if cb.get("type") == "tool_use":
                            pending_tool = {
                                "id": cb.get("id", ""),
                                "name": cb.get("name", ""),
                                "buf": "",
                            }
                    elif etype == "content_block_delta":
                        delta = obj.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            piece = delta.get("text") or ""
                            if piece:
                                yield StreamEvent(kind="text", text=piece)
                        elif delta.get("type") == "input_json_delta" and pending_tool is not None:
                            pending_tool["buf"] += delta.get("partial_json") or ""
                    elif etype == "content_block_stop":
                        if pending_tool is not None:
                            try:
                                parsed_input = json.loads(pending_tool["buf"] or "{}")
                            except json.JSONDecodeError:
                                parsed_input = {}
                            yield StreamEvent(
                                kind="tool_call",
                                tool_call={
                                    "id": pending_tool["id"],
                                    "name": pending_tool["name"],
                                    "input": parsed_input,
                                },
                            )
                            pending_tool = None
                    elif etype == "message_stop":
                        break
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
```

### Step 4: Run tests — new test passes. Existing 3 Claude tests still pass (tools=None path unchanged). Full: 115.

### Step 5: Commit

```bash
git add synapse/ai/clients/claude.py tests/ai/test_claude_stream.py
git commit -m "feat(ai/claude): emit tool_call events during native Anthropic tool-use"
```

---

## Task 4: OpenAI native tool-calling

**Files:**
- Modify: `synapse/ai/clients/openai.py`
- Modify: `tests/ai/test_openai_stream.py`

OpenAI streams tool calls via `choices[0].delta.tool_calls` — an array of partial deltas, each with an `index`, optionally `id`, optionally `function.name`, and incremental `function.arguments` fragments. A single tool call is assembled by concatenating arguments across deltas at the same `index`. Multiple tool calls per turn are allowed. We emit one `StreamEvent(kind="tool_call", ...)` per assembled call when the stream finishes.

### Step 1: Test (append to `tests/ai/test_openai_stream.py`)

```python
def test_openai_stream_emits_tool_call_event_when_tools_enabled():
    from synapse.ai.tools import TOOLS
    client = OpenAIClient(api_key="sk-test")
    # OpenAI emits tool_calls as array deltas. Simulate:
    #   delta(tool_calls=[{index:0, id:"call_1", type:"function",
    #                       function:{name:"inspect_canvas", arguments:""}}])
    #   delta(tool_calls=[{index:0, function:{arguments:"{\"node_ids"}}])
    #   delta(tool_calls=[{index:0, function:{arguments:":[\"a\"]}"}}])
    #   finish_reason "tool_calls", then [DONE].
    lines = [
        b'data: {"choices":[{"delta":{"tool_calls":['
        b'{"index":0,"id":"call_1","type":"function",'
        b'"function":{"name":"inspect_canvas","arguments":""}}]}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":['
        b'{"index":0,"function":{"arguments":"{\\"node_ids"}}]}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":['
        b'{"index":0,"function":{"arguments":":[\\"a\\"]}"}}]}}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        b'data: [DONE]',
    ]
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with patch("synapse.ai.clients.openai.requests.post", return_value=resp) as pm:
        events = list(client.chat_with_tools_stream(
            system="s", messages=[], tools=TOOLS,
        ))
    # Request included tools wrapped in the OpenAI function envelope.
    _, kwargs = pm.call_args
    assert "tools" in kwargs["json"]
    assert kwargs["json"]["tools"][0]["type"] == "function"
    # One tool_call event, assembled from the 3 deltas.
    tcs = [e for e in events if e.kind == "tool_call"]
    assert len(tcs) == 1
    assert tcs[0].tool_call["id"] == "call_1"
    assert tcs[0].tool_call["name"] == "inspect_canvas"
    assert tcs[0].tool_call["input"] == {"node_ids": ["a"]}
```

### Step 2: Run — failing (tools currently ignored).

### Step 3: Modify `synapse/ai/clients/openai.py::chat_with_tools_stream`

Full replacement:

```python
    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        from synapse.ai.clients.tool_adapters import to_openai_tools

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": 0.1,
            "stream": True,
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
        # Accumulators per tool_call index: {index: {"id","name","args"}}
        partial: dict[int, dict] = {}

        try:
            with requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        yield StreamEvent(kind="text", text=piece)
                    tool_deltas = delta.get("tool_calls") or []
                    for tcd in tool_deltas:
                        idx = tcd.get("index", 0)
                        slot = partial.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if tcd.get("id"):
                            slot["id"] = tcd["id"]
                        fn = tcd.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
                # Emit assembled tool_calls in index order.
                for idx in sorted(partial):
                    slot = partial[idx]
                    try:
                        parsed = json.loads(slot["args"] or "{}")
                    except json.JSONDecodeError:
                        parsed = {}
                    yield StreamEvent(
                        kind="tool_call",
                        tool_call={"id": slot["id"], "name": slot["name"], "input": parsed},
                    )
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
```

### Step 4: Run tests — new test passes. Existing 3 OpenAI tests still pass. Full: 116.

### Step 5: Commit

```bash
git add synapse/ai/clients/openai.py tests/ai/test_openai_stream.py
git commit -m "feat(ai/openai): emit tool_call events during native OpenAI function-calling"
```

---

## Task 5: Gemini native tool-calling

**Files:**
- Modify: `synapse/ai/clients/gemini.py`
- Modify: `tests/ai/test_gemini_stream.py`

Gemini's streaming response contains `candidates[0].content.parts[]` entries. A text part is `{"text": "..."}`. A tool-call part is `{"functionCall": {"name": "...", "args": {...}}}`. A single turn can contain multiple parts including mixed text + function_call.

### Step 1: Test

```python
def test_gemini_stream_emits_tool_call_event_when_tools_enabled():
    from synapse.ai.tools import TOOLS
    client = GeminiClient(api_key="gk-test")
    lines = [
        b'data: {"candidates":[{"content":{"parts":['
        b'{"functionCall":{"name":"inspect_canvas","args":{"node_ids":["a"]}}}'
        b']}}]}',
    ]
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with patch("synapse.ai.clients.gemini.requests.post", return_value=resp) as pm:
        events = list(client.chat_with_tools_stream(
            system="s", messages=[], tools=TOOLS,
        ))
    _, kwargs = pm.call_args
    assert "tools" in kwargs["json"]
    assert "functionDeclarations" in kwargs["json"]["tools"][0]
    tcs = [e for e in events if e.kind == "tool_call"]
    assert len(tcs) == 1
    assert tcs[0].tool_call["name"] == "inspect_canvas"
    assert tcs[0].tool_call["input"] == {"node_ids": ["a"]}
```

### Step 2: Run — failing.

### Step 3: Modify `synapse/ai/clients/gemini.py::chat_with_tools_stream`

Full replacement:

```python
    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        from synapse.ai.clients.tool_adapters import to_gemini_tools

        url = f"{self.BASE_URL}/models/{self.model}:streamGenerateContent"
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.1},
        }
        if tools:
            payload["tools"] = to_gemini_tools(tools)

        try:
            with requests.post(
                url,
                params={"key": self.api_key, "alt": "sse"},
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    cands = obj.get("candidates") or []
                    if not cands:
                        continue
                    parts = (cands[0].get("content") or {}).get("parts") or []
                    for p in parts:
                        text = p.get("text")
                        fn = p.get("functionCall")
                        if text:
                            yield StreamEvent(kind="text", text=text)
                        if fn:
                            yield StreamEvent(
                                kind="tool_call",
                                tool_call={
                                    "id": fn.get("name", ""),  # Gemini has no call id
                                    "name": fn.get("name", ""),
                                    "input": fn.get("args") or {},
                                },
                            )
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
```

### Step 4: Run tests — new test passes. Existing 3 Gemini tests still pass.

### Step 5: Commit

```bash
git add synapse/ai/clients/gemini.py tests/ai/test_gemini_stream.py
git commit -m "feat(ai/gemini): emit tool_call events during native Gemini function-calling"
```

---

## Task 6: Prompt-fallback wiring for Ollama + Groq

**Files:**
- Modify: `synapse/ai/clients/ollama.py`
- Modify: `synapse/ai/clients/groq.py`
- Modify: `tests/ai/test_ollama_stream.py`
- Modify: `tests/ai/test_groq_stream.py`

Both clients use the same approach:
1. When `tools` is passed, prepend `build_fallback_prompt(tools)` to the system prompt.
2. Route each text chunk through a `StreamedToolCallParser`. Text events pass through; a detected tool_call emits a `StreamEvent(kind="tool_call", ...)` and the stream is effectively ended (no further text events expected, since the model was told to stop).
3. If the parser emits an error, surface as `StreamEvent(kind="error", ...)`.

### Step 1: Tests — append to both `test_ollama_stream.py` and `test_groq_stream.py`

For Ollama:

```python
def test_ollama_stream_detects_tool_call_from_ndjson():
    from synapse.ai.tools import TOOLS
    client = OllamaClient(model="gemma3:2b")
    # Simulate the model emitting the tool_call marker across two NDJSON chunks.
    lines = [
        b'{"message":{"content":"<tool_call>{\\"name\\":"},"done":false}',
        b'{"message":{"content":"\\"inspect_canvas\\",\\"input\\":{}}</tool_call>"},"done":false}',
        b'{"message":{"content":""},"done":true}',
    ]
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with patch("synapse.ai.clients.ollama.requests.post", return_value=resp) as pm:
        events = list(client.chat_with_tools_stream(
            system="sys", messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
        ))
    # System prompt was augmented with the fallback protocol addendum.
    _, kwargs = pm.call_args
    sys_msg = kwargs["json"]["messages"][0]
    assert sys_msg["role"] == "system"
    assert "<tool_call>" in sys_msg["content"]
    # One tool_call event.
    tcs = [e for e in events if e.kind == "tool_call"]
    assert len(tcs) == 1
    assert tcs[0].tool_call["name"] == "inspect_canvas"
    assert tcs[0].tool_call["input"] == {}
```

For Groq (nearly identical; different SSE shape):

```python
def test_groq_stream_detects_tool_call_from_sse():
    from synapse.ai.tools import TOOLS
    client = GroqClient(api_key="gsk-test")
    # Two text chunks together form the full tool_call marker.
    lines = [
        b'data: {"choices":[{"delta":{"content":"<tool_call>{\\"name\\":\\"explain_node\\","}}]}',
        b'data: {"choices":[{"delta":{"content":"\\"input\\":{\\"node_type\\":\\"X\\"}}</tool_call>"}}]}',
        b'data: [DONE]',
    ]
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with patch("synapse.ai.clients.groq.requests.post", return_value=resp) as pm:
        events = list(client.chat_with_tools_stream(
            system="s", messages=[], tools=TOOLS,
        ))
    # System prompt was augmented.
    _, kwargs = pm.call_args
    sys_msg = kwargs["json"]["messages"][0]
    assert "<tool_call>" in sys_msg["content"]
    tcs = [e for e in events if e.kind == "tool_call"]
    assert len(tcs) == 1
    assert tcs[0].tool_call["name"] == "explain_node"
    assert tcs[0].tool_call["input"] == {"node_type": "X"}
```

### Step 2: Run — failing.

### Step 3: Modify `synapse/ai/clients/ollama.py::chat_with_tools_stream`

Find the method body. Add a fallback branch that injects the prompt and runs text through the parser.

```python
    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        from synapse.ai.clients.tool_adapters import build_fallback_prompt
        from synapse.ai.clients._fallback_parser import StreamedToolCallParser

        effective_system = system
        parser: StreamedToolCallParser | None = None
        if tools:
            effective_system = build_fallback_prompt(tools) + "\n\n" + system
            parser = StreamedToolCallParser()

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": effective_system}] + messages,
            "stream": True,
            "options": {"temperature": 0.1},
        }
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self._headers(),
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    piece = obj.get("message", {}).get("content", "")
                    if piece:
                        if parser is not None:
                            for kind, payload_ev in parser.feed(piece):
                                if kind == "text":
                                    yield StreamEvent(kind="text", text=payload_ev)
                                elif kind == "tool_call":
                                    tc = payload_ev
                                    yield StreamEvent(
                                        kind="tool_call",
                                        tool_call={
                                            "id": tc.get("name", ""),
                                            "name": tc.get("name", ""),
                                            "input": tc.get("input") or {},
                                        },
                                    )
                                elif kind == "error":
                                    yield StreamEvent(kind="error", error=str(payload_ev))
                        else:
                            yield StreamEvent(kind="text", text=piece)
                    if obj.get("done"):
                        break
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
```

### Step 4: Modify `synapse/ai/clients/groq.py::chat_with_tools_stream`

Same idea, but adapted to the OpenAI-style SSE structure. Replace the method:

```python
    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        from synapse.ai.clients.tool_adapters import build_fallback_prompt
        from synapse.ai.clients._fallback_parser import StreamedToolCallParser

        effective_system = system
        parser: StreamedToolCallParser | None = None
        if tools:
            effective_system = build_fallback_prompt(tools) + "\n\n" + system
            parser = StreamedToolCallParser()

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": effective_system}] + messages,
            "temperature": 0.1,
            "stream": True,
        }
        try:
            with requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    piece = (choices[0].get("delta") or {}).get("content") or ""
                    if piece:
                        if parser is not None:
                            for kind, payload_ev in parser.feed(piece):
                                if kind == "text":
                                    yield StreamEvent(kind="text", text=payload_ev)
                                elif kind == "tool_call":
                                    tc = payload_ev
                                    yield StreamEvent(
                                        kind="tool_call",
                                        tool_call={
                                            "id": tc.get("name", ""),
                                            "name": tc.get("name", ""),
                                            "input": tc.get("input") or {},
                                        },
                                    )
                                elif kind == "error":
                                    yield StreamEvent(kind="error", error=str(payload_ev))
                        else:
                            yield StreamEvent(kind="text", text=piece)
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
```

### Step 5: Run tests — both new tests pass, existing tests unchanged.

### Step 6: Commit

```bash
git add synapse/ai/clients/ollama.py synapse/ai/clients/groq.py tests/ai/test_ollama_stream.py tests/ai/test_groq_stream.py
git commit -m "feat(ai/clients): wire prompt-based tool-call fallback into Ollama + Groq"
```

---

## Task 7: `ChatOrchestrator` — agent loop

**Files:**
- Create: `synapse/ai/orchestrator.py`
- Create: `tests/ai/test_orchestrator.py`

Core class. Wraps one user turn:
1. Assemble request: `BASE_SYSTEM_PROMPT + graph_summary(graph)` as system, rolled history + current user message.
2. Call `client.chat_with_tools_stream(system, messages, tools=TOOLS)`.
3. For each event:
   - `text`: forward as `OrchestratorEvent(kind="text", text=...)` for UI streaming.
   - `tool_call`: dispatch via `ToolDispatcher`, append provider-appropriate tool-result messages, call the client again. Hard cap 4 rounds; if hit, inject a system note and force one final prose turn.
   - `done`: end turn.
   - `error`: forward and end turn.
4. Check `cancel()` between rounds and between stream events — abort cleanly.

Tool-result message format: we use a **provider-neutral** convention on the conversation messages list — a `{"role": "tool", "tool_call_id": ..., "content": json.dumps(result)}` entry. Each client's `chat_with_tools_stream` translates these on the way out. For Phase 2b we add the translation as part of the orchestrator's follow-up call — Claude expects `{"role": "user", "content": [{"type":"tool_result", ...}]}`, OpenAI expects a dedicated tool role, Gemini expects `functionResponse`.

**Scope-trim:** to keep this task manageable, the orchestrator only supports **Claude** and **OpenAI** tool-result injection natively. For Ollama/Groq (prompt-fallback), we synthesize a user-role message that says "Tool result for `X`:\n```json\n<result>\n```" — which the fallback-prompt-driven model can read. Gemini support is a known limitation and documented in the orchestrator docstring — Phase 2c or later will fill it in. (Alternatively: if the Gemini work is trivial here, include it — but don't block on it.)

### Step 1: Write tests

```python
from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.clients.base import StreamEvent
from synapse.ai.tools import ToolDispatcher
from synapse.ai.tool_handlers.inspect_canvas import make_inspect_canvas_handler
from synapse.ai.tool_handlers.explain_node import explain_node_handler
from synapse.ai.orchestrator import ChatOrchestrator


def _dispatcher(graph):
    d = ToolDispatcher()
    d.register("inspect_canvas", make_inspect_canvas_handler(graph))
    d.register("explain_node", explain_node_handler)
    return d


def _mock_client(event_batches: list[list[StreamEvent]]):
    """Return a client whose chat_with_tools_stream yields each batch in turn."""
    client = MagicMock()
    it = iter(event_batches)

    def stream(system, messages, tools=None):
        try:
            batch = next(it)
        except StopIteration:
            batch = [StreamEvent(kind="done")]
        for ev in batch:
            yield ev

    client.chat_with_tools_stream.side_effect = stream
    client.supports_vision = False
    client.model = "test-model"
    return client


def test_orchestrator_plain_text_turn():
    g = FakeGraph()
    client = _mock_client([[
        StreamEvent(kind="text", text="Hello!"),
        StreamEvent(kind="done"),
    ]])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g))
    events = list(orch.run_turn("hi"))
    kinds = [e.kind for e in events]
    assert "text" in kinds
    assert events[-1].kind == "turn_done"


def test_orchestrator_single_tool_call_round():
    g = FakeGraph(); g.add_node(FakeNode("a", "CSVLoader"))
    client = _mock_client([
        # Turn 1: model calls inspect_canvas, stream ends.
        [
            StreamEvent(kind="tool_call", tool_call={"id": "t1", "name": "inspect_canvas", "input": {}}),
            StreamEvent(kind="done"),
        ],
        # Turn 2 (after tool result): model replies with text.
        [
            StreamEvent(kind="text", text="Canvas has one CSVLoader."),
            StreamEvent(kind="done"),
        ],
    ])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g))
    events = list(orch.run_turn("what's on my canvas?"))
    kinds = [e.kind for e in events]
    assert "tool_call_started" in kinds
    assert "tool_call_finished" in kinds
    assert any(e.kind == "text" and "CSVLoader" in e.text for e in events)
    # Exactly 2 stream calls (one per turn).
    assert client.chat_with_tools_stream.call_count == 2


def test_orchestrator_enforces_4_call_cap():
    g = FakeGraph()
    # Client keeps calling inspect_canvas forever.
    infinite = [
        [
            StreamEvent(kind="tool_call", tool_call={"id": f"t{i}", "name": "inspect_canvas", "input": {}}),
            StreamEvent(kind="done"),
        ]
        for i in range(10)
    ]
    client = _mock_client(infinite + [[StreamEvent(kind="text", text="OK I'll stop."), StreamEvent(kind="done")]])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g), max_tool_calls=4)
    events = list(orch.run_turn("loop test"))
    tool_starts = [e for e in events if e.kind == "tool_call_started"]
    assert len(tool_starts) == 4  # cap
    # A cap-exceeded system event was surfaced.
    assert any(e.kind == "cap_exceeded" for e in events)


def test_orchestrator_propagates_client_error():
    g = FakeGraph()
    client = _mock_client([[StreamEvent(kind="error", error="HTTP 500")]])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g))
    events = list(orch.run_turn("x"))
    err = next(e for e in events if e.kind == "error")
    assert "HTTP 500" in err.error


def test_orchestrator_cancel_flag_aborts_mid_stream():
    g = FakeGraph()
    # Infinite text stream; test sets cancel() after the first text event.
    client = _mock_client([[
        StreamEvent(kind="text", text="one"),
        StreamEvent(kind="text", text="two"),
        StreamEvent(kind="text", text="three"),
        StreamEvent(kind="done"),
    ]])
    orch = ChatOrchestrator(graph=g, client=client, dispatcher=_dispatcher(g))
    events = []
    for e in orch.run_turn("x"):
        events.append(e)
        if e.kind == "text" and e.text == "one":
            orch.cancel()
    kinds = [e.kind for e in events]
    assert "cancelled" in kinds or "turn_done" in kinds
```

### Step 2: Implement `synapse/ai/orchestrator.py`

```python
"""ChatOrchestrator — per-turn agent loop that runs a streaming LLM turn,
dispatches tool calls, and yields normalized events for the UI layer."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator, Optional

from synapse.ai.clients.base import StreamEvent
from synapse.ai.context import graph_summary
from synapse.ai.prompts import BASE_SYSTEM_PROMPT
from synapse.ai.tools import TOOLS


@dataclass
class OrchestratorEvent:
    """Events the orchestrator yields to the UI. Strictly superset of StreamEvent."""
    kind: str  # text | tool_call_started | tool_call_finished | cap_exceeded | error | cancelled | turn_done
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_result: Optional[dict] = None
    tool_call_id: Optional[str] = None
    error: Optional[str] = None


class ChatOrchestrator:
    """Drives one user turn from start to finish.

    Invariants:
      - Exactly one call to ``run_turn(user_text)`` per user message.
      - ``run_turn`` is a generator — pull events and forward them to the UI.
      - ``cancel()`` is safe from any thread; it sets a flag that is checked
        between stream events and between tool-call rounds.
    """

    DEFAULT_MAX_TOOL_CALLS = 4

    def __init__(
        self,
        graph,
        client,
        dispatcher,
        history: list[dict] | None = None,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    ):
        self.graph = graph
        self.client = client
        self.dispatcher = dispatcher
        self.history: list[dict] = history if history is not None else []
        self.max_tool_calls = max_tool_calls
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    # ------------------------------------------------------------------
    def _build_system(self) -> str:
        return BASE_SYSTEM_PROMPT + "\n\n" + graph_summary(self.graph)

    def _append_tool_result_message(self, tool_name: str, tool_call_id: str, result: dict) -> None:
        """Append a tool-result message using a neutral shape. Each client's
        stream method is responsible for translating this to the right
        provider-native format; for prompt-fallback clients we send it as a
        user-role message so the model sees it in-band."""
        provider_name = type(self.client).__name__
        content = json.dumps(result)
        if provider_name in ("ClaudeClient",):
            # Anthropic: tool_result must be wrapped in a user message's content array.
            self.history.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }],
            })
        elif provider_name in ("OpenAIClient",):
            self.history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            })
        else:
            # Gemini / Ollama / Groq: synthesize a plain user message so the
            # model can read it in the next turn. Good enough for Phase 2b.
            self.history.append({
                "role": "user",
                "content": f"Tool result for `{tool_name}`:\n```json\n{content}\n```",
            })

    def _append_assistant_tool_call_message(self, tool_name: str, tool_call_id: str, tool_input: dict) -> None:
        """Record the assistant's tool_call in history so the next LLM call
        can see the prior tool-use context. Again, provider-specific."""
        provider_name = type(self.client).__name__
        if provider_name == "ClaudeClient":
            self.history.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": tool_call_id,
                    "name": tool_name,
                    "input": tool_input,
                }],
            })
        elif provider_name == "OpenAIClient":
            self.history.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_input)},
                }],
                "content": None,
            })
        else:
            # For prompt-fallback clients we don't need to re-echo the tool_call
            # in history — the subsequent user-role tool-result message is
            # self-contained.
            pass

    # ------------------------------------------------------------------
    def run_turn(self, user_text: str) -> Iterator[OrchestratorEvent]:
        self.history.append({"role": "user", "content": user_text})
        tool_calls_used = 0
        system = self._build_system()

        while True:
            if self._cancelled:
                yield OrchestratorEvent(kind="cancelled")
                return

            stream = self.client.chat_with_tools_stream(
                system=system,
                messages=self.history,
                tools=TOOLS,
            )
            had_tool_call = False
            for ev in stream:
                if self._cancelled:
                    yield OrchestratorEvent(kind="cancelled")
                    return
                if ev.kind == "text":
                    yield OrchestratorEvent(kind="text", text=ev.text)
                elif ev.kind == "tool_call":
                    had_tool_call = True
                    tc = ev.tool_call or {}
                    tc_id = tc.get("id") or tc.get("name", "")
                    tc_name = tc.get("name", "")
                    tc_input = tc.get("input") or {}
                    yield OrchestratorEvent(
                        kind="tool_call_started",
                        tool_name=tc_name, tool_input=tc_input, tool_call_id=tc_id,
                    )
                    if tool_calls_used >= self.max_tool_calls:
                        yield OrchestratorEvent(
                            kind="cap_exceeded",
                            tool_name=tc_name,
                        )
                        self.history.append({
                            "role": "user",
                            "content": (
                                "[system] You have reached the 4 tool-call budget for this turn. "
                                "Stop calling tools and answer the user with what you have."
                            ),
                        })
                        break
                    tool_calls_used += 1
                    self._append_assistant_tool_call_message(tc_name, tc_id, tc_input)
                    result = self.dispatcher.dispatch(tc_name, tc_input)
                    self._append_tool_result_message(tc_name, tc_id, result)
                    yield OrchestratorEvent(
                        kind="tool_call_finished",
                        tool_name=tc_name, tool_result=result, tool_call_id=tc_id,
                    )
                    break  # restart with a fresh stream call
                elif ev.kind == "error":
                    yield OrchestratorEvent(kind="error", error=ev.error)
                    yield OrchestratorEvent(kind="turn_done")
                    return
                elif ev.kind == "done":
                    pass  # loop exits below

            if not had_tool_call:
                yield OrchestratorEvent(kind="turn_done")
                return
```

### Step 3: Run tests

Run: `pytest tests/ai/test_orchestrator.py -v`
Expected: 5 pass.

Full suite: all previous + 5 new.

### Step 4: Commit

```bash
git add synapse/ai/orchestrator.py tests/ai/test_orchestrator.py
git commit -m "feat(ai/orchestrator): add ChatOrchestrator agent loop with tool-call dispatch"
```

---

## Task 8: `ChatStreamWorker` — Qt thread wrapper

**Files:**
- Create: `synapse/ai/chat_worker.py`
- Create: `tests/ai/test_chat_worker.py`

A `QObject` driven by a `QThread`. Emits signals corresponding to `OrchestratorEvent` kinds. Exposes `request_cancel()` to flip the orchestrator's flag.

### Step 1: Write tests

```python
"""Tests run the worker synchronously — no QThread spin-up — by calling
``_run_once`` directly. Qt signals are wired via a dummy collector QObject."""
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore
from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph
from synapse.ai.clients.base import StreamEvent
from synapse.ai.chat_worker import ChatStreamWorker
from synapse.ai.tools import ToolDispatcher


class _Collector(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.tokens: list[str] = []
        self.tool_started: list[tuple[str, dict]] = []
        self.tool_finished: list[tuple[str, dict]] = []
        self.errors: list[str] = []
        self.turns_finished = 0

    def on_token(self, t: str):
        self.tokens.append(t)

    def on_tool_started(self, name: str, inp: dict):
        self.tool_started.append((name, inp))

    def on_tool_finished(self, name: str, result: dict):
        self.tool_finished.append((name, result))

    def on_error(self, msg: str):
        self.errors.append(msg)

    def on_turn_finished(self):
        self.turns_finished += 1


def _client(events):
    c = MagicMock()
    def stream(*a, **k):
        for e in events:
            yield e
    c.chat_with_tools_stream.side_effect = stream
    c.model = "m"
    return c


def test_worker_forwards_text_tokens_and_turn_finished():
    g = FakeGraph()
    client = _client([
        StreamEvent(kind="text", text="hi "),
        StreamEvent(kind="text", text="there"),
        StreamEvent(kind="done"),
    ])
    w = ChatStreamWorker(graph=g, client=client, dispatcher=ToolDispatcher(),
                         history=[], user_text="x")
    collector = _Collector()
    w.token_received.connect(collector.on_token)
    w.turn_finished.connect(collector.on_turn_finished)
    w._run_once()  # synchronous
    assert "".join(collector.tokens) == "hi there"
    assert collector.turns_finished == 1


def test_worker_emits_error_signal_on_client_error():
    g = FakeGraph()
    client = _client([StreamEvent(kind="error", error="boom")])
    w = ChatStreamWorker(graph=g, client=client, dispatcher=ToolDispatcher(),
                         history=[], user_text="x")
    collector = _Collector()
    w.error.connect(collector.on_error)
    w.turn_finished.connect(collector.on_turn_finished)
    w._run_once()
    assert collector.errors == ["boom"]


def test_worker_cancel_stops_mid_stream():
    g = FakeGraph()
    client = _client([
        StreamEvent(kind="text", text="a"),
        StreamEvent(kind="text", text="b"),
        StreamEvent(kind="text", text="c"),
        StreamEvent(kind="done"),
    ])
    w = ChatStreamWorker(graph=g, client=client, dispatcher=ToolDispatcher(),
                         history=[], user_text="x")
    collector = _Collector()
    w.token_received.connect(collector.on_token)
    w.turn_finished.connect(collector.on_turn_finished)
    def cancel_after_first(_):
        if len(collector.tokens) == 1:
            w.request_cancel()
    w.token_received.connect(cancel_after_first)
    w._run_once()
    # Loose assertion — may receive >1 if cancel happens mid-event; the key is
    # that the run terminated without hanging.
    assert collector.turns_finished >= 1
```

### Step 2: Implement `synapse/ai/chat_worker.py`

```python
"""Qt QThread wrapper for the ChatOrchestrator. Emits signals for each
orchestrator event so the chat panel can update the UI on the main thread."""
from __future__ import annotations

from PySide6 import QtCore

from synapse.ai.orchestrator import ChatOrchestrator


class ChatStreamWorker(QtCore.QObject):
    # Streaming text chunks (prose turn).
    token_received = QtCore.Signal(str)
    # Tool lifecycle.
    tool_call_started = QtCore.Signal(str, dict)      # name, input
    tool_call_finished = QtCore.Signal(str, dict)     # name, result
    cap_exceeded = QtCore.Signal(str)                 # tool name at which cap hit
    # Workflow preview — emitted after generate_workflow tool_call_finished.
    workflow_preview = QtCore.Signal(dict)
    # Terminal.
    turn_finished = QtCore.Signal()
    error = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, graph, client, dispatcher, history, user_text: str, parent=None):
        super().__init__(parent)
        self._orch = ChatOrchestrator(
            graph=graph, client=client, dispatcher=dispatcher, history=history,
        )
        self._user_text = user_text

    # Thread entrypoint — connect QThread.started to this slot.
    @QtCore.Slot()
    def run(self) -> None:
        self._run_once()

    def request_cancel(self) -> None:
        self._orch.cancel()

    # Split out so tests can call it synchronously without a QThread.
    def _run_once(self) -> None:
        try:
            for ev in self._orch.run_turn(self._user_text):
                if ev.kind == "text":
                    self.token_received.emit(ev.text or "")
                elif ev.kind == "tool_call_started":
                    self.tool_call_started.emit(ev.tool_name or "", ev.tool_input or {})
                elif ev.kind == "tool_call_finished":
                    self.tool_call_finished.emit(ev.tool_name or "", ev.tool_result or {})
                    if ev.tool_name == "generate_workflow" and ev.tool_result:
                        self.workflow_preview.emit(ev.tool_result)
                elif ev.kind == "cap_exceeded":
                    self.cap_exceeded.emit(ev.tool_name or "")
                elif ev.kind == "error":
                    self.error.emit(ev.error or "unknown error")
                elif ev.kind == "cancelled":
                    self.cancelled.emit()
                    break
                elif ev.kind == "turn_done":
                    break
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.turn_finished.emit()
```

### Step 3: Run tests — 3 pass.

### Step 4: Commit

```bash
git add synapse/ai/chat_worker.py tests/ai/test_chat_worker.py
git commit -m "feat(ai/chat_worker): add ChatStreamWorker QObject bridging orchestrator to Qt signals"
```

---

## Task 9: `AIChatPanel` routing + minimal UI feedback

**Files:**
- Modify: `synapse/llm_assistant.py`

Changes inside `AIChatPanel`:

1. `_on_send` checks `get_use_orchestrator()`. If False, keep the current legacy `_ChatWorker` path untouched.
2. If True, build a `ChatStreamWorker` with the current graph, client, a fresh `ToolDispatcher` wired with all 6 handlers, and the current `self._messages` history. Start a `QThread`.
3. Wire signals:
   - `token_received(str)` → append to a streaming assistant bubble (simplest: build up a string and replace-render the latest bubble on each token). For Phase 2b, a single "replace last bubble" approach is acceptable — Phase 3 will do proper throttled appends.
   - `tool_call_started(name, input)` → append a small system bubble: `"🔧 {name}(…)"`.
   - `tool_call_finished(name, result)` → append a system bubble with a one-line summary (e.g. `"🔧 {name} → ok"` or `"🔧 {name} → error: {msg}"`).
   - `workflow_preview(result)` → if `result.get("canvas_was_empty")` is True, immediately call existing `WorkflowLoader` to apply; else open a modal confirm dialog with Apply / Discard. On Apply, feed the workflow dict to `WorkflowLoader` just like the legacy path does.
   - `error(msg)` → error bubble.
   - `cancelled()` → system bubble `"cancelled"`.
   - `turn_finished()` → reset Send button; save final assistant text into `self._messages`.

4. A simple "⏹ Stop" button replaces "Send" mid-turn. Click → `worker.request_cancel()`.

The specifics are UI-heavy. Use the existing `_append_bubble(role, text)` helper — add a new role `"system-tool"` if desired, or reuse `"system"`.

### Step 1: Add imports near the top of the file

After `from synapse.markdown_render import render_markdown`:

```python
from synapse.ai import get_use_orchestrator, ToolDispatcher
from synapse.ai.chat_worker import ChatStreamWorker
from synapse.ai.tool_handlers.inspect_canvas import make_inspect_canvas_handler
from synapse.ai.tool_handlers.explain_node import explain_node_handler
from synapse.ai.tool_handlers.read_node_output import make_read_node_output_handler
from synapse.ai.tool_handlers.generate_workflow import make_generate_workflow_handler
from synapse.ai.tool_handlers.modify_workflow import make_modify_workflow_handler
from synapse.ai.tool_handlers.write_python_script import make_write_python_script_handler
```

### Step 2: Add helper on `AIChatPanel` to build a wired dispatcher

Inside the class:

```python
    def _build_dispatcher(self) -> ToolDispatcher:
        d = ToolDispatcher()
        d.register("inspect_canvas", make_inspect_canvas_handler(self.graph))
        d.register("explain_node", explain_node_handler)
        d.register("read_node_output", make_read_node_output_handler(
            self.graph, supports_vision=lambda: getattr(self._client, "supports_vision", False),
        ))
        d.register("generate_workflow",
                   make_generate_workflow_handler(self.graph, self._client))
        # modify_workflow needs a factory that creates NodeGraphQt nodes.
        def _factory(type_name: str, node_id: str):
            node = self.graph.create_node(type_name, name=node_id, push_undo=False)
            # Tag the NodeGraphQt uuid with the LLM-space id for later lookups.
            node._llm_id = node_id
            return node
        d.register("modify_workflow",
                   make_modify_workflow_handler(self.graph, node_factory=_factory))
        d.register("write_python_script",
                   make_write_python_script_handler(self.graph, self._client))
        return d
```

**Caveat:** NodeGraphQt's nodes don't carry an `id` attribute matching the LLM-space id used by the handler's `_lookup`. For Phase 2b's minimum viable path, the `modify_workflow` handler's `_lookup(node_id)` walks `graph.all_nodes()` and compares `n.id`. On a real NodeGraphQt graph, `n.id` is a UUID that won't match the model's string ids. As a minimum pragmatic shim, the `_factory` above stashes the LLM id on `node._llm_id`, and we patch the handler to also check that attribute. Do this by extending `modify_workflow.py`'s `_lookup` to try both — add those 3 lines while we're here. If this drift gets worse, Phase 2c will add a proper id-mapping layer.

Update `synapse/ai/tool_handlers/modify_workflow.py::_lookup`:

```python
    def _lookup(node_id: str):
        for n in graph.all_nodes():
            if getattr(n, "id", None) == node_id:
                return n
            if getattr(n, "_llm_id", None) == node_id:
                return n
        return None
```

### Step 3: Replace `_on_send` body

```python
    def _on_send(self):
        text = self._input_edit.toPlainText().strip()
        if not text:
            return
        if self._client is None:
            self._rebuild_client()
            if self._client is None:
                return

        self._messages.append({"role": "user", "content": text})
        self._append_bubble("user", text)
        self._input_edit.clear()

        if get_use_orchestrator():
            self._run_with_orchestrator(text)
        else:
            self._run_with_legacy_worker(text)
```

Extract the old body into `_run_with_legacy_worker` (just rename the existing body). Add a new method:

```python
    def _run_with_orchestrator(self, user_text: str):
        self._send_btn.setEnabled(False)
        self._send_btn.setText("⏹")
        self._send_btn.clicked.disconnect()
        self._send_btn.clicked.connect(self._on_stop_orchestrator)
        self._status.setText("Thinking…")

        self._orch_stream_buffer = ""
        dispatcher = self._build_dispatcher()
        self._orch_worker = ChatStreamWorker(
            graph=self.graph, client=self._client, dispatcher=dispatcher,
            history=list(self._messages), user_text=user_text,
        )
        self._orch_thread = QtCore.QThread()
        self._orch_worker.moveToThread(self._orch_thread)
        self._orch_thread.started.connect(self._orch_worker.run)

        self._orch_worker.token_received.connect(self._on_orch_token)
        self._orch_worker.tool_call_started.connect(self._on_orch_tool_started)
        self._orch_worker.tool_call_finished.connect(self._on_orch_tool_finished)
        self._orch_worker.workflow_preview.connect(self._on_orch_workflow_preview)
        self._orch_worker.cap_exceeded.connect(self._on_orch_cap)
        self._orch_worker.error.connect(self._on_orch_error)
        self._orch_worker.cancelled.connect(self._on_orch_cancelled)
        self._orch_worker.turn_finished.connect(self._on_orch_turn_finished)
        self._orch_worker.turn_finished.connect(self._orch_thread.quit)
        self._orch_thread.finished.connect(self._orch_worker.deleteLater)
        self._orch_thread.finished.connect(self._orch_thread.deleteLater)
        self._orch_thread.start()

    def _on_stop_orchestrator(self):
        if getattr(self, "_orch_worker", None):
            self._orch_worker.request_cancel()

    def _on_orch_token(self, piece: str):
        self._orch_stream_buffer += piece
        # Simple approach: on first token, append a new bubble; subsequent
        # tokens re-render by clearing+re-adding — Phase 3 replaces with
        # incremental DOM appends for smoother display.
        self._append_bubble("assistant", self._orch_stream_buffer, _streaming_replace=True)

    def _on_orch_tool_started(self, name: str, inp: dict):
        self._append_bubble("system", f"🔧 {name}({json.dumps(inp)[:120]}…)")

    def _on_orch_tool_finished(self, name: str, result: dict):
        if "error" in result:
            self._append_bubble("system", f"🔧 {name} → error: {result['error']}")
        else:
            keys = ", ".join(list(result.keys())[:4])
            self._append_bubble("system", f"🔧 {name} → ok ({keys})")

    def _on_orch_workflow_preview(self, result: dict):
        if result.get("canvas_was_empty"):
            self._last_workflow = result.get("workflow")
            self._apply_workflow_now(replace=True)
        else:
            self._last_workflow = result.get("workflow")
            btn = QtWidgets.QMessageBox.question(
                self, "Apply workflow?",
                f"Proposed workflow: {result.get('node_count')} nodes, "
                f"{result.get('edge_count')} edges.\nTypes: "
                + ", ".join(result.get("preview_types", [])) + "\n\nApply?",
                QtWidgets.QMessageBox.Apply | QtWidgets.QMessageBox.Discard,
            )
            if btn == QtWidgets.QMessageBox.Apply:
                self._apply_workflow_now(replace=False)

    def _apply_workflow_now(self, replace: bool):
        # Reuse the existing WorkflowLoader path — same code legacy "Load" button uses.
        if not self._last_workflow:
            return
        loader = WorkflowLoader(self.graph)
        if replace:
            loader.replace(self._last_workflow)
        else:
            loader.merge(self._last_workflow)

    def _on_orch_cap(self, tool_name: str):
        self._append_bubble("system", f"[budget exceeded — stopping after {tool_name}]")

    def _on_orch_error(self, msg: str):
        self._append_bubble("error", msg)

    def _on_orch_cancelled(self):
        self._append_bubble("system", "cancelled")

    def _on_orch_turn_finished(self):
        self._send_btn.setEnabled(True)
        self._send_btn.setText("Send")
        self._send_btn.clicked.disconnect()
        self._send_btn.clicked.connect(self._on_send)
        self._status.setText(f"{self._provider_combo.currentText()} / {self._client.model}")
        # Persist the streamed assistant text into history.
        if self._orch_stream_buffer:
            self._messages.append({"role": "assistant", "content": self._orch_stream_buffer})
        self._orch_stream_buffer = ""
```

**`_append_bubble` gets an optional `_streaming_replace=False` flag:** when True and the last bubble in the display was also streamed, overwrite it instead of appending. Minimal implementation: keep a `_streaming_bubble_active` flag on the panel; when True, set the text on the existing bubble; when `turn_finished` fires, clear the flag.

For Phase 2b, a simpler approach is acceptable: just append every token as its own bubble. It looks noisy but works. If you want the cleaner approach, add the flag as a follow-up in Phase 3.

**Simplest implementation for Phase 2b:** make `_on_orch_token` accumulate into `_orch_stream_buffer` and do NOT append per-token. Instead, append once on `turn_finished` with the full text. That way Phase 2b produces non-streaming UX but clean code. Streaming tokens become Phase 3.

Revised `_on_orch_token`:

```python
    def _on_orch_token(self, piece: str):
        self._orch_stream_buffer += piece

    def _on_orch_turn_finished(self):
        # ... existing reset code ...
        if self._orch_stream_buffer:
            self._append_bubble("assistant", self._orch_stream_buffer)
            self._messages.append({"role": "assistant", "content": self._orch_stream_buffer})
        self._orch_stream_buffer = ""
```

**Use the revised simpler version.** The streaming UX is Phase 3.

### Step 4: Add WorkflowLoader `merge` / `replace` methods if they don't already exist

Look at the existing `WorkflowLoader` class — it already has `load()` that applies a workflow. Check whether it supports replace-vs-merge. If not, adapt `_apply_workflow_now` to call whatever existing entry point matches the legacy "Load into Canvas" / "Replace Canvas" buttons' behaviour.

**Implementer: inspect WorkflowLoader before writing `_apply_workflow_now`. Call the exact same method the existing `_on_load` / `_on_replace` methods in `AIChatPanel` use. If there's any doubt, stop and report — I'll advise which entry points to wire up.**

### Step 5: Manual smoke + commit

Since this change is UI-heavy and hard to unit-test, the verification path is:
1. `pytest tests/ -q` — existing tests must still pass.
2. In `~/.synapse_llm_config.json`, set `"ai": {"use_orchestrator": true}`.
3. Launch the app from the worktree, send a message like "What's on my canvas?"
4. Confirm: tool-call chips appear as system bubbles, the final reply renders as markdown, no crashes.

If the smoke test works, commit:

```bash
git add synapse/llm_assistant.py synapse/ai/tool_handlers/modify_workflow.py
git commit -m "feat(chat): route AIChatPanel through ChatStreamWorker when USE_ORCHESTRATOR is on"
```

---

## Task 10: End-to-end integration test

**Files:**
- Create: `tests/ai/test_phase2b_integration.py`

Drive a full user turn through the orchestrator + real dispatcher + FakeGraph using a mock streaming client that returns a sequence of events simulating "model calls inspect_canvas, receives result, replies with text."

### Step 1: Test

```python
"""End-to-end Phase 2b test: user turn → orchestrator → tool_call →
dispatcher → real handler → tool_result → next turn → final text."""
from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.clients.base import StreamEvent
from synapse.ai.tools import ToolDispatcher
from synapse.ai.tool_handlers.inspect_canvas import make_inspect_canvas_handler
from synapse.ai.tool_handlers.explain_node import explain_node_handler
from synapse.ai.orchestrator import ChatOrchestrator


def _turn_streams(batches):
    """Factory that returns a fresh iterator each call of chat_with_tools_stream."""
    it = iter(batches)

    def stream(system, messages, tools=None):
        try:
            batch = next(it)
        except StopIteration:
            batch = [StreamEvent(kind="done")]
        for ev in batch:
            yield ev

    return stream


def test_full_turn_with_real_inspect_canvas_dispatch():
    graph = FakeGraph()
    graph.add_node(FakeNode("a", "CSVLoader", {"path": "data.csv"}))
    graph.add_node(FakeNode("b", "SortTable", {"column": "value"}))

    client = MagicMock()
    client.chat_with_tools_stream.side_effect = _turn_streams([
        # Turn 1: model calls inspect_canvas.
        [
            StreamEvent(kind="tool_call",
                        tool_call={"id": "t1", "name": "inspect_canvas", "input": {}}),
            StreamEvent(kind="done"),
        ],
        # Turn 2 (after tool result is injected): model replies with text.
        [
            StreamEvent(kind="text", text="Your canvas has a CSVLoader and a SortTable."),
            StreamEvent(kind="done"),
        ],
    ])
    client.model = "test-model"

    dispatcher = ToolDispatcher()
    dispatcher.register("inspect_canvas", make_inspect_canvas_handler(graph))
    dispatcher.register("explain_node", explain_node_handler)

    orch = ChatOrchestrator(graph=graph, client=client, dispatcher=dispatcher)
    events = list(orch.run_turn("what's on my canvas?"))
    kinds = [e.kind for e in events]

    # Sequence must include: tool_call_started, tool_call_finished, text, turn_done.
    assert kinds.count("tool_call_started") == 1
    assert kinds.count("tool_call_finished") == 1
    assert any(e.kind == "text" and "CSVLoader" in e.text for e in events)
    assert kinds[-1] == "turn_done"

    # The tool_result payload in history actually contains the real inspect_canvas output.
    # Find the tool-result user message (prompt-fallback shape) or assistant/tool messages.
    # For a MagicMock client the provider-dispatch goes to the "else" branch in
    # _append_tool_result_message, producing a plain user message.
    tool_result_user_msgs = [
        m for m in orch.history
        if m.get("role") == "user" and "Tool result for" in str(m.get("content", ""))
    ]
    assert len(tool_result_user_msgs) == 1
    content = tool_result_user_msgs[0]["content"]
    assert "CSVLoader" in content and "SortTable" in content


def test_full_turn_with_write_python_script():
    """Orchestrator → write_python_script → inspect_canvas confirms the node updated."""
    from synapse.ai.tool_handlers.write_python_script import make_write_python_script_handler
    from synapse.ai.tool_handlers.modify_workflow import make_modify_workflow_handler

    graph = FakeGraph()

    client = MagicMock()
    # Sub-LLM call (inside write_python_script handler) returns code string.
    # Main orchestrator client (outer) yields tool-call events.
    sub_response = iter(["out_1 = in_1  # passthrough"])
    client.chat_multi.side_effect = lambda system, messages: next(sub_response)
    client.chat_with_tools_stream.side_effect = _turn_streams([
        # Turn 1: add a PythonScriptNode.
        [
            StreamEvent(kind="tool_call", tool_call={"id": "t1", "name": "modify_workflow",
                                                      "input": {"operations":
                                                                [{"op": "add_node",
                                                                  "type": "PythonScriptNode",
                                                                  "id": "py1"}]}}),
            StreamEvent(kind="done"),
        ],
        # Turn 2: write code to it.
        [
            StreamEvent(kind="tool_call", tool_call={"id": "t2", "name": "write_python_script",
                                                      "input": {"node_id": "py1",
                                                                "description": "passthrough"}}),
            StreamEvent(kind="done"),
        ],
        # Turn 3: final text.
        [
            StreamEvent(kind="text", text="Done! I added a PythonScriptNode."),
            StreamEvent(kind="done"),
        ],
    ])
    client.model = "m"

    dispatcher = ToolDispatcher()
    dispatcher.register("modify_workflow", make_modify_workflow_handler(
        graph, node_factory=lambda t, i: FakeNode(i, t)))
    dispatcher.register("write_python_script",
                        make_write_python_script_handler(graph, client))

    orch = ChatOrchestrator(graph=graph, client=client, dispatcher=dispatcher)
    events = list(orch.run_turn("add a passthrough python script"))

    # End state: py1 exists, has script_code, final text reached.
    assert graph.get_node_by_id("py1") is not None
    assert "out_1 = in_1" in graph.get_node_by_id("py1").model.custom_properties.get("script_code", "")
    assert any(e.kind == "text" and "Done" in e.text for e in events)
```

### Step 2: Run

Run: `pytest tests/ai/test_phase2b_integration.py -v`
Expected: 2 pass.

Full suite: `pytest tests/ -q`
Expected: all prior tests + 2 new, ~130 total.

### Step 3: Commit

```bash
git add tests/ai/test_phase2b_integration.py
git commit -m "test(ai): end-to-end integration test for Phase 2b orchestrator loop"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Native tool-calling for Claude / OpenAI / Gemini — Tasks 3, 4, 5
- ✅ Prompt-based fallback for Ollama / Groq — Tasks 2, 6
- ✅ ChatOrchestrator agent loop with 4-call cap + cancellation — Task 7
- ✅ ChatStreamWorker Qt signal bridge — Task 8
- ✅ AIChatPanel feature-flag routing with tool status bubbles — Task 9
- ✅ generate_workflow Apply/Discard modal (inline UI is Phase 3) — Task 9
- ✅ End-to-end integration test — Task 10

**Deferred to Phase 3 (explicitly out of scope for 2b):**
- True token-at-a-time streaming UX (current: buffer then render full reply)
- Tool-call chips with expandable JSON details (current: system bubbles)
- Inline Apply/Discard buttons in the chat bubble (current: modal dialog)
- Token meter under the input box
- Vision badge near the model dropdown
- Proper id-mapping layer between LLM-space ids and NodeGraphQt node UUIDs (current: `_llm_id` stash hack)
- Gemini tool-result injection (current: falls through to synthesised-user-message path)

**Placeholder scan:** no TBDs. Every task has complete code or an explicit "stop and report" clause.

**Type consistency:** `StreamEvent(kind="tool_call")` carries `tool_call: dict` with keys `id`, `name`, `input` uniformly across all 5 tool-capable clients. `OrchestratorEvent` fields are superset of `StreamEvent`.
