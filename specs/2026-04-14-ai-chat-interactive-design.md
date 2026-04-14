# AI Chat — Interactive, Tool-Using Assistant

**Date:** 2026-04-14
**Scope:** Rework the Synapse AI chat panel from a JSON-only workflow generator into a conversational, tool-using assistant.

## Problem

`AIChatPanel` currently replies only with raw JSON workflow templates produced by `TwoPassLLMWorker`. Users cannot ask questions, get explanations, debug their canvas, or request Python scripts. Any non-workflow intent falls through as plain text with no rich rendering.

## Goals

- The AI chats normally by default (markdown, explanations, follow-ups).
- Workflow generation becomes **one tool among several**, not the default response.
- The AI can build workflows, edit the current canvas incrementally, write Python for `PythonScriptNode`, inspect the canvas, explain individual nodes, and read node outputs (with image thumbnails on vision-capable models).
- Token usage stays bounded through tiered context, history rollup, and hard budgets.
- Responses stream for prose turns; tool-call turns remain blocking.
- All six existing providers (Ollama, Ollama Cloud, OpenAI, Claude, Groq, Gemini) are supported.

## Non-Goals (v1)

- Multi-file or multi-script generation in a single tool call.
- Re-rolling or editing a prior assistant turn.
- Long-running / asynchronous tools with background progress.
- Replacing the existing `WorkflowLoader` auto-layout logic — it is reused inside the `generate_workflow` tool.

## Architecture Overview

A new `ChatOrchestrator` sits between `AIChatPanel` and the LLM clients and runs an agent loop per user turn:

1. Assemble request: base system prompt + tool schemas + terse canvas summary + rolled-up history + current user message.
2. Call the active client's `chat_with_tools_stream(...)`.
3. If response is **prose** → stream tokens into the in-progress assistant bubble and finish the turn.
4. If response is a **tool call** → dispatch via `ToolDispatcher`, append the result as a tool message, loop back to step 2.
5. Hard cap of 4 tool calls per user turn. On cap: inject a system note and force a prose turn.

Native tool-calling APIs are used where available (Claude, OpenAI, Gemini). Ollama and Groq use a prompt-based fallback protocol: the system prompt instructs the model to emit `<tool_call>{...}</tool_call>` markers, which the orchestrator parses out of the stream.

`TwoPassLLMWorker` and `WorkflowLoader` are preserved: the `generate_workflow` tool handler calls them internally. No rewrite of the selection → detailed-cards flow.

## Tool Set

Six tools, shared across native and fallback code paths.

### 1. `generate_workflow`

- **Input:** `{goal: str, constraints?: str}`
- **Behavior:** runs existing two-pass selection → detailed cards → JSON flow. Does **not** auto-apply on a non-empty canvas (inline Apply/Discard preview). Auto-applies silently when canvas is empty.
- **Output:** `{node_count, edge_count, preview: [node_types], workflow_id}`

### 2. `modify_workflow`

- **Input:** `{operations: [Op]}` where `Op ∈ {add_node, remove_node, connect, disconnect, set_prop}`
- **Behavior:** applies ops directly to the canvas inside a single NodeGraphQt undo group. Partial success allowed — failed ops are reported; applied ops stay.
- **Output:** `{applied: [...], failed: [{op, reason}]}`

### 3. `write_python_script`

- **Input:**
  ```json
  {
    "node_id": "string (optional; target an existing PythonScriptNode)",
    "description": "string",
    "n_inputs": "int 1..8",
    "n_outputs": "int 1..8",
    "input_hints":  "[{port, kind, schema?}]",
    "output_hints": "[{port, kind}]"
  }
  ```
- **Behavior:**
  1. Resolve `node_id` to a `PythonScriptNode`; refuse if type is wrong.
  2. Set `n_inputs` / `n_outputs` props first — triggers existing `_sync_ports` logic.
  3. Run a **sub-LLM call** with a specialized system prompt containing the node's docstring (the authoritative API: `in_1`, `out_1`, pre-imports `pd/np/scipy/skimage/cv2/PIL/plt`, type wrappers, `set_progress`) plus the user's description and port hints. Sub-prompt forbids markdown fences and requires assignment to every declared `out_N`.
  4. Set `node.set_property('script_code', code, push_undo=True)`.
- **Output:** `{target_node_id, line_count, assigned_outputs: ["out_1", ...]}`
- **Never executes the code.**

### 4. `inspect_canvas`

- **Input:** `{node_ids?: [str], include_props?: bool}`
- **Output:** full nodes/edges/props for requested nodes (or all). Capped at ~2k tokens with a `[truncated]` marker if over.

### 5. `explain_node`

- **Input:** `{node_type: str}`
- **Output:** the node's detailed catalog card — ports, props, docstring. Same data built by `build_detailed_cards()` in the existing Pass 2.

### 6. `read_node_output`

- **Input:** `{node_id: str}`
- **Output:** `{kind, metadata, text_preview, thumbnail?}`
  - Images: shape, dtype, min/max, NaN count; text-only on non-vision clients. A 256-pixel PNG thumbnail is included **only when `client.supports_vision` is true**.
  - Tables: shape, `head(10)` as markdown, basic describe.
  - Errors: error message.
- Vision capability is a hardcoded per-model flag in each client adapter; user can override in settings.

## Context & History

Assembled fresh per turn by `ChatOrchestrator.build_request()`:

1. **Base system prompt** (~400 tokens, static): role, tool-use rules, markdown guidance, the rule *never dump raw JSON to the user — use tools instead*.
2. **Tool schemas** (~600 tokens): passed via SDK `tools=[...]` for native clients, appended to the system prompt (with the `<tool_call>…</tool_call>` protocol description) for fallback clients.
3. **Canvas summary line** (~50–150 tokens): built by a new `graph_summary()` helper that walks the node graph. Example: `"Canvas: 4 nodes — CSVLoader_1 → ParticleProps_2 → SortTable_3 → TopN_4. 1 error on ParticleProps_2."`
4. **Rolled-up history** (dynamic, capped, see below).
5. **Current user message.**

### History rollup

- Keep the last **8 turns** verbatim. A turn = user message + assistant reply + any tool-call round-trips.
- Older turns collapse into one synthetic system message (`"Earlier in this session the user: <bullets>. Key decisions: <bullets>."`), generated by a cheap call to the same provider the first time the window slides, cached until it slides again.
- **Tool-call results older than 2 turns** are truncated to their summary line (`{applied: 3 ops}`). Full JSON drops out — the single biggest token saving.
- **Image thumbnails from `read_node_output`** are kept only for the turn they were returned in; prior-turn thumbnails become the text placeholder `"[image thumbnail from earlier turn]"`.

### Hard budget

Estimated tokens computed before sending. At >80% of the model's context window, an extra rollup pass compresses turns 5–8. Still over → oldest turns drop with a `[truncated]` marker. Budget shown in the UI.

## Streaming & UI

### Client streaming interface

```python
class LLMClient:
    def chat_with_tools_stream(self, system, messages, tools) -> Iterator[StreamEvent]: ...
```

`StreamEvent` is `{kind: "text"|"tool_call"|"done"|"error", text?, tool_call?, error?}`.

- **Prose turns** emit `text` events as tokens arrive.
- **Tool-call turns** emit one `tool_call` event once the tool-use block is fully received, then `done`. Per the agreed scope, tool-call payloads are **not** streamed partial.
- Fallback clients buffer output looking for `<tool_call>`; if seen, buffer through the closing tag, emit one `tool_call`, discard any trailing text.

### Qt threading

A new `ChatStreamWorker(QThread)` runs the orchestrator and emits:

- `token_received(str)` — append to current assistant bubble.
- `tool_call_started(name, input_dict)` / `tool_call_finished(name, result_summary)` — update inline chips.
- `turn_finished(final_markdown)` — close the bubble.
- `error(msg)` — red bubble.
- `token_usage(per_turn, session)` — update the meter.

### `AIChatPanel` changes

1. **Markdown rendering.** New `markdown_render.py` using the `markdown` library (fallback `mistune`) with Pygments for code. Existing bubble wrapper (blue user / gray assistant) stays; only inner content changes to rendered HTML.
2. **Streaming bubble.** Empty assistant bubble created on turn start; token appends buffer and the bubble re-renders at ~50 ms throttle. Final render on `turn_finished`.
3. **Tool-call chips.** Small pills at the top of an assistant bubble: `🔧 inspect_canvas → 4 nodes`. Clicking a chip expands a collapsible `<details>` with the full JSON.
4. **Inline Apply/Discard preview** for `generate_workflow` on a non-empty canvas. Auto-applies silently on empty canvas.
5. **Token meter** under the input: `"this turn: ~3.2k / session: 24k / model limit: 200k"`.
6. **Stop button** next to Send (visible mid-turn). Sets a cancel flag checked between stream chunks and between tool-call iterations. Cancel never leaves a tool call half-applied.
7. **Vision badge** near the model dropdown: `👁 vision` or muted `text-only` based on `client.supports_vision`.

## Safety Rails

- **Tool-call cap:** max 4 per user turn.
- **`modify_workflow` atomicity:** single undo group; partial success surfaced to the LLM.
- **`generate_workflow` on non-empty canvas:** never auto-applies.
- **`write_python_script`:** refuses when `node_id` is not a `PythonScriptNode`; never executes code.
- **Duplicate-call short-circuit:** identical tool call with identical input twice in a row returns the cached result (loop guard).
- **Stop button:** cancels cleanly between tool calls, never mid-call.

## File Layout

```
synapse/
  llm_assistant.py          (existing; pruned to AIChatPanel + client registry)
  ai/
    orchestrator.py         (new: ChatOrchestrator, ChatStreamWorker)
    tools.py                (new: TOOLS schema list + ToolDispatcher)
    context.py              (new: graph_summary, history rollup, token estimator)
    prompts.py              (new: base system prompt + write_python_script sub-prompt)
    clients/
      base.py               (new: LLMClient abstract + StreamEvent dataclass)
      ollama.py             (split out; streaming + fallback tool protocol)
      openai.py             (split out; streaming + native tools)
      claude.py             (split out; streaming + native tools)
      gemini.py             (split out; streaming + native tools)
      groq.py               (split out; streaming + fallback tool protocol)
  markdown_render.py        (new: markdown → HTML with Pygments)
```

`TwoPassLLMWorker` and `WorkflowLoader` stay in `llm_assistant.py`; `generate_workflow`'s handler imports and reuses them.

## Rollout (Phased)

- **Phase 1 — Plumbing.** Split clients into `ai/clients/`. Add `chat_with_tools_stream` (text-only for now). Add markdown rendering to bubbles. Existing JSON-reply flow keeps working.
- **Phase 2 — Orchestrator & tools.** Build `ChatOrchestrator`, all 6 tools, native + fallback tool protocols. Feature flag `USE_ORCHESTRATOR` in settings for opt-in.
- **Phase 3 — UI polish.** Streaming bubbles, tool-call chips, inline Apply/Discard, token meter, Stop button, vision badge.
- **Phase 4 — Drop legacy.** Remove feature flag and old `TwoPassLLMWorker.run()` path; its logic survives inside `generate_workflow`.

## Testing Strategy

- **Unit:** tool dispatcher handlers (mock graph); context builder (fixed inputs → expected token count); fallback protocol parser (various malformed inputs).
- **Integration:** one test per provider with a stub endpoint — verify streaming events, tool-call round-trips, cancel behavior.
- **Manual smoke tests:** scripted prompts covering each tool on each provider (a small matrix in the repo docs).

## Open Questions

None at time of writing; all clarifiers answered during brainstorm.
