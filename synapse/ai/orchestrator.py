"""ChatOrchestrator — per-turn agent loop that runs a streaming LLM turn,
dispatches tool calls, and yields normalized events for the UI layer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Optional

from synapse.ai.clients.base import StreamEvent
from synapse.ai.context import graph_summary
from synapse.ai.prompts import BASE_SYSTEM_PROMPT
from synapse.ai.tools import TOOLS


@dataclass
class OrchestratorEvent:
    """Events the orchestrator yields to the UI. Strictly superset of StreamEvent."""
    kind: str  # text | tool_call_started | tool_call_finished | cap_exceeded |
               # error | cancelled | turn_done
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

    Tool-result messages follow provider-specific conventions, chosen by the
    client's class name:
      - ClaudeClient : user role with content: [{type:tool_result, ...}]
      - OpenAIClient : tool role with tool_call_id + content JSON
      - others       : plain user message with inline JSON (Gemini/Ollama/Groq)
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

    def _append_assistant_tool_call_message(self, tool_name: str, tool_call_id: str, tool_input: dict) -> None:
        """Echo the assistant's tool_call into history so the next LLM call
        sees the prior tool-use context. Provider-specific."""
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
        # Prompt-fallback / Gemini: no explicit echo needed; the subsequent
        # user-role tool-result message is self-contained.

    def _append_tool_result_message(self, tool_name: str, tool_call_id: str, result: dict) -> None:
        """Inject the tool's result using each provider's expected shape."""
        provider_name = type(self.client).__name__
        content = json.dumps(result)
        if provider_name == "ClaudeClient":
            self.history.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }],
            })
        elif provider_name == "OpenAIClient":
            self.history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            })
        else:
            # Gemini / Ollama / Groq fallback — inline as a user message.
            self.history.append({
                "role": "user",
                "content": f"Tool result for `{tool_name}`:\n```json\n{content}\n```",
            })

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
            cap_hit = False
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

                    tool_calls_used += 1
                    if tool_calls_used > self.max_tool_calls:
                        cap_hit = True
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

                    yield OrchestratorEvent(
                        kind="tool_call_started",
                        tool_name=tc_name, tool_input=tc_input, tool_call_id=tc_id,
                    )
                    self._append_assistant_tool_call_message(tc_name, tc_id, tc_input)
                    result = self.dispatcher.dispatch(tc_name, tc_input)
                    self._append_tool_result_message(tc_name, tc_id, result)
                    yield OrchestratorEvent(
                        kind="tool_call_finished",
                        tool_name=tc_name, tool_result=result, tool_call_id=tc_id,
                    )
                    break  # restart the loop with a fresh stream call
                elif ev.kind == "error":
                    yield OrchestratorEvent(kind="error", error=ev.error)
                    yield OrchestratorEvent(kind="turn_done")
                    return
                elif ev.kind == "done":
                    pass  # natural end; outer loop exits if had_tool_call is False

            if not had_tool_call or cap_hit:
                yield OrchestratorEvent(kind="turn_done")
                return
