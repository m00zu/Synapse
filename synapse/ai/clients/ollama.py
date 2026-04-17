"""Ollama client (local + cloud via base_url + api_key)."""
from __future__ import annotations

import json
import platform
import time
import requests
from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent
from synapse.ai.schema import RESPONSE_SCHEMA


# Ollama Cloud's WAF rejects the default `python-requests/x.y.z` User-Agent
# with 401 on /api/chat (while /api/tags still works). Mirror the headers
# sent by the official `ollama` Python SDK so cloud requests pass through.
_UA = (
    f"synapse-ollama/1 ({platform.machine()} {platform.system().lower()}) "
    f"Python/{platform.python_version()}"
)

# Ollama Cloud's free-tier inference nodes return transient 500s at large
# system prompts — same request often succeeds on a retry. Back off briefly
# and try twice before giving up.
_RETRY_DELAYS = (1.5, 4.0)


def _post_with_retry(url: str, *, headers: dict, json_payload: dict,
                     stream: bool, timeout: float):
    """POST with backoff on 5xx. Returns the final Response; caller decides
    whether to raise_for_status."""
    last = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        resp = requests.post(url, headers=headers, json=json_payload,
                             stream=stream, timeout=timeout)
        last = resp
        # Retry only on server errors (500-599), never on 4xx (auth, bad input).
        if resp.status_code < 500:
            return resp
        if attempt >= len(_RETRY_DELAYS):
            return resp
        resp.close()
        time.sleep(_RETRY_DELAYS[attempt])
    return last


class OllamaClient(LLMClient):
    DEFAULT_MODEL    = "gemma3:12b"
    DEFAULT_BASE_URL = "http://localhost:11434"
    CLOUD_BASE_URL   = "https://ollama.com"

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.api_key  = api_key

    def _headers(self) -> dict:
        h = {
            "User-Agent": _UA,
            "Accept": "application/json",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5,
                                headers=self._headers())
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        """Legacy non-streaming single-turn (kept for backward compat — JSON format)."""
        user_msg: dict = {"role": "user", "content": user}
        if images:
            user_msg["images"] = images
        payload = {
            "model":   self.model,
            "messages": [
                {"role": "system",  "content": system},
                user_msg,
            ],
            "stream":  False,
            "options": {"temperature": 0.1},
        }
        if not self.api_key:
            payload["format"] = RESPONSE_SCHEMA
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers=self._headers(),
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def chat_multi(self, system: str, messages: list[dict]) -> str:
        """Legacy non-streaming multi-turn."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat", json=payload,
            headers=self._headers(), timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

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
            resp = _post_with_retry(
                f"{self.base_url}/api/chat",
                headers=self._headers(),
                json_payload=payload,
                stream=True,
                timeout=120,
            )
            with resp:
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
