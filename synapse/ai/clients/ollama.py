"""Ollama client (local + cloud via base_url + api_key)."""
from __future__ import annotations

import json
import platform
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
        """Stream a plain-text turn. Tools arg is ignored in Phase 1."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
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
                        yield StreamEvent(kind="text", text=piece)
                    if obj.get("done"):
                        break
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
