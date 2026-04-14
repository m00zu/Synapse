"""Google Gemini client."""
from __future__ import annotations

import json
import requests
from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


class GeminiClient(LLMClient):
    DEFAULT_MODEL = "gemini-2.5-flash-lite"   # fallback; real list populated via Refresh
    BASE_URL      = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model   = model

    def list_models(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            resp = requests.get(
                f"{self.BASE_URL}/models",
                params={"key": self.api_key},
                timeout=5,
            )
            resp.raise_for_status()
            return sorted(
                m["name"].replace("models/", "")
                for m in resp.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            )
        except Exception:
            return []

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        parts = [{"text": user}]
        if images:
            for b64 in images:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64,
                    }
                })
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }
        resp = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "unknown")
            raise ValueError(f"Gemini returned no candidates (blockReason: {block_reason})")
        candidate = candidates[0]
        if "content" not in candidate:
            finish = candidate.get("finishReason", "unknown")
            raise ValueError(f"Gemini candidate has no content (finishReason: {finish})")
        return candidate["content"]["parts"][0]["text"]

    def chat_multi(self, system: str, messages: list[dict]) -> str:
        """Multi-turn chat for Gemini."""
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.1},
        }
        resp = requests.post(url, params={"key": self.api_key},
                             json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"Gemini returned no candidates")
        candidate = candidates[0]
        if "content" not in candidate:
            raise ValueError(f"Gemini candidate has no content")
        return candidate["content"]["parts"][0]["text"]

    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
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
                        piece = p.get("text") or ""
                        if piece:
                            yield StreamEvent(kind="text", text=piece)
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
