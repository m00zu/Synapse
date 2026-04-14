"""Anthropic Claude client."""
from __future__ import annotations

import json
import requests
from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


class ClaudeClient(LLMClient):
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    BASE_URL      = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model   = model

    def list_models(self) -> list[str]:
        """Return a curated list of Claude models (Anthropic has no list endpoint)."""
        if not self.api_key:
            return []
        return [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250506",
        ]

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        if images:
            user_content = [{"type": "text", "text": user}]
            for b64 in images:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                })
        else:
            user_content = user
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
        }
        resp = requests.post(
            f"{self.BASE_URL}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if not content:
            raise ValueError(f"Claude returned no content (stop_reason: {data.get('stop_reason', 'unknown')})")
        return content[0].get("text", "")

    def chat_multi(self, system: str, messages: list[dict]) -> str:
        """Multi-turn chat for Claude."""
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
            "temperature": 0.1,
        }
        resp = requests.post(
            f"{self.BASE_URL}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if not content:
            raise ValueError(f"Claude returned no content")
        return content[0].get("text", "")

    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
            "stream": True,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
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
                    if etype == "content_block_delta":
                        delta = obj.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            piece = delta.get("text") or ""
                            if piece:
                                yield StreamEvent(kind="text", text=piece)
                    elif etype == "message_stop":
                        break
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
