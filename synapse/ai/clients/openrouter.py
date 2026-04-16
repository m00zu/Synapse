"""OpenRouter client — OpenAI-compatible gateway with free-tier models.

OpenRouter proxies many providers (OpenAI, Anthropic, Meta, Google, …) behind
an OpenAI-shaped API. Free-tier models are tagged with the ``:free`` suffix,
e.g. ``meta-llama/llama-3.3-70b-instruct:free``. The wire protocol is
effectively identical to ``OpenAIClient``, so this client mirrors that class
with a different base URL, model list filter, and a couple of recommended
OpenRouter-specific headers (``HTTP-Referer``, ``X-Title``) for attribution.

Retries: free-tier models often hit 429 rate limits when generate_workflow
makes 2 sequential calls back-to-back. _post_with_retry backs off and retries
twice before surfacing the 429 so transient caps don't break a turn.
"""
from __future__ import annotations

import json
import time
import requests
from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


# Optional attribution headers OpenRouter logs on the dashboard.
# Harmless if OpenRouter ever drops support for them.
_REFERER = "https://github.com/m00zu/Synapse"
_TITLE = "Synapse"

# 429 retry policy — total wait at most ~10 seconds before giving up.
_RETRY_DELAYS = (2.0, 5.0)


def _post_with_retry(url: str, *, headers: dict, json_payload: dict,
                     stream: bool, timeout: float):
    """POST with retry-on-429 backoff. Returns the final Response object
    (which may still be non-200 — caller decides whether to raise)."""
    last = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        resp = requests.post(url, headers=headers, json=json_payload,
                             stream=stream, timeout=timeout)
        last = resp
        if resp.status_code != 429:
            return resp
        if attempt >= len(_RETRY_DELAYS):
            return resp
        # OpenRouter may include Retry-After (in seconds).
        delay = _RETRY_DELAYS[attempt]
        try:
            ra = resp.headers.get("Retry-After")
            if ra is not None:
                delay = max(delay, float(ra))
        except Exception:
            pass
        resp.close()
        time.sleep(delay)
    return last


class OpenRouterClient(LLMClient):
    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
    BASE_URL      = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model   = model

    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer":  _REFERER,
            "X-Title":       _TITLE,
        }

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Return all OpenRouter model ids. No key required for /models."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/models",
                headers=self._auth_headers() if self.api_key else {},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            # Keep the id; sort free-tier models to the top so users see them first.
            ids = sorted(m["id"] for m in data if m.get("id"))
            free = [i for i in ids if i.endswith(":free")]
            paid = [i for i in ids if not i.endswith(":free")]
            return free + paid
        except Exception:
            return []

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        """Legacy non-streaming single-turn — mirrors OpenAIClient.chat."""
        if images:
            user_content: list | str = [{"type": "text", "text": user}]
            for b64 in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        else:
            user_content = user
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        resp = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers=self._auth_headers(),
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def chat_multi(self, system: str, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": 0.1,
        }
        resp = _post_with_retry(
            f"{self.BASE_URL}/chat/completions",
            headers=self._auth_headers(),
            json_payload=payload,
            stream=False,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        """Streaming chat with optional OpenAI-style tool-calling."""
        from synapse.ai.clients.tool_adapters import to_openai_tools

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": 0.1,
            "stream": True,
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
        partial: dict[int, dict] = {}

        try:
            resp = _post_with_retry(
                f"{self.BASE_URL}/chat/completions",
                headers=self._auth_headers(),
                json_payload=payload,
                stream=True,
                timeout=120,
            )
            with resp:
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
