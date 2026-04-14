"""RunPod client (serverless inference)."""
from __future__ import annotations

import requests
from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


class RunPodClient(LLMClient):
    """
    Client for RunPod serverless vLLM endpoints using the synchronous runsync API.

    Uses raw prompt completion (not OpenAI chat format) — the actual output format is:
      {"output": [{"choices": [{"tokens": ["..."]}], "usage": {...}}], "status": "COMPLETED"}

    System + user message are combined using the ChatML template (Qwen2.5 / instruct models).
    No model field is needed — the model is fixed by the RunPod deployment.
    """
    BASE_URL = "https://api.runpod.ai/v2"
    # Sentinel attributes so _on_model_changed doesn't crash; not used in requests
    model         = ""
    DEFAULT_MODEL = ""

    def __init__(self, api_key: str = "", endpoint_id: str = ""):
        self.api_key     = api_key
        self.endpoint_id = endpoint_id

    def _headers(self) -> dict:
        return {
            "Content-Type":  "application/json",
            "Authorization": f"{self.api_key}",
        }

    def list_models(self) -> list[str]:
        """No model list for RunPod — model is fixed at deployment."""
        return []

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        if not self.endpoint_id:
            raise ValueError("RunPod endpoint ID is not configured.")

        # ChatML / Qwen2.5 instruct template
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        payload = {
            "input": {
                "prompt": prompt,
                "sampling_params": {
                    "temperature": 0.1,
                    "max_tokens":  4096,
                    "stop":        ["<|im_end|>"],
                },
            }
        }
        url  = f"{self.BASE_URL}/{self.endpoint_id}/runsync"
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=120)
        resp.raise_for_status()
        data   = resp.json()
        status = data.get("status", "")
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise ValueError(f"RunPod job {status}: {data.get('error', 'no detail')}")

        output = data.get("output", [])
        if isinstance(output, list) and output:
            choices = output[0].get("choices", [])
            if choices:
                tokens = choices[0].get("tokens", [])
                return "".join(tokens)
        raise ValueError(f"Unexpected RunPod output format: {output}")

    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        yield StreamEvent(
            kind="error",
            error="RunPod streaming not supported in Phase 1.",
        )
