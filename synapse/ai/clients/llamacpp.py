"""LlamaCpp client (local GGUF, optional dep)."""
from __future__ import annotations

from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


class LlamaCppClient(LLMClient):
    """
    Drop-in replacement for OllamaClient that loads a GGUF model directly
    via llama-cpp-python.  No server, no Torch, no Ollama required.

    The model is loaded once on first use and cached for subsequent calls.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096,
                 n_threads: int = 0, n_gpu_layers: int = 0):
        """
        model_path   : path to the .gguf file
        n_ctx        : context window in tokens (must cover system + user + JSON reply)
        n_threads    : CPU threads (0 = auto-detect)
        n_gpu_layers : layers to offload to GPU (0 = CPU-only, the default)
        """
        self.model_path   = model_path
        self.n_ctx        = n_ctx
        self.n_threads    = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._llm         = None   # lazy-loaded on first chat() call

    @property
    def supports_vision(self) -> bool:
        # GGUF text models don't support vision.
        return False

    # ------------------------------------------------------------------
    def _load(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python"
            ) from e

        import os
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"GGUF model not found: {self.model_path}\n"
                "Fine-tune and export the model first (see finetune/train.py)."
            )

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads or None,   # None → llama.cpp auto-selects
            n_gpu_layers=self.n_gpu_layers,      # 0 = CPU-only
            verbose=False,
        )

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Returns the model filename so the UI can display it."""
        import os
        name = os.path.basename(self.model_path)
        return [name] if os.path.isfile(self.model_path) else []

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        """
        Runs inference and returns the raw JSON string.
        Raises FileNotFoundError if the GGUF file is missing.
        Note: images are ignored — GGUF models don't support vision.
        """
        self._load()
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},   # constrained JSON output
        )
        return response["choices"][0]["message"]["content"]

    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        yield StreamEvent(
            kind="error",
            error="LlamaCpp streaming not supported in Phase 1. Use Ollama for local models.",
        )
