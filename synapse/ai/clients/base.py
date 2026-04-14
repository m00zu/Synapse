"""Base interface for Synapse LLM clients.

Phase 1 defines the streaming interface. Tools are accepted but ignored;
Phase 2 will wire them up through the orchestrator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Any


_VALID_KINDS = frozenset({"text", "tool_call", "done", "error"})


@dataclass
class StreamEvent:
    """One event yielded from a streaming LLM turn."""
    kind: str
    text: Optional[str] = None
    tool_call: Optional[dict] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"StreamEvent.kind must be one of {sorted(_VALID_KINDS)}, "
                f"got {self.kind!r}"
            )


# Hardcoded vision-capable model substrings. Matched case-insensitively as a
# substring of the model name. Conservative — prefer false-negative over
# false-positive (text-only fallback is always safe).
VISION_MODELS: tuple[str, ...] = (
    # Anthropic
    "claude-3", "claude-sonnet-4", "claude-opus-4", "claude-haiku-4",
    # OpenAI
    "gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-4-vision", "o1", "o3",
    # Google
    "gemini-1.5", "gemini-2", "gemini-pro-vision",
    # Ollama
    "llava", "llama3.2-vision", "qwen2.5-vl", "minicpm-v", "bakllava",
)


def is_vision_model(model: str) -> bool:
    """True if model name matches a known vision-capable family."""
    m = (model or "").lower()
    return any(tag in m for tag in VISION_MODELS)


class LLMClient(ABC):
    """Abstract base for all Synapse LLM clients.

    Existing sync methods `chat()` and `chat_multi()` are kept on concrete
    subclasses for backward compatibility but are not part of this ABC —
    they have provider-specific image handling and will be folded into the
    streaming interface in a later phase.
    """

    model: str

    @property
    def supports_vision(self) -> bool:
        return is_vision_model(self.model)

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return available model names, or [] if unavailable."""
        raise NotImplementedError

    @abstractmethod
    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        """Stream a turn. Phase 1: only yields 'text' then 'done' (or 'error').
        The *tools* argument is accepted for forward compatibility and ignored."""
        raise NotImplementedError
