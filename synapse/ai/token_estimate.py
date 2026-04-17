"""
token_estimate.py — Lightweight token estimator and per-model context-window lookup.

This is a display utility for the token-meter UI gauge; it is NOT a real
tokenizer. Rule of thumb: ~4 characters per token. Phase 4 will replace the
estimator with a real tokenizer + rollup, but the API surface stays the same.

No Qt, no network, no LLM SDK — pure Python only.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD = 10
_FALLBACK_CONTEXT_WINDOW = 8192

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Return an approximate token count for *text*.

    Uses the ~4-chars-per-token heuristic.  Empty string returns 0.
    """
    if not text:
        return 0
    return len(text) // _CHARS_PER_TOKEN


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Return an approximate token count for a list of chat messages.

    For each message the estimate is:
        estimate_tokens(role) + estimate_tokens(content) + PER_MESSAGE_OVERHEAD

    Structured tool-call fields beyond ``role`` and ``content`` are ignored;
    a flat +10 fudge per message covers their overhead.  A missing ``content``
    key (e.g. tool-result stubs) is treated as an empty string.
    """
    total = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        # content can be a list of content-blocks (OpenAI vision / tool use)
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        total += estimate_tokens(role) + estimate_tokens(str(content)) + _PER_MESSAGE_OVERHEAD
    return total


# ---------------------------------------------------------------------------
# Per-model context-window table
# ---------------------------------------------------------------------------
# The table uses (provider_lower, model_prefix_lower) prefix matching so that
# minor-version bumps (e.g. claude-sonnet-4-7) are covered without edits.
# Entries are checked in order; first match wins.
# ---------------------------------------------------------------------------

_CONTEXT_TABLE: list[tuple[str, str, int]] = [
    # provider prefix, model prefix, context window (tokens)

    # --- Anthropic / Claude ---
    ("claude",     "claude-",       200_000),
    ("anthropic",  "claude-",       200_000),      # provider-name robustness

    # --- OpenAI ---
    ("openai",  "gpt-4.1",          1_000_000),    # gpt-4.1 has 1M window
    ("openai",  "o1",               200_000),
    ("openai",  "o3",               200_000),
    ("openai",  "o4",               200_000),
    ("openai",  "gpt-4o",           128_000),
    ("openai",  "gpt-4",            128_000),
    ("openai",  "gpt-3.5",          16_385),

    # --- Google Gemini ---
    ("gemini",  "gemini-2.5",       1_000_000),
    ("gemini",  "gemini-2.0",       1_000_000),
    ("gemini",  "gemini-1.5",       1_000_000),
    ("gemini",  "gemini-1.0",       32_000),
    ("gemini",  "gemini-",          1_000_000),    # catch-all for future Gemini

    # --- Groq (hosts Llama variants; 32k is a safe underestimate) ---
    ("groq",    "",                 32_000),

    # NOTE: "ollama cloud" rows must come before "ollama" rows below — both match
    # .startswith("ollama") so order is load-bearing. Do not resort alphabetically.
    # --- Ollama Cloud (free-tier cloud-hosted models, genuinely served at 128k) ---
    ("ollama cloud", "gemma3:",     128_000),
    ("ollama cloud", "gemma3-",     128_000),
    ("ollama cloud", "nemotron",    128_000),
    ("ollama cloud", "",            128_000),      # catch-all for Ollama Cloud

    # --- Ollama local (uses runtime num_ctx default of ~2k-8k, NOT architectural max;
    #     underestimate to 8192 so the UI flags OOM instead of hiding it) ---
    ("ollama",  "",                 8_192),        # catch-all for local Ollama

    # --- OpenRouter (vendor/model-name[:free] format) ---
    ("openrouter", "openai/",       128_000),
    ("openrouter", "anthropic/",    200_000),
    ("openrouter", "google/",       1_000_000),
    ("openrouter", "meta-llama/",   32_000),
    ("openrouter", "",              32_000),       # catch-all for OpenRouter
]


def model_context_window(provider: str, model: str) -> int:
    """Return a best-guess context window (in tokens) for *provider* + *model*.

    Matching is case-insensitive prefix matching on both the provider and the
    model name.  Falls back to 8 192 when no entry matches — an intentional
    underestimate so the UI errs toward "near budget" rather than hiding a
    real overflow.
    """
    p = provider.lower().strip()
    m = model.lower().strip()

    for prov_prefix, model_prefix, window in _CONTEXT_TABLE:
        if p == prov_prefix or p.startswith(prov_prefix):
            if model_prefix == "" or m.startswith(model_prefix):
                return window

    return _FALLBACK_CONTEXT_WINDOW
