from synapse.ai.token_estimate import (
    estimate_tokens,
    estimate_messages_tokens,
    model_context_window,
)


def test_empty_string_zero_tokens():
    assert estimate_tokens("") == 0


def test_four_chars_roughly_one_token():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_messages_sums_role_and_content_with_per_message_overhead():
    msgs = [
        {"role": "user", "content": "a" * 400},       # ~100 + 1 (role) + 10
        {"role": "assistant", "content": "b" * 400},  # ~100 + 2 + 10
    ]
    n = estimate_messages_tokens(msgs)
    assert 200 <= n <= 250


def test_messages_tolerates_missing_content():
    # Orchestrator's tool_result blocks may have no `content` key.
    msgs = [{"role": "tool", "tool_call_id": "abc"}]
    assert estimate_messages_tokens(msgs) >= 0


def test_context_window_known_models():
    assert model_context_window("Claude", "claude-sonnet-4-6") >= 200_000
    assert model_context_window("Gemini", "gemini-2.5-pro") >= 1_000_000


def test_context_window_unknown_falls_back_to_8k():
    assert model_context_window("Whatever", "unheard-of-3b") == 8192
