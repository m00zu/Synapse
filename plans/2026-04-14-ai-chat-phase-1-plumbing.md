# AI Chat Phase 1 — Plumbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split LLM client classes out of `synapse/llm_assistant.py` into a new `synapse/ai/clients/` package, add a streaming text interface (`chat_with_tools_stream`) to each of the 6 active providers, and render assistant bubbles with markdown instead of plain escaped HTML. No orchestrator or tool-calling yet. Existing JSON-reply workflow keeps working unchanged.

**Architecture:**
- New `synapse/ai/clients/base.py` defines `LLMClient` ABC, `StreamEvent` dataclass, and `VISION_MODELS` capability registry.
- Each existing client (Ollama/OpenAI/Claude/Gemini/Groq) moves to its own file under `synapse/ai/clients/`. The existing `chat()` and `chat_multi()` methods stay verbatim for backward compatibility; a new `chat_with_tools_stream(system, messages, tools=None) -> Iterator[StreamEvent]` method is added. The `tools` arg is accepted but ignored in Phase 1.
- LlamaCpp and RunPod clients move too (to keep the file layout clean) but do not get streaming — they raise `NotImplementedError` if `chat_with_tools_stream` is called.
- `synapse/llm_assistant.py` re-exports each client from its new location so existing callers and workflow files that import `OllamaClient` etc. keep working.
- New `synapse/markdown_render.py` provides `render_markdown(text: str) -> str` using `markdown` + `pygments`. Phase-1 `_append_bubble` calls it for assistant bubbles only; user/error/system bubbles stay as-is.

**Tech Stack:** Python 3.13+, PySide6, `requests`, `markdown` (new dep), `pygments` (new dep), `pytest` (new dev dep).

---

## File Structure

Files created in this phase:

```
synapse/
  ai/
    __init__.py                  (new, empty)
    clients/
      __init__.py                (new: re-exports all client classes)
      base.py                    (new: LLMClient ABC + StreamEvent + VISION_MODELS)
      ollama.py                  (new: OllamaClient, moved + streaming)
      openai.py                  (new: OpenAIClient, moved + streaming)
      claude.py                  (new: ClaudeClient, moved + streaming)
      gemini.py                  (new: GeminiClient, moved + streaming)
      groq.py                    (new: GroqClient, moved + streaming)
      llamacpp.py                (new: LlamaCppClient, moved; streaming raises NotImplementedError)
      runpod.py                  (new: RunPodClient, moved; streaming raises NotImplementedError)
  markdown_render.py             (new: render_markdown())
tests/
  __init__.py                    (new, empty)
  ai/
    __init__.py                  (new, empty)
    test_base.py                 (new: StreamEvent tests)
    test_ollama_stream.py        (new: mocked-HTTP streaming tests)
    test_openai_stream.py        (new: mocked-HTTP streaming tests)
    test_claude_stream.py        (new: mocked-HTTP streaming tests)
    test_gemini_stream.py        (new: mocked-HTTP streaming tests)
    test_groq_stream.py          (new: mocked-HTTP streaming tests)
  test_markdown_render.py        (new: markdown rendering tests)
```

Files modified:

```
synapse/llm_assistant.py         (lines 603-1206: remove client class bodies, replace with re-export imports; _append_bubble updated to render markdown for assistant role)
pyproject.toml                   (add markdown, pygments to deps; add pytest to dev optional-deps)
requirements.txt                 (add markdown, pygments)
```

---

## Task 1: Dependencies and empty package scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `synapse/ai/__init__.py`
- Create: `synapse/ai/clients/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/ai/__init__.py`

- [ ] **Step 1: Add runtime deps to `pyproject.toml`**

Edit `[project]` → `dependencies` (currently ends at line 38 with `"zstandard>=0.25",`). Add two lines:

```toml
    "zstandard>=0.25",
    "markdown>=3.6",
    "pygments>=2.17",
]
```

Also add a `pytest` entry under `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
llm = ["llama-cpp-python>=0.3"]
dev = ["mkdocs", "mkdocs-material", "pymdown-extensions", "pytest>=8.0"]
```

- [ ] **Step 2: Add same runtime deps to `requirements.txt`**

Append to end of file (after `fpdf2>=2.8.7`):

```
markdown>=3.6
pygments>=2.17
```

- [ ] **Step 3: Install new deps**

Run: `pip install "markdown>=3.6" "pygments>=2.17" "pytest>=8.0"`

Expected: Successfully installed markdown, pygments, pytest (or "Requirement already satisfied").

- [ ] **Step 4: Create empty init files**

Create `synapse/ai/__init__.py`, `synapse/ai/clients/__init__.py`, `tests/__init__.py`, `tests/ai/__init__.py` — each as an empty file (zero bytes).

- [ ] **Step 5: Verify pytest runs**

Run: `pytest tests/ -q`
Expected: `no tests ran in 0.00s` (exit code 5 is fine).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt synapse/ai/__init__.py synapse/ai/clients/__init__.py tests/__init__.py tests/ai/__init__.py
git commit -m "chore: add ai package scaffolding + markdown/pygments/pytest deps"
```

---

## Task 2: `base.py` — StreamEvent dataclass + LLMClient ABC + VISION_MODELS

**Files:**
- Create: `synapse/ai/clients/base.py`
- Create: `tests/ai/test_base.py`

- [ ] **Step 1: Write the failing tests in `tests/ai/test_base.py`**

```python
"""Tests for StreamEvent and vision capability detection."""
import pytest
from synapse.ai.clients.base import (
    StreamEvent, LLMClient, is_vision_model, VISION_MODELS,
)


def test_stream_event_text_kind():
    ev = StreamEvent(kind="text", text="hello")
    assert ev.kind == "text"
    assert ev.text == "hello"
    assert ev.tool_call is None
    assert ev.error is None


def test_stream_event_done_kind():
    ev = StreamEvent(kind="done")
    assert ev.kind == "done"
    assert ev.text is None


def test_stream_event_error_kind():
    ev = StreamEvent(kind="error", error="boom")
    assert ev.kind == "error"
    assert ev.error == "boom"


def test_stream_event_rejects_unknown_kind():
    with pytest.raises(ValueError):
        StreamEvent(kind="not_a_kind")


def test_is_vision_model_known_vision():
    assert is_vision_model("claude-sonnet-4-20250514") is True
    assert is_vision_model("gpt-4o") is True
    assert is_vision_model("gemini-2.5-flash") is True
    assert is_vision_model("llava:13b") is True


def test_is_vision_model_known_text_only():
    assert is_vision_model("llama-3.3-70b-versatile") is False
    assert is_vision_model("gemma3:2b") is False


def test_is_vision_model_unknown_defaults_false():
    assert is_vision_model("some-future-model-xyz") is False


def test_llm_client_abstract():
    with pytest.raises(TypeError):
        LLMClient()  # abstract methods not implemented
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ai/test_base.py -v`
Expected: ImportError (`synapse.ai.clients.base` does not exist).

- [ ] **Step 3: Implement `synapse/ai/clients/base.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ai/test_base.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/clients/base.py tests/ai/test_base.py
git commit -m "feat(ai): add LLMClient ABC, StreamEvent, vision-model registry"
```

---

## Task 3: Move + stream — `OllamaClient`

**Files:**
- Create: `synapse/ai/clients/ollama.py`
- Create: `tests/ai/test_ollama_stream.py`
- Modify: `synapse/llm_assistant.py` (remove OllamaClient body, add re-export)

Ollama streaming API: POST `/api/chat` with `"stream": true` returns **newline-delimited JSON** (NDJSON). Each line is `{"message": {"content": "..."}, "done": false}`. Final line has `"done": true`.

- [ ] **Step 1: Write failing test in `tests/ai/test_ollama_stream.py`**

```python
"""Streaming tests for OllamaClient using mocked HTTP."""
from unittest.mock import patch, MagicMock
from synapse.ai.clients.ollama import OllamaClient
from synapse.ai.clients.base import StreamEvent


def _fake_ndjson_response(chunks: list[str], done: bool = True):
    """Build a MagicMock resembling requests.Response with iter_lines()."""
    lines: list[bytes] = []
    for c in chunks:
        lines.append(f'{{"message":{{"content":"{c}"}},"done":false}}'.encode())
    if done:
        lines.append(b'{"message":{"content":""},"done":true}')
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    return resp


def test_ollama_stream_yields_text_then_done():
    client = OllamaClient(model="gemma3:2b")
    with patch("synapse.ai.clients.ollama.requests.post",
               return_value=_fake_ndjson_response(["Hel", "lo", " world"])) as pm:
        events = list(client.chat_with_tools_stream(
            system="sys", messages=[{"role": "user", "content": "hi"}]))
    pm.assert_called_once()
    assert [e.kind for e in events] == ["text", "text", "text", "done"]
    assert "".join(e.text or "" for e in events if e.kind == "text") == "Hello world"


def test_ollama_stream_sends_stream_true():
    client = OllamaClient()
    with patch("synapse.ai.clients.ollama.requests.post",
               return_value=_fake_ndjson_response([])) as pm:
        list(client.chat_with_tools_stream(system="s", messages=[]))
    _, kwargs = pm.call_args
    assert kwargs["json"]["stream"] is True


def test_ollama_stream_error_emits_error_event():
    client = OllamaClient()
    def boom(*a, **k):
        raise ConnectionError("network down")
    with patch("synapse.ai.clients.ollama.requests.post", side_effect=boom):
        events = list(client.chat_with_tools_stream(
            system="s", messages=[{"role": "user", "content": "hi"}]))
    assert events[-1].kind == "error"
    assert "network down" in events[-1].error
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `pytest tests/ai/test_ollama_stream.py -v`
Expected: ImportError for `synapse.ai.clients.ollama`.

- [ ] **Step 3: Create `synapse/ai/clients/ollama.py`**

Copy the OllamaClient class from `synapse/llm_assistant.py:603-677` verbatim into the new file, then add the streaming method. The new file:

```python
"""Ollama client (local + cloud via base_url + api_key)."""
from __future__ import annotations

import json
import requests
from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


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
        h = {}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5,
                                headers=self._headers())
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        """Legacy non-streaming single-turn (kept for backward compat — JSON format)."""
        # Import here to avoid circular import at module load time.
        from synapse.llm_assistant import RESPONSE_SCHEMA
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

    # ------------------------------------------------------------------
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
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self._headers(),
                stream=True,
                timeout=120,
            )
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ai/test_ollama_stream.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Remove old `OllamaClient` body from `llm_assistant.py` and re-export**

In `synapse/llm_assistant.py`, replace lines 600-677 (the `OllamaClient` class definition, including the `# Ollama HTTP client` banner comment) with:

```python
# ---------------------------------------------------------------------------
# Ollama HTTP client — moved to synapse/ai/clients/ollama.py
# ---------------------------------------------------------------------------

from synapse.ai.clients.ollama import OllamaClient  # re-export
```

- [ ] **Step 6: Run the app's smoke import**

Run: `python -c "from synapse.llm_assistant import OllamaClient; c = OllamaClient(); print(c.DEFAULT_MODEL)"`
Expected: `gemma3:12b`

- [ ] **Step 7: Commit**

```bash
git add synapse/ai/clients/ollama.py tests/ai/test_ollama_stream.py synapse/llm_assistant.py
git commit -m "refactor(ai): extract OllamaClient, add streaming text interface"
```

---

## Task 4: Move + stream — `OpenAIClient`

**Files:**
- Create: `synapse/ai/clients/openai.py`
- Create: `tests/ai/test_openai_stream.py`
- Modify: `synapse/llm_assistant.py`

OpenAI Chat Completions streaming: POST `/chat/completions` with `"stream": true` returns **Server-Sent Events**. Lines begin with `data: `, body is JSON. The last event is `data: [DONE]`. Content is at `choices[0].delta.content`.

- [ ] **Step 1: Write failing test in `tests/ai/test_openai_stream.py`**

```python
from unittest.mock import patch, MagicMock
from synapse.ai.clients.openai import OpenAIClient


def _fake_sse_response(contents: list[str]):
    lines: list[bytes] = []
    for c in contents:
        lines.append(
            f'data: {{"choices":[{{"delta":{{"content":"{c}"}}}}]}}'.encode()
        )
    lines.append(b'data: [DONE]')
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    return resp


def test_openai_stream_concatenates_deltas():
    client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
    with patch("synapse.ai.clients.openai.requests.post",
               return_value=_fake_sse_response(["Hel", "lo"])) as pm:
        events = list(client.chat_with_tools_stream(
            system="s", messages=[{"role": "user", "content": "hi"}]))
    assert [e.kind for e in events] == ["text", "text", "done"]
    assert "".join(e.text for e in events if e.kind == "text") == "Hello"
    _, kwargs = pm.call_args
    assert kwargs["json"]["stream"] is True


def test_openai_stream_ignores_blank_and_keepalive_lines():
    client = OpenAIClient(api_key="sk-test")
    resp = MagicMock()
    resp.iter_lines.return_value = iter([b"", b": keepalive",
        b'data: {"choices":[{"delta":{"content":"X"}}]}',
        b'data: [DONE]'])
    resp.raise_for_status.return_value = None
    with patch("synapse.ai.clients.openai.requests.post", return_value=resp):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert [e.kind for e in events] == ["text", "done"]
    assert events[0].text == "X"


def test_openai_stream_error_emits_error_event():
    client = OpenAIClient(api_key="sk-test")
    with patch("synapse.ai.clients.openai.requests.post",
               side_effect=RuntimeError("timeout")):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert events[-1].kind == "error"
    assert "timeout" in events[-1].error
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/ai/test_openai_stream.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `synapse/ai/clients/openai.py`**

```python
"""OpenAI client (cloud Chat Completions)."""
from __future__ import annotations

import json
import requests
from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


class OpenAIClient(LLMClient):
    DEFAULT_MODEL = "gpt-4o-mini"
    BASE_URL      = "https://api.openai.com/v1"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model   = model

    def list_models(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            resp = requests.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            resp.raise_for_status()
            return sorted(
                m["id"] for m in resp.json().get("data", [])
                if m["id"].startswith("gpt-") or m["id"].startswith("o1") or m["id"].startswith("o3")
            )
        except Exception:
            return []

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
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
            headers={"Authorization": f"Bearer {self.api_key}"},
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
        resp = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": 0.1,
            "stream": True,
        }
        try:
            resp = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                stream=True,
                timeout=120,
            )
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
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    yield StreamEvent(kind="text", text=piece)
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ai/test_openai_stream.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Remove old `OpenAIClient` body from `llm_assistant.py` and re-export**

In `synapse/llm_assistant.py`, locate the block starting at line 680 (`# OpenAI client (cloud)` banner comment) through the end of the `OpenAIClient` class at line 771 (inclusive of the blank line after the class, stop before the `LlamaCppClient` banner). Replace the entire block with:

```python
# ---------------------------------------------------------------------------
# OpenAI client — moved to synapse/ai/clients/openai.py
# ---------------------------------------------------------------------------

from synapse.ai.clients.openai import OpenAIClient  # re-export
```

- [ ] **Step 6: Smoke import**

Run: `python -c "from synapse.llm_assistant import OpenAIClient; print(OpenAIClient.DEFAULT_MODEL)"`
Expected: `gpt-4o-mini`

- [ ] **Step 7: Commit**

```bash
git add synapse/ai/clients/openai.py tests/ai/test_openai_stream.py synapse/llm_assistant.py
git commit -m "refactor(ai): extract OpenAIClient, add streaming text interface"
```

---

## Task 5: Move + stream — `ClaudeClient`

**Files:**
- Create: `synapse/ai/clients/claude.py`
- Create: `tests/ai/test_claude_stream.py`
- Modify: `synapse/llm_assistant.py`

Anthropic Messages streaming: POST `/v1/messages` with `"stream": true`. Returns SSE with multiple event types. For Phase 1 we only care about `event: content_block_delta` frames whose `data:` payload has `delta.type == "text_delta"`, and `event: message_stop` (end of turn). Other event types (e.g. `message_start`, `ping`) are ignored.

- [ ] **Step 1: Write failing test in `tests/ai/test_claude_stream.py`**

```python
from unittest.mock import patch, MagicMock
from synapse.ai.clients.claude import ClaudeClient


def _fake_anthropic_sse(pieces: list[str]):
    lines: list[bytes] = [b"event: message_start",
                          b'data: {"type":"message_start"}', b""]
    for p in pieces:
        lines.append(b"event: content_block_delta")
        lines.append(
            f'data: {{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{p}"}}}}'.encode()
        )
        lines.append(b"")
    lines.append(b"event: message_stop")
    lines.append(b'data: {"type":"message_stop"}')
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    return resp


def test_claude_stream_concatenates_text_deltas():
    client = ClaudeClient(api_key="sk-ant-test")
    with patch("synapse.ai.clients.claude.requests.post",
               return_value=_fake_anthropic_sse(["Hi", " there"])):
        events = list(client.chat_with_tools_stream(
            system="s", messages=[{"role": "user", "content": "hi"}]))
    text_joined = "".join(e.text for e in events if e.kind == "text")
    assert text_joined == "Hi there"
    assert events[-1].kind == "done"


def test_claude_stream_ignores_non_text_events():
    client = ClaudeClient(api_key="sk-ant-test")
    resp = MagicMock()
    resp.iter_lines.return_value = iter([
        b"event: ping", b'data: {"type":"ping"}', b"",
        b"event: content_block_delta",
        b'data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{}"}}',
        b"event: message_stop", b'data: {"type":"message_stop"}',
    ])
    resp.raise_for_status.return_value = None
    with patch("synapse.ai.clients.claude.requests.post", return_value=resp):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    text_events = [e for e in events if e.kind == "text"]
    assert text_events == []
    assert events[-1].kind == "done"


def test_claude_stream_error_emits_error_event():
    client = ClaudeClient(api_key="sk-ant-test")
    with patch("synapse.ai.clients.claude.requests.post",
               side_effect=RuntimeError("bad key")):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert events[-1].kind == "error"
    assert "bad key" in events[-1].error
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/ai/test_claude_stream.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `synapse/ai/clients/claude.py`**

Copy the `ClaudeClient` class body from `synapse/llm_assistant.py:1018-1106` verbatim (the `list_models`, `chat`, `chat_multi` methods) into the new file structure below, then add the streaming method. Do not change the body of `chat` or `chat_multi`.

```python
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

    # === BEGIN: copy list_models, chat, chat_multi verbatim from
    # === synapse/llm_assistant.py:1026-1106 ===
    # (Retain the existing bodies unchanged.)

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
            resp = requests.post(
                f"{self.BASE_URL}/messages",
                headers=headers,
                json=payload,
                stream=True,
                timeout=120,
            )
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
```

**Important:** the `=== BEGIN / === copy` markers in the code block above are a directive to the engineer — literally copy the three methods from `synapse/llm_assistant.py` lines 1026-1106 into this file, replacing those comment lines. Do not leave the marker comments in the final file.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_claude_stream.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Remove old `ClaudeClient` body from `llm_assistant.py` and re-export**

In `synapse/llm_assistant.py`, locate the block starting at the `# Claude / Anthropic client` banner (around line 1014) through the end of the `ClaudeClient` class (around line 1106). Replace it with:

```python
# ---------------------------------------------------------------------------
# Claude client — moved to synapse/ai/clients/claude.py
# ---------------------------------------------------------------------------

from synapse.ai.clients.claude import ClaudeClient  # re-export
```

- [ ] **Step 6: Smoke import**

Run: `python -c "from synapse.llm_assistant import ClaudeClient; print(ClaudeClient.DEFAULT_MODEL)"`
Expected: `claude-sonnet-4-20250514`

- [ ] **Step 7: Commit**

```bash
git add synapse/ai/clients/claude.py tests/ai/test_claude_stream.py synapse/llm_assistant.py
git commit -m "refactor(ai): extract ClaudeClient, add streaming text interface"
```

---

## Task 6: Move + stream — `GroqClient`

**Files:**
- Create: `synapse/ai/clients/groq.py`
- Create: `tests/ai/test_groq_stream.py`
- Modify: `synapse/llm_assistant.py`

Groq is OpenAI-compatible; streaming format is identical to OpenAI SSE.

- [ ] **Step 1: Write failing test in `tests/ai/test_groq_stream.py`**

```python
from unittest.mock import patch, MagicMock
from synapse.ai.clients.groq import GroqClient


def _fake_sse(contents: list[str]):
    lines: list[bytes] = []
    for c in contents:
        lines.append(f'data: {{"choices":[{{"delta":{{"content":"{c}"}}}}]}}'.encode())
    lines.append(b'data: [DONE]')
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    return resp


def test_groq_stream_concatenates_deltas():
    client = GroqClient(api_key="gsk-test")
    with patch("synapse.ai.clients.groq.requests.post",
               return_value=_fake_sse(["Foo", "Bar"])):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert "".join(e.text for e in events if e.kind == "text") == "FooBar"
    assert events[-1].kind == "done"


def test_groq_stream_error_event():
    client = GroqClient(api_key="gsk-test")
    with patch("synapse.ai.clients.groq.requests.post",
               side_effect=RuntimeError("rate limit")):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert events[-1].kind == "error"
    assert "rate limit" in events[-1].error
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/ai/test_groq_stream.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `synapse/ai/clients/groq.py`**

Structure is identical to OpenAI but with `DEFAULT_MODEL = "llama-3.3-70b-versatile"` and `BASE_URL = "https://api.groq.com/openai/v1"`. Copy the `GroqClient` class from `synapse/llm_assistant.py:854-916` verbatim, then add `chat_with_tools_stream` identical in shape to OpenAI's (same SSE parsing):

```python
"""Groq client (OpenAI-compatible API)."""
from __future__ import annotations

import json
import requests
from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


class GroqClient(LLMClient):
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    BASE_URL      = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model   = model

    # === BEGIN: copy list_models, chat, chat_multi verbatim from
    # === synapse/llm_assistant.py:862-916 ===

    def chat_with_tools_stream(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> Iterator[StreamEvent]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": 0.1,
            "stream": True,
        }
        try:
            resp = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                stream=True,
                timeout=120,
            )
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
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    yield StreamEvent(kind="text", text=piece)
            yield StreamEvent(kind="done")
        except Exception as e:
            yield StreamEvent(kind="error", error=str(e))
```

**Important:** the `=== BEGIN / === copy` markers are a directive — replace them with the actual method bodies from the specified lines in `synapse/llm_assistant.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_groq_stream.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Remove old `GroqClient` body from `llm_assistant.py` and re-export**

In `synapse/llm_assistant.py`, replace the `GroqClient` class block (roughly lines 852-917) with:

```python
# ---------------------------------------------------------------------------
# Groq client — moved to synapse/ai/clients/groq.py
# ---------------------------------------------------------------------------

from synapse.ai.clients.groq import GroqClient  # re-export
```

- [ ] **Step 6: Smoke import**

Run: `python -c "from synapse.llm_assistant import GroqClient; print(GroqClient.DEFAULT_MODEL)"`
Expected: `llama-3.3-70b-versatile`

- [ ] **Step 7: Commit**

```bash
git add synapse/ai/clients/groq.py tests/ai/test_groq_stream.py synapse/llm_assistant.py
git commit -m "refactor(ai): extract GroqClient, add streaming text interface"
```

---

## Task 7: Move + stream — `GeminiClient`

**Files:**
- Create: `synapse/ai/clients/gemini.py`
- Create: `tests/ai/test_gemini_stream.py`
- Modify: `synapse/llm_assistant.py`

Gemini streaming: POST `models/{model}:streamGenerateContent?alt=sse&key=...`. Returns SSE where each `data:` event is a full partial `GenerateContentResponse`. Text is at `candidates[0].content.parts[0].text`. No explicit `[DONE]` sentinel — stream simply ends.

- [ ] **Step 1: Write failing test in `tests/ai/test_gemini_stream.py`**

```python
from unittest.mock import patch, MagicMock
from synapse.ai.clients.gemini import GeminiClient


def _fake_sse(pieces: list[str]):
    lines: list[bytes] = []
    for p in pieces:
        lines.append(
            f'data: {{"candidates":[{{"content":{{"parts":[{{"text":"{p}"}}]}}}}]}}'.encode()
        )
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    return resp


def test_gemini_stream_concatenates_parts():
    client = GeminiClient(api_key="gk-test")
    with patch("synapse.ai.clients.gemini.requests.post",
               return_value=_fake_sse(["He", "llo"])) as pm:
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert "".join(e.text for e in events if e.kind == "text") == "Hello"
    assert events[-1].kind == "done"
    # Called the streaming endpoint
    args, kwargs = pm.call_args
    assert ":streamGenerateContent" in args[0]


def test_gemini_stream_handles_empty_candidates():
    client = GeminiClient(api_key="gk-test")
    resp = MagicMock()
    resp.iter_lines.return_value = iter([
        b'data: {"candidates":[]}',
        b'data: {"candidates":[{"content":{"parts":[{"text":"X"}]}}]}',
    ])
    resp.raise_for_status.return_value = None
    with patch("synapse.ai.clients.gemini.requests.post", return_value=resp):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert any(e.kind == "text" and e.text == "X" for e in events)
    assert events[-1].kind == "done"


def test_gemini_stream_error_event():
    client = GeminiClient(api_key="gk-test")
    with patch("synapse.ai.clients.gemini.requests.post",
               side_effect=RuntimeError("quota exceeded")):
        events = list(client.chat_with_tools_stream(system="s", messages=[]))
    assert events[-1].kind == "error"
    assert "quota exceeded" in events[-1].error
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/ai/test_gemini_stream.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `synapse/ai/clients/gemini.py`**

```python
"""Google Gemini client."""
from __future__ import annotations

import json
import requests
from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


class GeminiClient(LLMClient):
    DEFAULT_MODEL = "gemini-2.5-flash-lite"
    BASE_URL      = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model   = model

    # === BEGIN: copy list_models, chat, chat_multi verbatim from
    # === synapse/llm_assistant.py:931-1011 ===

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
            resp = requests.post(
                url,
                params={"key": self.api_key, "alt": "sse"},
                json=payload,
                stream=True,
                timeout=120,
            )
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
```

Replace the `=== BEGIN / === copy` markers with the actual method bodies from `synapse/llm_assistant.py:931-1011`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ai/test_gemini_stream.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Remove old `GeminiClient` body from `llm_assistant.py` and re-export**

In `synapse/llm_assistant.py`, replace the `GeminiClient` class block (roughly lines 919-1012) with:

```python
# ---------------------------------------------------------------------------
# Gemini client — moved to synapse/ai/clients/gemini.py
# ---------------------------------------------------------------------------

from synapse.ai.clients.gemini import GeminiClient  # re-export
```

- [ ] **Step 6: Smoke import**

Run: `python -c "from synapse.llm_assistant import GeminiClient; print(GeminiClient.DEFAULT_MODEL)"`
Expected: `gemini-2.5-flash-lite`

- [ ] **Step 7: Commit**

```bash
git add synapse/ai/clients/gemini.py tests/ai/test_gemini_stream.py synapse/llm_assistant.py
git commit -m "refactor(ai): extract GeminiClient, add streaming text interface"
```

---

## Task 8: Move — `LlamaCppClient` and `RunPodClient` (no streaming)

These providers are out of scope for Phase 1 streaming but still need to live alongside their siblings. Their `chat_with_tools_stream` raises `NotImplementedError` with a clear message.

**Files:**
- Create: `synapse/ai/clients/llamacpp.py`
- Create: `synapse/ai/clients/runpod.py`
- Modify: `synapse/llm_assistant.py`

- [ ] **Step 1: Create `synapse/ai/clients/llamacpp.py`**

Template:

```python
"""LlamaCpp client (local gguf, optional dep)."""
from __future__ import annotations

from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


class LlamaCppClient(LLMClient):
    # === BEGIN: copy ALL of LlamaCppClient verbatim (incl. class attributes,
    # === __init__, list_models, chat) from synapse/llm_assistant.py:774-853 ===

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
```

Replace the `=== BEGIN` marker with the verbatim body from `synapse/llm_assistant.py:774-853`.

- [ ] **Step 2: Create `synapse/ai/clients/runpod.py`**

Template:

```python
"""RunPod client (serverless inference)."""
from __future__ import annotations

from typing import Iterator, Optional

from synapse.ai.clients.base import LLMClient, StreamEvent


class RunPodClient(LLMClient):
    # === BEGIN: copy ALL of RunPodClient verbatim from
    # === synapse/llm_assistant.py:1107 through the end of the class ===

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
```

Find the end of `RunPodClient` in `synapse/llm_assistant.py` (it starts at line 1107; scan forward until the next top-level `class` or module-level statement). Copy its body verbatim.

- [ ] **Step 3: Replace old class bodies in `llm_assistant.py` with re-exports**

Replace the `LlamaCppClient` block (starts around line 774, `# LlamaCpp client` banner) with:

```python
# ---------------------------------------------------------------------------
# LlamaCpp client — moved to synapse/ai/clients/llamacpp.py
# ---------------------------------------------------------------------------

from synapse.ai.clients.llamacpp import LlamaCppClient  # re-export
```

Replace the `RunPodClient` block (starts around line 1107) with:

```python
# ---------------------------------------------------------------------------
# RunPod client — moved to synapse/ai/clients/runpod.py
# ---------------------------------------------------------------------------

from synapse.ai.clients.runpod import RunPodClient  # re-export
```

- [ ] **Step 4: Smoke import for both**

Run: `python -c "from synapse.llm_assistant import LlamaCppClient, RunPodClient; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add synapse/ai/clients/llamacpp.py synapse/ai/clients/runpod.py synapse/llm_assistant.py
git commit -m "refactor(ai): extract LlamaCppClient and RunPodClient (no streaming)"
```

---

## Task 9: `clients/__init__.py` convenience re-exports

**Files:**
- Modify: `synapse/ai/clients/__init__.py`

- [ ] **Step 1: Write re-exports**

Replace the empty `synapse/ai/clients/__init__.py` with:

```python
"""Synapse LLM client implementations."""
from synapse.ai.clients.base import LLMClient, StreamEvent, VISION_MODELS, is_vision_model
from synapse.ai.clients.ollama import OllamaClient
from synapse.ai.clients.openai import OpenAIClient
from synapse.ai.clients.claude import ClaudeClient
from synapse.ai.clients.gemini import GeminiClient
from synapse.ai.clients.groq import GroqClient
from synapse.ai.clients.llamacpp import LlamaCppClient
from synapse.ai.clients.runpod import RunPodClient

__all__ = [
    "LLMClient", "StreamEvent", "VISION_MODELS", "is_vision_model",
    "OllamaClient", "OpenAIClient", "ClaudeClient", "GeminiClient",
    "GroqClient", "LlamaCppClient", "RunPodClient",
]
```

- [ ] **Step 2: Verify import**

Run: `python -c "from synapse.ai.clients import OllamaClient, ClaudeClient, StreamEvent; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Run full client test suite**

Run: `pytest tests/ai/ -v`
Expected: all tests PASS (base + 5 provider stream tests = ~13 tests).

- [ ] **Step 4: Commit**

```bash
git add synapse/ai/clients/__init__.py
git commit -m "refactor(ai): add clients package convenience re-exports"
```

---

## Task 10: `markdown_render.py` — markdown → HTML with Pygments

**Files:**
- Create: `synapse/markdown_render.py`
- Create: `tests/test_markdown_render.py`

The renderer must:
- Produce valid inline HTML suitable for `QTextBrowser` (which supports a subset of HTML).
- Handle fenced code blocks with language hints (`python`, `json`, `bash`, etc.) and highlight via Pygments inline styles (not external CSS, since `QTextBrowser` doesn't load stylesheets reliably).
- Handle tables (GFM).
- Be reasonably fast — called on every token-throttle tick during streaming (~20 Hz).

- [ ] **Step 1: Write failing test in `tests/test_markdown_render.py`**

```python
from synapse.markdown_render import render_markdown


def test_renders_plain_paragraph():
    html = render_markdown("Hello **world**.")
    assert "<strong>world</strong>" in html


def test_renders_heading():
    html = render_markdown("# Title")
    assert "<h1" in html
    assert "Title" in html


def test_renders_fenced_code_with_language():
    src = "```python\nprint('hi')\n```"
    html = render_markdown(src)
    # Pygments emits inline style spans for syntax highlighting
    assert "style=" in html
    assert "print" in html


def test_renders_table():
    md = "| a | b |\n| - | - |\n| 1 | 2 |\n"
    html = render_markdown(md)
    assert "<table" in html
    assert "<td>1</td>" in html


def test_escapes_raw_html():
    # XSS guard: raw <script> in source markdown should not produce a live
    # script tag. Default python-markdown escapes unsafe HTML.
    html = render_markdown("<script>alert(1)</script>")
    assert "<script>" not in html.lower()


def test_empty_input_returns_empty_string():
    assert render_markdown("") == ""
    assert render_markdown(None) == ""  # tolerant of None
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_markdown_render.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `synapse/markdown_render.py`**

```python
"""Markdown → HTML renderer for chat bubbles.

Uses python-markdown with fenced_code, tables, codehilite (Pygments inline
styles). Output is safe for QTextBrowser which handles a subset of HTML.
"""
from __future__ import annotations

import markdown as _md

# Reusable Markdown instance with the extensions we need. codehilite with
# inline_css=True embeds Pygments styles directly into generated <span>s, so
# no external stylesheet is needed — important for QTextBrowser.
_RENDERER = _md.Markdown(
    extensions=["fenced_code", "tables", "codehilite"],
    extension_configs={
        "codehilite": {
            "guess_lang": False,
            "noclasses": True,       # inline styles via Pygments
            "pygments_style": "default",
        },
    },
    output_format="html",
)


def render_markdown(text: str | None) -> str:
    """Render *text* from markdown to an HTML snippet. Empty/None → ''."""
    if not text:
        return ""
    # Reset so internal state doesn't accumulate across calls.
    _RENDERER.reset()
    return _RENDERER.convert(text)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_markdown_render.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add synapse/markdown_render.py tests/test_markdown_render.py
git commit -m "feat: add markdown_render module for chat bubbles"
```

---

## Task 11: Wire markdown rendering into `AIChatPanel._append_bubble`

Only the assistant role gets markdown rendering in Phase 1. User bubbles stay plain-escaped (users type plain text; rendering their input as markdown is confusing). Error and system bubbles stay as they are.

**Files:**
- Modify: `synapse/llm_assistant.py` (lines 2987-3054, the `_append_bubble` method)

- [ ] **Step 1: Add import at top of `llm_assistant.py`**

Near the other synapse-local imports (look for lines that say `from data_models` or similar around lines 1-50 of the file), add:

```python
from synapse.markdown_render import render_markdown
```

- [ ] **Step 2: Update the `assistant` branch of `_append_bubble`**

In `synapse/llm_assistant.py`, inside `_append_bubble` (starts at line 2987), the `elif role == "assistant":` branch currently uses `text_html` (plain-escaped). Replace its inner text generation to use markdown rendering.

The existing block (approximately lines 3011-3027) looks like:

```python
elif role == "assistant":
    html = (
        f"<table width='100%' cellpadding='0' cellspacing='0'>"
        f"<tr><td style='background:{c['ai_bg']}; color:{c['ai_fg']}; "
        f"padding:8px 14px; border-radius:4px 14px 14px 14px; "
        f"border:1px solid {c['ai_border']};'>"
        f"<span style='color:{c['ai_label']}; font-size:10px; "
        f"font-weight:600;'>AI</span><br>"
        f"<span style='font-size:13px; line-height:1.5;'>"
        f"{text_html}</span></td>"
        ...
    )
```

Replace that branch with:

```python
elif role == "assistant":
    body_html = render_markdown(text)
    if not body_html:
        # Fallback for any edge case where markdown rendering returns empty
        body_html = text_html
    html = (
        f"<table width='100%' cellpadding='0' cellspacing='0'>"
        f"<tr><td style='background:{c['ai_bg']}; color:{c['ai_fg']}; "
        f"padding:8px 14px; border-radius:4px 14px 14px 14px; "
        f"border:1px solid {c['ai_border']};'>"
        f"<span style='color:{c['ai_label']}; font-size:10px; "
        f"font-weight:600;'>AI</span><br>"
        f"<div style='font-size:13px; line-height:1.5;'>"
        f"{body_html}</div></td>"
        f"<td width='15%'></td></tr>"
        f"<tr><td style='padding:0; line-height:0; font-size:0;'>"
        f"<span style='color:{c['ai_bg']}; font-size:14px; "
        f"line-height:0;'>&#9699;</span></td>"
        f"<td></td></tr>"
        f"</table>"
    )
```

Two differences from the original:
- `body_html = render_markdown(text)` — HTML is already safe-escaped by `markdown`.
- The outer wrapper changes from `<span>` to `<div>` because markdown output can contain block-level elements (`<p>`, `<pre>`, `<table>`), which are invalid inside `<span>`.

- [ ] **Step 3: Run the app and smoke-test manually**

Run: `python -m synapse`
Then:
1. Open the AI Chat dock (View → AI Chat).
2. Select any configured provider.
3. Send a message: "Explain what a ParticleProps node does in one sentence, include a bullet list of its inputs, and give a one-line python example in a fenced code block."

Expected: Reply arrives; bullets render as real bullets; code block shows syntax-highlighted Python (dark background with colored tokens); no raw `**` or `####` artifacts. If the reply is a JSON workflow template (current behavior for workflow-building prompts), it renders as a single code block — that is acceptable for Phase 1.

- [ ] **Step 4: Commit**

```bash
git add synapse/llm_assistant.py
git commit -m "feat(chat): render assistant bubbles with markdown (Phase 1)"
```

---

## Task 12: Phase-1 end-to-end smoke test

Non-regression check that the existing JSON-reply workflow still works unchanged, and all new streaming methods importable.

**Files:**
- Create: `tests/ai/test_phase1_integration.py`

- [ ] **Step 1: Write regression test in `tests/ai/test_phase1_integration.py`**

```python
"""Phase-1 non-regression checks — no network required."""
from synapse.llm_assistant import (
    OllamaClient, OpenAIClient, ClaudeClient,
    GeminiClient, GroqClient, LlamaCppClient, RunPodClient,
)
from synapse.ai.clients import (
    OllamaClient as OllamaNew, LLMClient, StreamEvent,
)


def test_reexported_classes_are_same():
    # Re-exports from llm_assistant resolve to the new-home classes.
    assert OllamaClient is OllamaNew


def test_all_streaming_clients_are_subclasses_of_base():
    for cls in (OllamaClient, OpenAIClient, ClaudeClient,
                GeminiClient, GroqClient, LlamaCppClient, RunPodClient):
        assert issubclass(cls, LLMClient), f"{cls.__name__} is not LLMClient"


def test_each_client_has_chat_with_tools_stream():
    for cls in (OllamaClient, OpenAIClient, ClaudeClient,
                GeminiClient, GroqClient, LlamaCppClient, RunPodClient):
        assert callable(getattr(cls, "chat_with_tools_stream", None)), \
            f"{cls.__name__} missing chat_with_tools_stream"


def test_legacy_chat_signature_preserved():
    # The existing app.py / AIChatPanel code calls .chat(system, user).
    # That signature MUST still work on the five streaming providers.
    import inspect
    for cls in (OllamaClient, OpenAIClient, ClaudeClient, GeminiClient, GroqClient):
        sig = inspect.signature(cls.chat)
        params = list(sig.parameters)
        assert "system" in params and "user" in params, \
            f"{cls.__name__}.chat lost its legacy signature"


def test_vision_flag_differs_per_model():
    assert ClaudeClient(api_key="x", model="claude-sonnet-4-20250514").supports_vision is True
    assert GroqClient(api_key="x", model="llama-3.3-70b-versatile").supports_vision is False
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass. Roughly 20 tests total.

- [ ] **Step 3: Launch the app and run a non-AI workflow**

Run: `python -m synapse`
Then: open an existing workflow file (e.g. any `.synapse` from `workflows/`) or build a 2-node graph manually (CSVLoader → anything) and evaluate it. Confirm node evaluation works normally — the purpose is to check we did not accidentally break app startup by any of the `llm_assistant.py` edits.

- [ ] **Step 4: Final Phase-1 commit**

```bash
git add tests/ai/test_phase1_integration.py
git commit -m "test(ai): add Phase-1 non-regression suite"
```

---

## Self-Review Checklist

The plan covers the spec's Phase 1:

- ✅ Split clients into `ai/clients/` — Tasks 3–8.
- ✅ Add `chat_with_tools_stream` (text-only, tools ignored) for the 6 active providers — Tasks 3–7. (LlamaCpp + RunPod stubbed in Task 8 as explicitly out of scope for streaming.)
- ✅ Markdown rendering — Tasks 10–11.
- ✅ Existing JSON-reply flow keeps working — Task 12 regression test and manual smoke in Task 11 Step 3.

**Not in Phase 1 (deferred to Phase 2+):**
- `ChatOrchestrator`, `ToolDispatcher`, tool schemas, context builder, history rollup, feature flag.
- Streaming bubble UI, tool-call chips, inline Apply/Discard, token meter, Stop button, vision badge.
- Removing the old `TwoPassLLMWorker`.

**Placeholder scan:** no `TBD`/`TODO`/"fill in later" tokens. Where instructions say "copy verbatim from lines X-Y", the engineer has an exact file + line range to copy, which is a directive, not a placeholder.

**Type consistency:** all 7 clients extend `LLMClient`, all implement `chat_with_tools_stream` with the same signature. `StreamEvent.kind` values (`text`, `tool_call`, `done`, `error`) are consistent across all tests. `render_markdown(text: str | None) -> str` signature matches its test expectations.
