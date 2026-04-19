"""AI chat routes: provider/model/key management + turn lifecycle."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Desktop's list of providers — keep in sync with synapse/llm_assistant.py's
# AIChatPanel._PROVIDERS.
_PROVIDERS = ("Ollama", "Ollama Cloud", "OpenRouter", "OpenAI", "Claude",
              "Groq", "Gemini")


class TurnReq(BaseModel):
    user_text: str
    provider: str
    model: str


class KeyReq(BaseModel):
    key: str


@router.get("/providers")
async def list_providers(request: Request) -> dict:
    """Return each provider + whether a stored API key exists."""
    from synapse.llm_assistant import _load_api_keys
    keys = _load_api_keys() or {}
    return {"providers": [
        {"name": p, "has_key": bool(keys.get(p))} for p in _PROVIDERS
    ]}


@router.post("/providers/{name}/key")
async def save_key(name: str, body: KeyReq) -> dict:
    if name not in _PROVIDERS:
        raise HTTPException(status_code=400, detail=f"unknown provider: {name}")
    from synapse.llm_assistant import _store_api_key
    _store_api_key(name, body.key.strip())
    return {"ok": True}


@router.get("/models")
async def list_models(request: Request, provider: str) -> dict:
    """Return the provider's model list (live from its /models endpoint)."""
    from synapse.llm_assistant import _build_client
    client = _build_client(provider)
    if client is None:
        raise HTTPException(status_code=400, detail=f"unknown provider: {provider}")
    try:
        models = client.list_models()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")
    return {"models": list(models) if models else []}


@router.post("/turn")
async def start_turn(request: Request, body: TurnReq) -> dict:
    """Kick off one chat turn. Events stream via /api/ws."""
    session = request.app.state.session
    # Lazily attach a WebChatSession the first time.
    if session.chat_session is None:
        from synapse.server.chat_session import WebChatSession
        session.chat_session = WebChatSession(session)

    # Build client + dispatcher using the same factories the desktop panel uses.
    from synapse.llm_assistant import _build_client, _build_dispatcher_for_graph
    client = _build_client(body.provider, model=body.model)
    if client is None:
        raise HTTPException(status_code=400, detail=f"unknown provider: {body.provider}")
    dispatcher = _build_dispatcher_for_graph(session.graph, client=client)

    # Conversation history: for Phase 1e we keep history per-session in memory.
    # Persisted history lands in Phase 2+ when we add workflow save/load.
    history = session.chat_history
    turn_id = session.chat_session.start_turn(
        user_text=body.user_text, client=client,
        dispatcher=dispatcher, history=history,
    )
    history.append({"role": "user", "content": body.user_text})
    return {"turn_id": turn_id}


@router.post("/stop")
async def stop_turn(request: Request) -> dict:
    session = request.app.state.session
    if session.chat_session is not None:
        session.chat_session.stop()
    return {"ok": True}
