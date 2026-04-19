import pytest

pytest.importorskip("PySide6")


@pytest.mark.asyncio
async def test_list_providers_returns_expected_set(client):
    resp = await client.get("/api/chat/providers")
    assert resp.status_code == 200
    names = {p["name"] for p in resp.json()["providers"]}
    assert {"Ollama", "Claude", "OpenAI", "Gemini"} <= names


@pytest.mark.asyncio
async def test_save_key_rejects_unknown_provider(client):
    resp = await client.post("/api/chat/providers/Bogus/key", json={"key": "x"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_stop_is_idempotent(client):
    for _ in range(3):
        resp = await client.post("/api/chat/stop")
        assert resp.status_code == 200
