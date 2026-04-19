import io
import pytest
from pathlib import Path

pytest.importorskip("PySide6")



@pytest.mark.asyncio
async def test_upload_returns_server_path(client, tmp_path, monkeypatch):
    # Redirect ~/.synapse/uploads to tmp_path for this test.
    monkeypatch.setenv("HOME", str(tmp_path))
    fake = io.BytesIO(b"hello world")
    resp = await client.post("/api/files/upload",
                             files={"file": ("hi.txt", fake, "text/plain")})
    assert resp.status_code == 200
    sp = resp.json()["server_path"]
    assert Path(sp).exists()
    assert Path(sp).read_bytes() == b"hello world"


@pytest.mark.asyncio
async def test_upload_is_content_addressable(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    fake1 = io.BytesIO(b"identical content")
    r1 = await client.post("/api/files/upload",
                           files={"file": ("a.txt", fake1, "text/plain")})
    fake2 = io.BytesIO(b"identical content")
    r2 = await client.post("/api/files/upload",
                           files={"file": ("b.txt", fake2, "text/plain")})
    # Same hash -> same server_path (deduplicated).
    assert r1.json()["server_path"] == r2.json()["server_path"]


@pytest.mark.asyncio
async def test_browse_returns_listing(client, tmp_path, monkeypatch):
    # Seed a tmp dir structure we're allowed to browse.
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    resp = await client.get(f"/api/files/browse?path={tmp_path}")
    assert resp.status_code == 200
    names = {e["name"] for e in resp.json()["entries"]}
    assert {"a.txt", "sub"} <= names


@pytest.mark.asyncio
async def test_browse_rejects_path_traversal(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # /etc/passwd is outside the allowed root ($HOME).
    resp = await client.get("/api/files/browse?path=/etc/passwd")
    assert resp.status_code in (403, 400, 404)


@pytest.mark.asyncio
async def test_preview_stub_returns_404(client):
    resp = await client.get("/api/files/preview/nX/port0")
    assert resp.status_code == 404
