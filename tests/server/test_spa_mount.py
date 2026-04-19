"""Serve the built React SPA from / when synapse/web/dist is present."""
import pytest
pytest.importorskip("PySide6")


@pytest.mark.asyncio
async def test_spa_served_when_dist_present(tmp_path, monkeypatch):
    """Patch _DIST to a synthesized dist dir, rebuild app, verify / returns it."""
    from unittest.mock import MagicMock
    from synapse.server import app as app_mod
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>SPA-OK</body></html>")
    monkeypatch.setattr(app_mod, "_DIST", dist)

    rebuilt = app_mod.create_app()
    # Wire app.state without spinning up a real SessionState (Qt must stay on
    # main thread; instantiating it inside an async test aborts the process).
    rebuilt.state.catalog = None
    rebuilt.state.session = MagicMock()

    from httpx import ASGITransport, AsyncClient
    async with AsyncClient(transport=ASGITransport(app=rebuilt),
                           base_url="http://t") as c:
        resp = await c.get("/")
    assert resp.status_code == 200
    assert "SPA-OK" in resp.text


@pytest.mark.asyncio
async def test_placeholder_served_when_dist_missing(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from synapse.server import app as app_mod
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(app_mod, "_DIST", missing)

    rebuilt = app_mod.create_app()
    rebuilt.state.catalog = None
    rebuilt.state.session = MagicMock()

    from httpx import ASGITransport, AsyncClient
    async with AsyncClient(transport=ASGITransport(app=rebuilt),
                           base_url="http://t") as c:
        resp = await c.get("/")
    assert resp.status_code == 200
    assert "Synapse serve" in resp.text  # placeholder content
