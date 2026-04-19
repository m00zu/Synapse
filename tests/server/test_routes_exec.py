import pytest
pytest.importorskip("PySide6")


@pytest.fixture(autouse=True, scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.mark.asyncio
async def test_run_returns_run_id(client):
    await client.post("/api/graph/nodes", json={"type": "ImageReadNode"})
    resp = await client.post("/api/exec/run")
    assert resp.status_code == 202
    assert "run_id" in resp.json()


@pytest.mark.asyncio
async def test_stop_is_idempotent(client):
    r1 = await client.post("/api/exec/stop")
    assert r1.status_code in (200, 204)
    r2 = await client.post("/api/exec/stop")
    assert r2.status_code in (200, 204)
