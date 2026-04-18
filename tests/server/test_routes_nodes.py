import pytest
pytest.importorskip("PySide6")


@pytest.fixture(autouse=True, scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.mark.asyncio
async def test_get_nodes_returns_catalog(client):
    resp = await client.get("/api/nodes")
    assert resp.status_code == 200
    body = resp.json()
    assert "GaussianBlurNode" in body
    gb = body["GaussianBlurNode"]
    assert isinstance(gb, list)
    assert all("kind" in s for s in gb)
