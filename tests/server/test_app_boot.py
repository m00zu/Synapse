import pytest
pytest.importorskip("PySide6")


@pytest.fixture(autouse=True, scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.mark.asyncio
async def test_root_returns_200(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "Synapse" in resp.text


@pytest.mark.asyncio
async def test_catalog_available_via_api_nodes(client):
    """Catalog is built lazily on first /api/nodes call."""
    resp = await client.get("/api/nodes")
    assert resp.status_code == 200
    catalog = resp.json()
    assert isinstance(catalog, dict)
    assert "GaussianBlurNode" in catalog
