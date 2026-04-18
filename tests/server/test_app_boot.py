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
async def test_app_has_catalog_attached(client):
    from synapse.server.app import app
    assert hasattr(app.state, "catalog")
    assert isinstance(app.state.catalog, dict)
    assert "GaussianBlurNode" in app.state.catalog
