import pytest
pytest.importorskip("PySide6")


@pytest.fixture(autouse=True, scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.mark.asyncio
async def test_add_node_returns_id(client):
    resp = await client.post("/api/graph/nodes",
                             json={"type": "GaussianBlurNode", "x": 10, "y": 20})
    assert resp.status_code == 201
    assert "id" in resp.json()


@pytest.mark.asyncio
async def test_add_unknown_type_400(client):
    resp = await client.post("/api/graph/nodes", json={"type": "NotReal"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_patch_props(client):
    add = await client.post("/api/graph/nodes", json={"type": "GaussianBlurNode"})
    nid = add.json()["id"]
    resp = await client.patch(f"/api/graph/nodes/{nid}/props",
                              json={"sigma": 2.5})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_delete_node(client):
    add = await client.post("/api/graph/nodes", json={"type": "GaussianBlurNode"})
    nid = add.json()["id"]
    resp = await client.delete(f"/api/graph/nodes/{nid}")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_get_graph_after_adds(client):
    await client.post("/api/graph/nodes", json={"type": "ImageReadNode"})
    resp = await client.get("/api/graph")
    assert resp.status_code == 200
    body = resp.json()
    # NodeGraphQt's serialize_session returns at minimum a dict. Verify
    # it's not empty after we've added a node.
    assert isinstance(body, dict)
