import pytest
pytest.importorskip("PySide6")



@pytest.mark.asyncio
async def test_get_nodes_returns_catalog(client):
    resp = await client.get("/api/nodes")
    assert resp.status_code == 200
    body = resp.json()
    assert "GaussianBlurNode" in body
    gb = body["GaussianBlurNode"]
    assert isinstance(gb, list)
    assert all("kind" in s for s in gb)


@pytest.mark.asyncio
async def test_get_node_categories_returns_identifier_and_category(client):
    resp = await client.get("/api/nodes/categories")
    assert resp.status_code == 200
    body = resp.json()
    gb = body.get("GaussianBlurNode")
    assert gb is not None
    assert gb["identifier"].startswith("nodes.image_process")
    assert gb["category"] == "Image"
    # Spot-check other categories
    assert body["FileReadNode"]["category"] in ("Table", "I/O")  # file read
    assert body["ImageReadNode"]["category"] == "I/O"
    assert body["BarPlotNode"]["category"] == "Plot"
