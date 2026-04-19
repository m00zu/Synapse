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
    assert gb["display_name"]  # non-empty human-readable name
    # Spot-check other categories
    assert body["FileReadNode"]["category"] in ("Table", "I/O")  # file read
    assert body["ImageReadNode"]["category"] == "I/O"
    assert body["BarPlotNode"]["category"] == "Plot"
    # FileReadNode's NODE_NAME is "Table Reader" on desktop.
    assert body["FileReadNode"]["display_name"] == "Table Reader"


@pytest.mark.asyncio
async def test_get_node_categories_exposes_per_port_info(client):
    resp = await client.get("/api/nodes/categories")
    body = resp.json()
    # SplitRGBNode has one image input and three image outputs
    # (red, green, blue) — all should appear in the response.
    split = body["SplitRGBNode"]
    assert split["inputs"] == [{"name": "image", "type": "image"}]
    out_names = [p["name"] for p in split["outputs"]]
    assert out_names == ["red", "green", "blue"]
    assert all(p["type"] == "image" for p in split["outputs"])
