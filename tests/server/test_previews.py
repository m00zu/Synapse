"""Unit tests for synapse.server.previews — serialization of node outputs."""
import json
import pytest
from pathlib import Path


def test_write_previews_skips_non_preview_types(tmp_path):
    from synapse.server.previews import write_previews
    out = write_previews("n1", {"scalar": 42, "none": None}, tmp_path)
    assert out == []
    assert list(tmp_path.iterdir()) == []


def test_write_previews_writes_image_png(tmp_path):
    pytest.importorskip("PIL")
    import numpy as np
    from synapse.data_models import ImageData
    from synapse.server.previews import write_previews
    arr = np.zeros((100, 80, 3), dtype="float32")
    arr[..., 0] = 0.5
    img = ImageData(payload=arr)
    written = write_previews("n1", {"out": img}, tmp_path)
    assert written == [{"port": "out", "kind": "image"}]
    assert (tmp_path / "n1__out.png").is_file()


def test_write_previews_writes_table_json(tmp_path):
    pandas = pytest.importorskip("pandas")
    import pandas as pd
    from synapse.data_models import TableData
    from synapse.server.previews import write_previews
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl = TableData(payload=df)
    written = write_previews("n1", {"out": tbl}, tmp_path)
    assert written == [{"port": "out", "kind": "table"}]
    blob = json.loads((tmp_path / "n1__out.json").read_text())
    assert blob["columns"] == ["a", "b"]
    assert blob["total_rows"] == 3
    assert blob["rows"][0] == [1, "x"]


def test_write_previews_tolerates_serializer_failures(tmp_path, caplog):
    """A broken payload must not raise — it must log and skip."""
    from synapse.data_models import ImageData
    from synapse.server.previews import write_previews
    broken = ImageData(payload=None)
    with caplog.at_level("WARNING"):
        written = write_previews("n1", {"out": broken}, tmp_path)
    assert written == []
    assert any("preview:" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_preview_endpoint_serves_existing_file(client, tmp_path, monkeypatch):
    """The /api/files/preview/{id}/{port} endpoint returns the written PNG."""
    import numpy as np
    from synapse.data_models import ImageData
    from synapse.server.previews import write_previews
    # Use session's preview_dir so the route looks in the right place.
    from synapse.server.app import app
    session = app.state.session
    arr = np.zeros((50, 50, 3), dtype="float32")
    img = ImageData(payload=arr)
    write_previews("n1", {"out": img}, session.preview_dir)
    resp = await client.get("/api/files/preview/n1/out")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert len(resp.content) > 50  # any valid PNG is at least this large


@pytest.mark.asyncio
async def test_preview_endpoint_404_when_missing(client):
    resp = await client.get("/api/files/preview/never-ran/out")
    assert resp.status_code == 404
