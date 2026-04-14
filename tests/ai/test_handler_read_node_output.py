import numpy as np
import pandas as pd

from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.tool_handlers.read_node_output import make_read_node_output_handler


class _TableStub:
    def __init__(self, df):
        self.df = df


class _ImageStub:
    def __init__(self, arr):
        self.payload = arr


def test_read_table_output_returns_head_and_shape():
    g = FakeGraph()
    n = FakeNode("n1", "SomeNode")
    n.output_values["out_1"] = _TableStub(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    n.add_output("out_1")
    g.add_node(n)
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_id": "n1"})
    assert out["kind"] == "table"
    assert out["metadata"]["shape"] == [3, 2]
    assert "a" in out["text_preview"] and "b" in out["text_preview"]
    assert "thumbnail" not in out


def test_read_image_output_reports_shape_dtype_range_no_vision():
    g = FakeGraph()
    n = FakeNode("n1", "SomeNode")
    n.output_values["out_1"] = _ImageStub(np.zeros((10, 10), dtype=np.uint8))
    n.add_output("out_1")
    g.add_node(n)
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_id": "n1"})
    assert out["kind"] == "image"
    assert out["metadata"]["shape"] == [10, 10]
    assert out["metadata"]["dtype"] == "uint8"
    assert "thumbnail" not in out


def test_read_image_output_includes_thumbnail_when_vision_supported():
    __import__("PIL.Image", fromlist=["Image"])
    g = FakeGraph()
    n = FakeNode("n1", "SomeNode")
    n.output_values["out_1"] = _ImageStub(np.zeros((10, 10), dtype=np.uint8))
    n.add_output("out_1")
    g.add_node(n)
    handler = make_read_node_output_handler(g, supports_vision=lambda: True)
    out = handler({"node_id": "n1"})
    assert "thumbnail" in out
    import base64
    assert base64.b64decode(out["thumbnail"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_read_unknown_node_returns_error():
    g = FakeGraph()
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_id": "does_not_exist"})
    assert "error" in out


def test_read_node_with_no_output_values_returns_empty_marker():
    g = FakeGraph()
    n = FakeNode("n1", "Empty")
    n.add_output("out_1")
    g.add_node(n)
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_id": "n1"})
    assert out.get("kind") == "empty"


# --- multi-node (node_ids) batch path ------------------------------------

def _build_table_node(node_id: str, df: pd.DataFrame, graph: FakeGraph) -> FakeNode:
    n = FakeNode(node_id, "TableSrc")
    n.output_values["out_1"] = _TableStub(df)
    n.add_output("out_1")
    graph.add_node(n)
    return n


def _build_image_node(node_id: str, arr: np.ndarray, graph: FakeGraph) -> FakeNode:
    n = FakeNode(node_id, "ImgSrc")
    n.output_values["out_1"] = _ImageStub(arr)
    n.add_output("out_1")
    graph.add_node(n)
    return n


def test_node_ids_returns_results_list_one_per_node():
    g = FakeGraph()
    _build_table_node("a", pd.DataFrame({"x": [1, 2]}), g)
    _build_table_node("b", pd.DataFrame({"y": [3, 4]}), g)
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_ids": ["a", "b"]})
    assert "results" in out and len(out["results"]) == 2
    assert {r["node_id"] for r in out["results"]} == {"a", "b"}
    assert all(r["kind"] == "table" for r in out["results"])
    assert out["truncated"] is False


def test_node_ids_unknown_node_reports_per_entry_error():
    g = FakeGraph()
    _build_table_node("a", pd.DataFrame({"x": [1]}), g)
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_ids": ["a", "ghost"]})
    assert len(out["results"]) == 2
    by_id = {r["node_id"]: r for r in out["results"]}
    assert by_id["a"]["kind"] == "table"
    assert "error" in by_id["ghost"]


def test_node_ids_thumbnail_skipped_when_three_or_more_images_in_batch():
    g = FakeGraph()
    for nid in ("i1", "i2", "i3"):
        _build_image_node(nid, np.zeros((10, 10), dtype=np.uint8), g)
    handler = make_read_node_output_handler(g, supports_vision=lambda: True)
    out = handler({"node_ids": ["i1", "i2", "i3"]})
    assert out["included_thumbnails"] is False
    assert all("thumbnail" not in r for r in out["results"])


def test_node_ids_thumbnail_included_when_two_images_with_vision():
    __import__("PIL.Image", fromlist=["Image"])
    g = FakeGraph()
    _build_image_node("i1", np.zeros((10, 10), dtype=np.uint8), g)
    _build_image_node("i2", np.zeros((10, 10), dtype=np.uint8), g)
    handler = make_read_node_output_handler(g, supports_vision=lambda: True)
    out = handler({"node_ids": ["i1", "i2"]})
    assert out["included_thumbnails"] is True
    assert all("thumbnail" in r for r in out["results"])


def test_node_ids_thumbnail_skipped_without_vision_even_for_one_image():
    g = FakeGraph()
    _build_image_node("i1", np.zeros((10, 10), dtype=np.uint8), g)
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_ids": ["i1"]})
    assert out["included_thumbnails"] is False
    assert "thumbnail" not in out["results"][0]


def test_node_ids_rejects_too_many():
    g = FakeGraph()
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_ids": [f"n{i}" for i in range(9)]})
    assert "error" in out and "8" in out["error"]


def test_node_ids_rejects_empty_list():
    g = FakeGraph()
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_ids": []})
    assert "error" in out


def test_rejects_both_node_id_and_node_ids():
    g = FakeGraph()
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({"node_id": "a", "node_ids": ["a"]})
    assert "error" in out


def test_rejects_neither_field():
    g = FakeGraph()
    handler = make_read_node_output_handler(g, supports_vision=lambda: False)
    out = handler({})
    assert "error" in out


def test_node_ids_truncation_clears_text_previews_first():
    """When a batch overflows, the handler should first clear text_previews
    (cheaper than dropping whole results) and mark truncated=True."""
    g = FakeGraph()
    big = pd.DataFrame({"col" + str(i): ["x" * 200] * 20 for i in range(10)})
    for nid in ("a", "b", "c", "d"):
        _build_table_node(nid, big, g)
    handler = make_read_node_output_handler(
        g, supports_vision=lambda: False, token_cap=300,
    )
    out = handler({"node_ids": ["a", "b", "c", "d"]})
    assert out["truncated"] is True
    # At least one preview was cleared (the cheap-trim path fired).
    assert any(r.get("text_preview") == "" for r in out["results"])


def test_node_ids_truncation_pops_results_when_cap_too_tight():
    """If even cleared previews exceed the cap, whole results get popped."""
    g = FakeGraph()
    # 5 nodes each with 60 long-name columns — metadata alone is heavy.
    big = pd.DataFrame({f"longish_column_name_{i}": [1] for i in range(60)})
    for nid in ("a", "b", "c", "d", "e"):
        _build_table_node(nid, big, g)
    handler = make_read_node_output_handler(
        g, supports_vision=lambda: False, token_cap=200,  # very tight
    )
    out = handler({"node_ids": ["a", "b", "c", "d", "e"]})
    assert out["truncated"] is True
    assert 0 < len(out["results"]) < 5
