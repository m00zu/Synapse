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
