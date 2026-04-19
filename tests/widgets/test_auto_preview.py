"""Auto-emit Preview specs for image/table/figure output ports.

Task 2 of Synapse Web Phase 1d: every node whose output port type is image,
table, or figure automatically gets a Preview spec in the catalog, without
the node author touching their class.

Port-name resolution uses the actual instance outputs() dict (runtime names)
and falls back to PORT_SPEC for the type.  FileReadNode and ImageReadNode
both register their output as 'out' at runtime even though PORT_SPEC lists
['table'] and ['image'] respectively, so the Preview sources should be
'output:out' — matching the executor's <node>__out.png / <node>__out.json
naming convention.
"""
import pytest

pytest.importorskip("PySide6")


@pytest.fixture(autouse=True, scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(scope="module")
def catalog(qapp):
    from synapse.widgets.catalog import collect_widget_catalog
    return collect_widget_catalog()


# ---------------------------------------------------------------------------
# Test 1: GaussianBlurNode emits an image Preview with source='output:image'
# ---------------------------------------------------------------------------

def test_gaussian_blur_has_image_preview(catalog):
    """GaussianBlurNode outputs an image port named 'image' — auto-Preview
    should appear with preview_kind='image' and source='output:image'."""
    specs = catalog["GaussianBlurNode"]
    previews = [s for s in specs if s["kind"] == "Preview"]
    assert previews, "GaussianBlurNode should have at least one Preview spec"
    assert any(
        p["preview_kind"] == "image" and p["source"] == "output:image"
        for p in previews
    ), f"Expected image Preview with source='output:image'; got: {previews}"


# ---------------------------------------------------------------------------
# Test 2: FileReadNode emits a table Preview
# ---------------------------------------------------------------------------

def test_filereadnode_has_table_preview(catalog):
    """FileReadNode's PORT_SPEC lists ['table'] but the actual output port is
    named 'out'.  The catalog should emit a Preview with preview_kind='table'
    and source='output:out' (matching executor's <node>__out.json)."""
    specs = catalog["FileReadNode"]
    previews = [s for s in specs if s["kind"] == "Preview"]
    assert previews, "FileReadNode should have at least one Preview spec"
    # The important invariant: there is a table preview.
    assert any(p["preview_kind"] == "table" for p in previews), (
        f"FileReadNode should have a table Preview; got: {previews}"
    )
    # The source should point to the actual runtime port name 'out', not
    # the PORT_SPEC type string 'table'.
    assert any(
        p["preview_kind"] == "table" and p["source"] == "output:out"
        for p in previews
    ), (
        f"Expected table Preview with source='output:out' (runtime port name); "
        f"got: {previews}"
    )


# ---------------------------------------------------------------------------
# Test 3: At least one node in the catalog has no Preview spec
# ---------------------------------------------------------------------------

def test_non_preview_node_has_no_preview(catalog):
    """Nodes that output neither image, table, nor figure must not receive
    auto-emitted Preview specs.  There should be at least one such node
    (e.g. SaveNode, FolderIteratorNode, BatchGateNode, etc.)."""
    no_preview = [
        name for name, specs in catalog.items()
        if not any(s["kind"] == "Preview" for s in specs)
    ]
    assert no_preview, (
        "Expected at least one node without any Preview spec "
        "(e.g. a utility/sink node that outputs nothing)."
    )
