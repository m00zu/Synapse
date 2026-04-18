"""Catalog export walks all registered node classes and returns their specs."""
import json
import pytest

pytest.importorskip("PySide6")

from synapse.widgets.spec import NumberField, ComboBox, CheckBox


@pytest.fixture(autouse=True, scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(scope="module")
def catalog(qapp):
    # Import Synapse nodes so subclass registration happens.
    import synapse.nodes  # noqa
    import synapse.plugins  # noqa
    from synapse.widgets.catalog import collect_widget_catalog
    return collect_widget_catalog()


def test_collect_widget_catalog_returns_dict_keyed_by_class_name(catalog):
    assert isinstance(catalog, dict)
    assert "GaussianBlurNode" in catalog
    # Every value is a list of spec dicts (JSON-serializable).
    for cls_name, specs in catalog.items():
        assert isinstance(specs, list)
        assert all(isinstance(s, dict) and "kind" in s for s in specs)


def test_collect_widget_catalog_gaussian_blur_has_expected_shape(catalog):
    gb = catalog["GaussianBlurNode"]
    # Should have at least one NumberField for 'sigma'.
    sigma_entries = [s for s in gb if s["kind"] == "NumberField" and s.get("prop") == "sigma"]
    assert len(sigma_entries) == 1


def test_collect_widget_catalog_is_json_serializable(catalog):
    serialized = json.dumps(catalog)
    assert len(serialized) > 100  # non-empty
    deserialized = json.loads(serialized)
    assert deserialized == catalog


def test_collect_widget_catalog_coverage_gate(catalog):
    """Every BaseExecutionNode subclass that shows up in the runtime schema
    must appear in the catalog. Guards against registration drift."""
    import json as _json
    from pathlib import Path
    schema_path = Path(__file__).parent.parent.parent / "synapse" / "llm_node_schema.json"
    schema_nodes = set(_json.loads(schema_path.read_text())["node_catalog"].keys())
    cat = set(catalog.keys())
    missing = schema_nodes - cat
    # Some nodes may have no widgets at all (empty spec) — still listed with [].
    assert not missing, f"nodes in schema but not in catalog: {sorted(missing)[:10]}"
