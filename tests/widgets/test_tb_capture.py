"""Spec capture for PlotToolboxMixin._tb_* helpers."""
import pytest
pytest.importorskip("PySide6")

from synapse.widgets.spec import NumberField, CheckBox, ComboBox, TextField, Custom


@pytest.fixture(autouse=True, scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Install legacy shims so plot_nodes.py bare imports (data_models, nodes.base)
    # resolve correctly — same shims that catalog.py installs at startup.
    from synapse.widgets.catalog import _install_legacy_shims
    _install_legacy_shims()
    yield app


def _make_plot_node():
    """Instantiate BarPlotNode as a representative PlotToolboxMixin subclass."""
    from synapse.plugins.figure_plotting.plot_nodes import BarPlotNode
    return BarPlotNode()


def test_tb_text_appends_text_field_with_tab():
    n = _make_plot_node()
    before = len(n.get_widget_spec())
    n._tb_text("label", "Label", page="Axes", default="")
    spec = n.get_widget_spec()
    new = spec[before:]
    tf = [s for s in new if isinstance(s, TextField)][-1]
    assert tf.prop == "label"
    assert tf.tab == "Axes"


def test_tb_checkbox_appends_checkbox_with_tab():
    n = _make_plot_node()
    before = len(n.get_widget_spec())
    n._tb_checkbox("grid", "Show grid", page="Axes", default=True)
    spec = n.get_widget_spec()
    cb = [s for s in spec[before:] if isinstance(s, CheckBox)][-1]
    assert cb.prop == "grid"
    assert cb.default is True
    assert cb.tab == "Axes"


def test_tb_combo_appends_combobox_with_tab():
    n = _make_plot_node()
    before = len(n.get_widget_spec())
    n._tb_combo("palette", "Palette", page="Style", items=["tab10", "Set2"])
    cb = [s for s in n.get_widget_spec()[before:] if isinstance(s, ComboBox)][-1]
    assert cb.options == ["tab10", "Set2"]
    assert cb.tab == "Style"


def test_tb_spinbox_appends_number_field_with_tab():
    n = _make_plot_node()
    before = len(n.get_widget_spec())
    n._tb_spinbox("dpi", "DPI", page="Export",
                  default=300, min_val=50, max_val=1200, step=50, decimals=0)
    nf = [s for s in n.get_widget_spec()[before:] if isinstance(s, NumberField)][-1]
    assert nf.prop == "dpi" and nf.default == 300 and nf.decimals == 0
    assert nf.tab == "Export"


def test_tb_column_selector_appends_custom_with_tab():
    n = _make_plot_node()
    before = len(n.get_widget_spec())
    n._tb_column_selector("x_col", "X column", page="Data", default="")
    cu = [s for s in n.get_widget_spec()[before:] if isinstance(s, Custom)][-1]
    assert cu.component_id == "column_selector"
    assert cu.tab == "Data"


def test_tb_color_appends_custom_with_tab():
    n = _make_plot_node()
    before = len(n.get_widget_spec())
    n._tb_color("bar_color", "Bar color", page="Style")
    cu = [s for s in n.get_widget_spec()[before:] if isinstance(s, Custom)][-1]
    assert cu.component_id == "color_picker"
    assert cu.tab == "Style"


def test_tb_order_list_appends_custom_with_tab():
    n = _make_plot_node()
    before = len(n.get_widget_spec())
    n._tb_order_list("x_order", "X order", page="Data")
    cu = [s for s in n.get_widget_spec()[before:] if isinstance(s, Custom)][-1]
    assert cu.component_id == "order_list"
    assert cu.tab == "Data"


def test_plot_node_spec_is_nonempty_after_init():
    """Coverage gate: a fresh plot node should have >0 captured specs."""
    n = _make_plot_node()
    assert len(n.get_widget_spec()) > 0, \
        "BarPlotNode spec is empty — toolbox helpers aren't capturing specs"
