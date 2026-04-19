"""Verify that calling node helper methods populates self._spec_builder."""
import pytest

pytest.importorskip("PySide6")  # helpers depend on NodeGraphQt which needs Qt

from synapse.widgets.spec import (
    NumberField, CheckBox, ComboBox, TextField, HorizontalLayout, Custom,
)


@pytest.fixture(autouse=True, scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_node():
    """Construct a BaseExecutionNode subclass for testing without running a
    real Synapse node __init__ (which would register Qt widgets we don't need)."""
    from synapse.nodes.base import BaseExecutionNode
    class _TestNode(BaseExecutionNode):
        __identifier__ = "test.widgets"
        NODE_NAME = "TestNode"
        def __init__(self):
            super().__init__()
    return _TestNode()


def test_add_int_spinbox_appends_number_field():
    n = _make_node()
    n._add_int_spinbox("sigma", "Sigma", value=5, min_val=0, max_val=100, step=1)
    spec = n.get_widget_spec()
    assert any(isinstance(s, NumberField) and s.prop == "sigma" for s in spec)
    nf = [s for s in spec if isinstance(s, NumberField) and s.prop == "sigma"][0]
    assert nf.min == 0 and nf.max == 100 and nf.step == 1 and nf.default == 5
    assert nf.decimals == 0  # integer


def test_add_float_spinbox_appends_decimal_number_field():
    n = _make_node()
    n._add_float_spinbox("thresh", "Threshold", value=0.5,
                         min_val=0.0, max_val=1.0, step=0.01, decimals=3)
    spec = n.get_widget_spec()
    nf = [s for s in spec if isinstance(s, NumberField) and s.prop == "thresh"][0]
    assert nf.decimals == 3
    assert nf.default == 0.5


def test_add_checkbox_appends_checkbox():
    n = _make_node()
    n.add_checkbox("flag", "", text="Enable feature", state=True)
    spec = n.get_widget_spec()
    cb = [s for s in spec if isinstance(s, CheckBox)][0]
    assert cb.prop == "flag"
    assert cb.label == "Enable feature"
    assert cb.default is True


def test_add_combo_menu_appends_combobox():
    n = _make_node()
    n.add_combo_menu("mode", "Mode", items=["Keep", "Remove"])
    spec = n.get_widget_spec()
    cb = [s for s in spec if isinstance(s, ComboBox)][0]
    assert cb.options == ["Keep", "Remove"]
    assert cb.prop == "mode"


def test_add_text_input_appends_text_field():
    n = _make_node()
    n.add_text_input("pattern", "Pattern", text="*.csv")
    spec = n.get_widget_spec()
    tf = [s for s in spec if isinstance(s, TextField)][0]
    assert tf.prop == "pattern"
    assert tf.default == "*.csv"


def test_add_row_appends_horizontal_layout():
    n = _make_node()
    n._add_row("size", "Size", fields=[
        {"name": "w", "label": "W", "type": "int", "value": 100, "min_val": 0, "max_val": 1000, "step": 1},
        {"name": "h", "label": "H", "type": "int", "value": 200, "min_val": 0, "max_val": 1000, "step": 1},
    ])
    spec = n.get_widget_spec()
    hl = [s for s in spec if isinstance(s, HorizontalLayout)][0]
    assert len(hl.children) == 2
    assert all(isinstance(c, NumberField) for c in hl.children)
    assert [c.prop for c in hl.children] == ["w", "h"]


def test_add_column_selector_appends_custom():
    n = _make_node()
    n._add_column_selector("x_col", "X column", text="", mode="single")
    spec = n.get_widget_spec()
    cu = [s for s in spec if isinstance(s, Custom)][0]
    assert cu.component_id == "column_selector"
    assert cu.props["prop"] == "x_col"
    assert cu.props["mode"] == "single"


def test_get_widget_spec_returns_builder_list_order():
    n = _make_node()
    n._add_int_spinbox("a", "A", value=1)
    n.add_checkbox("b", "", text="B", state=False)
    n._add_float_spinbox("c", "C", value=0.0)
    spec = n.get_widget_spec()
    props_in_order = [getattr(s, "prop", None) for s in spec]
    assert props_in_order == ["a", "b", "c"]


def test_spec_builder_initialized_empty():
    n = _make_node()
    assert n.get_widget_spec() == []


def test_add_custom_widget_captures_nodefileselector_as_filepath():
    """NodeFileSelector (FileReadNode's path picker) is now a FilePath spec."""
    from synapse.nodes.base import NodeFileSelector
    from synapse.widgets.spec import FilePath
    n = _make_node()
    # Use the class the real FileReadNode uses.
    sel = NodeFileSelector(n.view, name="file_path", label="Input file")
    n.add_custom_widget(sel, tab="Properties")
    spec = n.get_widget_spec()
    fp = [s for s in spec if isinstance(s, FilePath)]
    assert len(fp) == 1
    assert fp[0].prop == "file_path"
    assert fp[0].label == "Input file"
    assert fp[0].mode == "either"


def test_add_custom_widget_captures_channel_selector_as_custom():
    from synapse.nodes.base import NodeChannelSelectorWidget
    n = _make_node()
    ch = NodeChannelSelectorWidget(n.view, name="channels", label="Channels",
                                   text="1,2,3")
    n.add_custom_widget(ch, tab="Properties")
    spec = n.get_widget_spec()
    cu = [s for s in spec if isinstance(s, Custom)
          and s.component_id == "channel_selector"]
    assert len(cu) == 1
    assert cu[0].props["prop"] == "channels"


def test_filereadnode_captures_filepath_and_separator():
    """Integration: a real FileReadNode's spec should include BOTH the path
    FilePath and the separator TextField."""
    from synapse.nodes.io_nodes import FileReadNode
    from synapse.widgets.spec import FilePath
    n = FileReadNode()
    spec = n.get_widget_spec()
    props = {s.prop: s for s in spec if hasattr(s, "prop")}
    assert "file_path" in props
    assert isinstance(props["file_path"], FilePath)
    assert "separator" in props
    assert isinstance(props["separator"], TextField)
