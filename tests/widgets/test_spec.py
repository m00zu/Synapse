"""Unit tests for synapse.widgets.spec — pure-Python, no Qt."""
import pytest
from synapse.widgets.spec import (
    WidgetSpec, VerticalLayout, HorizontalLayout,
    ComboBox, NumberField, CheckBox, TextField, FilePath,
    Button, Progress, Preview, Custom,
    spec_to_json, spec_from_json,
)


def test_number_field_fields():
    s = NumberField(prop="sigma", label="Sigma", min=0.0, max=10.0,
                    step=0.1, decimals=3, default=1.0)
    assert s.kind == "NumberField"
    assert s.prop == "sigma"
    assert s.decimals == 3


def test_combo_box_requires_options():
    s = ComboBox(prop="mode", label="Mode", options=["a", "b"], default="a")
    assert s.options == ["a", "b"]


def test_checkbox_defaults_to_false():
    s = CheckBox(prop="flag", label="Flag")
    assert s.default is False


def test_layout_holds_children():
    tree = VerticalLayout([
        NumberField(prop="x", label="X", default=0),
        CheckBox(prop="y", label="Y"),
    ])
    assert len(tree.children) == 2
    assert tree.children[0].kind == "NumberField"


def test_custom_carries_component_id():
    s = Custom(component_id="python_script_editor", props={"n_inputs": 1})
    assert s.component_id == "python_script_editor"
    assert s.props == {"n_inputs": 1}


def test_spec_to_json_roundtrip_leaf():
    original = NumberField(prop="sigma", label="Sigma", min=0.0, max=10.0,
                           step=0.1, decimals=3, default=1.0)
    blob = spec_to_json(original)
    assert blob["kind"] == "NumberField"
    assert blob["prop"] == "sigma"
    restored = spec_from_json(blob)
    assert restored == original


def test_spec_to_json_roundtrip_nested():
    original = VerticalLayout([
        NumberField(prop="x", label="X", default=0),
        HorizontalLayout([
            CheckBox(prop="a", label="A"),
            CheckBox(prop="b", label="B"),
        ]),
    ])
    blob = spec_to_json(original)
    restored = spec_from_json(blob)
    assert restored == original


def test_spec_from_json_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown widget kind"):
        spec_from_json({"kind": "DoesNotExist"})


def test_filepath_has_mode_with_either_default():
    s = FilePath(prop="path", label="Input", mode="either")
    assert s.mode == "either"
    s2 = FilePath(prop="path", label="Input")
    assert s2.mode == "either"


def test_preview_carries_source_and_kind():
    s = Preview(preview_kind="image", source="output:image")
    assert s.kind == "Preview"        # the widget kind
    assert s.preview_kind == "image"  # the media kind it shows
    assert s.source == "output:image"


def test_button_requires_action_and_label():
    s = Button(action="reset_roi", label="Reset")
    assert s.kind == "Button"
    assert s.action == "reset_roi"


def test_progress_defaults_empty_label():
    s = Progress(prop="pct")
    assert s.kind == "Progress"
    assert s.label == ""


def test_spec_to_json_roundtrip_custom_with_empty_props():
    c = Custom(component_id="foo")
    assert spec_from_json(spec_to_json(c)) == c
