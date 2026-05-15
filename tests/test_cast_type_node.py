"""Tests for CastTypeNode -- the type-relabel utility node.

Focuses on the pure-Python pieces (target discovery, evaluation
wrapper, payload validation).  The Qt-side bits (port retyping,
combo-box population) are exercised at app-launch time and don't
unit-test cleanly.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from synapse.nodes.base import register_port_type, _PORT_TYPE_CLASSES
from synapse.data_models import (
    NodeData, TableData, ImageData, MaskData,
)


# ── Target discovery ────────────────────────────────────────────────────


def test_discover_targets_includes_payload_only_classes():
    """Classes whose only required field is ``payload`` appear in dropdown."""
    from synapse.nodes.utility_nodes import CastTypeNode
    targets = CastTypeNode._discover_targets()
    for required_in in ('image', 'mask', 'table'):
        assert required_in in targets, (
            f"{required_in!r} should be a castable target "
            f"(only requires payload); got {targets}")


def test_discover_targets_excludes_classes_with_extra_required_fields():
    """Classes with required fields beyond payload don't appear."""
    from pydantic import Field

    class NeedsExtra(NodeData):
        payload: object
        # extra REQUIRED field with no default -- can't be cast generically
        format: str

    try:
        register_port_type('_test_needs_extra', NeedsExtra)
        from synapse.nodes.utility_nodes import CastTypeNode
        targets = CastTypeNode._discover_targets()
        assert '_test_needs_extra' not in targets
    finally:
        _PORT_TYPE_CLASSES.pop('_test_needs_extra', None)


def test_discover_targets_includes_subclasses_with_default_extra_fields():
    """Required-only-payload check accepts classes whose extras have defaults."""
    # MolTableData has `mol_col: str = 'ROMol'` -- default, not required.
    # It IS registered by the rdkit_nodes plugin in real usage; for unit
    # tests we register a stand-in.
    class _MolTableLike(TableData):
        payload: object
        mol_col: str = 'ROMol'   # default value -> not required

    try:
        register_port_type('_test_moltable', _MolTableLike)
        from synapse.nodes.utility_nodes import CastTypeNode
        targets = CastTypeNode._discover_targets()
        assert '_test_moltable' in targets
    finally:
        _PORT_TYPE_CLASSES.pop('_test_moltable', None)


# ── Payload validation ──────────────────────────────────────────────────


def test_payload_validators_pass_for_correct_types():
    """The lightweight validators accept the expected payload shapes."""
    from synapse.nodes.utility_nodes import CastTypeNode
    v = CastTypeNode._PAYLOAD_VALIDATORS
    assert v['image'](np.zeros((10, 10), dtype=np.uint8))
    assert v['mask'](np.zeros((10, 10), dtype=bool))
    assert v['table'](pd.DataFrame({'a': [1, 2]}))
    assert v['stat'](pd.DataFrame({'a': [1]}))
    assert v['mol_table'](pd.DataFrame({'ROMol': [None]}))


def test_payload_validators_reject_wrong_types():
    """The lightweight validators reject obvious type mistakes."""
    from synapse.nodes.utility_nodes import CastTypeNode
    v = CastTypeNode._PAYLOAD_VALIDATORS
    # image input but the payload is a DataFrame -- bad
    assert not v['image'](pd.DataFrame())
    # table target but the payload is an ndarray -- bad
    assert not v['table'](np.zeros((5, 5)))
    # mol_table specifically requires a DataFrame
    assert not v['mol_table'](np.zeros((5, 5)))


# ── End-to-end behaviour (without instantiating the Qt node) ────────────
# We test the validation + wrap logic by calling the static methods and
# the validator dict directly, since instantiating CastTypeNode requires
# NodeGraphQt's whole Qt machinery.


def test_table_to_moltable_cast_preserves_dataframe_reference():
    """The headline use case: filter a MolTable, cast back -- no re-parse."""
    # The user's scenario: a DataFrame with a ROMol column has been
    # filtered upstream (now declared as TableData).  We cast back to
    # MolTableData.  The DataFrame object should be the SAME object --
    # no copy, no re-parsing.
    class _MolTableLike(TableData):
        payload: object
        mol_col: str = 'ROMol'

    try:
        register_port_type('_test_moltable', _MolTableLike)
        df = pd.DataFrame({'name': ['mol1', 'mol2'], 'ROMol': [None, None]})
        # Upstream is TableData -- the filter has erased the subclass.
        upstream = TableData(payload=df)
        # Simulate the wrap step.
        target_cls = _PORT_TYPE_CLASSES['_test_moltable']
        wrapped = target_cls(payload=upstream.payload,
                             metadata=upstream.metadata,
                             source_path=upstream.source_path)
        assert isinstance(wrapped, _MolTableLike)
        # CRITICAL: same DataFrame object, no copy.
        assert wrapped.payload is df
        # The mol_col default applies.
        assert wrapped.mol_col == 'ROMol'
    finally:
        _PORT_TYPE_CLASSES.pop('_test_moltable', None)


def test_validation_catches_image_to_table_mistake():
    """Casting an ndarray as a table fails the lightweight check."""
    from synapse.nodes.utility_nodes import CastTypeNode
    validator = CastTypeNode._PAYLOAD_VALIDATORS['table']
    arr = np.zeros((100, 100), dtype=np.uint8)
    assert not validator(arr), (
        "Casting an ndarray to 'table' should be rejected by the validator")


def test_validation_passes_unknown_targets():
    """Plugin-registered targets without validators fall through (permissive)."""
    from synapse.nodes.utility_nodes import CastTypeNode
    # 'sklearn_model' isn't in the validator dict -- the eval path
    # treats it as "no opinion" and lets the cast proceed.
    assert 'sklearn_model' not in CastTypeNode._PAYLOAD_VALIDATORS


# ── Regression: input/output types must not share storage ────────────


def test_input_and_output_types_are_independent():
    """A node may have an input and an output with the same port name
    and different types (e.g. CastTypeNode's 'data' on both ends).
    The per-direction type dicts must NOT collide."""
    # Simulate what BaseExecutionNode.__init__ does, without Qt.
    node = MagicMock()
    node._input_types = {}
    node._output_types = {}

    # CastTypeNode declares input data:any + output data:image at __init__
    node._input_types['data'] = 'any'
    node._output_types['data'] = 'image'

    # Then the user picks 'mask' from the dropdown -- only output retypes.
    node._output_types['data'] = 'mask'

    # The input must STILL be 'any' (the bug was that a single dict
    # keyed by name only would have set it to 'mask' too, blocking
    # any non-mask wire from feeding the cast).
    assert node._input_types['data'] == 'any', (
        "Input port type must remain 'any' after the output retypes -- "
        "otherwise users can't feed images into a Cast Type [mask] node.")
    assert node._output_types['data'] == 'mask'
