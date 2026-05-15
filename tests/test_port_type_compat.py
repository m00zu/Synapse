"""Tests for Rust-style port-type compatibility (Liskov substitution).

Covers ``is_port_type_compatible`` in ``synapse/nodes/base.py``:

  - exact-string match
  - upcasts via ``issubclass`` (e.g. mask -> image)
  - downcasts rejected (image -> mask)
  - sibling-type rejection (mask -> label)
  - ``'any'`` wildcard
  - unregistered types fall through to exact-match only
  - permissive on empty/missing type
"""
from __future__ import annotations

import pytest

from synapse.nodes.base import (
    is_port_type_compatible, register_port_type, _PORT_TYPE_CLASSES,
)
from synapse.data_models import (
    NodeData, TableData, StatData, ImageData, MaskData, SkeletonData,
)


# ── Core types (assumed registered by base.py at import) ────────────────


def test_exact_match_allowed():
    assert is_port_type_compatible('image', 'image')
    assert is_port_type_compatible('table', 'table')
    assert is_port_type_compatible('mask', 'mask')


def test_upcast_allowed():
    # MaskData IS A ImageData -- mask output may feed an image input.
    assert is_port_type_compatible('mask', 'image')
    # SkeletonData IS A MaskData IS A ImageData -- transitive.
    assert is_port_type_compatible('skeleton', 'mask')
    assert is_port_type_compatible('skeleton', 'image')
    # StatData IS A TableData.
    assert is_port_type_compatible('stat', 'table')


def test_downcast_rejected():
    # Plain ImageData is NOT a MaskData.
    assert not is_port_type_compatible('image', 'mask')
    # TableData is NOT a StatData.
    assert not is_port_type_compatible('table', 'stat')
    # ImageData -> SkeletonData rejected.
    assert not is_port_type_compatible('image', 'skeleton')


def test_sibling_rejected():
    # mask and label both descend from NodeData but neither IS A the other.
    assert not is_port_type_compatible('mask', 'label')
    assert not is_port_type_compatible('label', 'mask')
    # image and table -- entirely different lineages.
    assert not is_port_type_compatible('image', 'table')
    assert not is_port_type_compatible('table', 'image')


def test_any_wildcard():
    assert is_port_type_compatible('any', 'image')
    assert is_port_type_compatible('image', 'any')
    assert is_port_type_compatible('any', 'any')
    # 'any' on either side beats sibling rejection.
    assert is_port_type_compatible('any', 'table')
    assert is_port_type_compatible('mask', 'any')


def test_permissive_on_empty():
    # Pre-typing nodes or ports without a recorded type fall through
    # to permissive behaviour so legacy graphs keep loading.
    assert is_port_type_compatible('', 'image')
    assert is_port_type_compatible('image', '')
    assert is_port_type_compatible('', '')


def test_unregistered_same_name_allowed():
    # An unregistered port type still matches itself by exact string.
    assert is_port_type_compatible('plugin_custom', 'plugin_custom')


def test_unregistered_different_names_rejected():
    # Two different unregistered names -- no class info, strict no.
    assert not is_port_type_compatible('plugin_a', 'plugin_b')


def test_one_registered_one_not_rejected():
    # Different names, only one side has class info -- still strict no
    # (the unregistered side has no subtype info to upcast through).
    assert not is_port_type_compatible('image', 'plugin_unknown')
    assert not is_port_type_compatible('plugin_unknown', 'image')


# ── Plugin registration is idempotent and works for new types ──────────


def test_register_port_type_is_idempotent():
    # Calling register_port_type twice with the same args is safe.
    register_port_type('table', TableData)
    register_port_type('table', TableData)
    assert _PORT_TYPE_CLASSES['table'] is TableData


def test_plugin_subtype_registration():
    # Simulate a plugin defining a new TableData subclass.
    class PluginTable(TableData):
        pass
    try:
        register_port_type('plugin_table', PluginTable)
        # Upcast to TableData allowed.
        assert is_port_type_compatible('plugin_table', 'table')
        # Downcast rejected.
        assert not is_port_type_compatible('table', 'plugin_table')
        # Sibling rejection still holds (PluginTable is not an ImageData).
        assert not is_port_type_compatible('plugin_table', 'image')
    finally:
        _PORT_TYPE_CLASSES.pop('plugin_table', None)


def test_plugin_double_subtype_upcast_works():
    # A plugin subtype of a core subtype (rare but legal) -- the upcast
    # should still walk all the way up the class hierarchy via issubclass.
    class PluginMask(MaskData):  # PluginMask <: MaskData <: ImageData
        pass
    try:
        register_port_type('plugin_mask', PluginMask)
        assert is_port_type_compatible('plugin_mask', 'mask')
        assert is_port_type_compatible('plugin_mask', 'image')
        # Downcasts still rejected.
        assert not is_port_type_compatible('mask', 'plugin_mask')
        assert not is_port_type_compatible('image', 'plugin_mask')
    finally:
        _PORT_TYPE_CLASSES.pop('plugin_mask', None)
