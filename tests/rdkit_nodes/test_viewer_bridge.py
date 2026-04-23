"""Unit tests for ViewerBridge — JSON → atom_clicked signal dispatch."""
import json

import pytest

pytest.importorskip("PySide6.QtCore")
pytest.importorskip("PySide6.QtWebChannel")

from PySide6 import QtCore  # noqa: E402

from plugins.rdkit_nodes.viewer_nodes import ViewerBridge


def _make_spy(bridge):
    """Collect every dict emitted on atom_clicked into a list."""
    received = []
    bridge.atom_clicked.connect(lambda d: received.append(d))
    return received


def test_valid_json_is_dispatched_as_dict():
    bridge = ViewerBridge()
    received = _make_spy(bridge)
    payload = {
        "x": 1.5, "y": -2.0, "z": 0.25,
        "chain": "A", "resn": "ALA", "resi": 42,
        "atom": "CA", "elem": "C",
    }
    bridge.onAtomClicked(json.dumps(payload))
    assert received == [payload]


def test_malformed_json_is_ignored_silently():
    bridge = ViewerBridge()
    received = _make_spy(bridge)
    bridge.onAtomClicked("{not valid json")
    bridge.onAtomClicked("")
    bridge.onAtomClicked(None)  # type: ignore[arg-type]
    assert received == []


def test_signal_delivers_plain_dict_not_json_string():
    bridge = ViewerBridge()
    received = _make_spy(bridge)
    bridge.onAtomClicked('{"x": 1, "y": 2, "z": 3}')
    assert isinstance(received[0], dict)
    assert received[0]["x"] == 1
