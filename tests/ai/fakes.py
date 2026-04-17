"""Lightweight fakes for handler unit tests. No Qt / NodeGraphQt required."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakePort:
    _name: str
    parent: "FakeNode"
    peers: list["FakePort"] = field(default_factory=list)

    def name(self) -> str:
        return self._name

    def node(self) -> "FakeNode":
        return self.parent

    def connected_ports(self) -> list["FakePort"]:
        return list(self.peers)

    def connect_to(self, other: "FakePort") -> None:
        if other not in self.peers:
            self.peers.append(other)
            other.peers.append(self)


@dataclass
class _Model:
    custom_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeNode:
    id: str
    type_name: str
    _inputs: dict[str, FakePort] = field(default_factory=dict)
    _outputs: dict[str, FakePort] = field(default_factory=dict)
    model: _Model = field(default_factory=_Model)
    output_values: dict[str, Any] = field(default_factory=dict)
    _last_error: str | None = None

    def __init__(self, node_id: str, type_name: str, props: dict | None = None):
        self.id = node_id
        self.type_name = type_name
        self._inputs = {}
        self._outputs = {}
        self.model = _Model(custom_properties=dict(props or {}))
        self.output_values = {}
        self._last_error = None

    def add_input(self, name: str) -> FakePort:
        p = FakePort(_name=name, parent=self)
        self._inputs[name] = p
        return p

    def add_output(self, name: str) -> FakePort:
        p = FakePort(_name=name, parent=self)
        self._outputs[name] = p
        return p

    def inputs(self) -> dict[str, FakePort]:
        return self._inputs

    def outputs(self) -> dict[str, FakePort]:
        return self._outputs

    def get_input(self, name: str) -> FakePort | None:
        return self._inputs.get(name)

    def get_output(self, name: str) -> FakePort | None:
        return self._outputs.get(name)

    def name(self) -> str:
        return self.model.custom_properties.get("name", self.id)

    def set_property(self, key: str, value: Any, push_undo: bool = False) -> None:
        self.model.custom_properties[key] = value


class FakeGraph:
    """Minimal stand-in for NodeGraphQt's NodeGraph used by handler tests."""

    def __init__(self) -> None:
        self._nodes: list[FakeNode] = []

    def add_node(self, node: FakeNode) -> FakeNode:
        self._nodes.append(node)
        return node

    def remove_node(self, node: FakeNode) -> None:
        self._nodes = [n for n in self._nodes if n.id != node.id]

    def all_nodes(self) -> list[FakeNode]:
        return list(self._nodes)

    def get_node_by_id(self, node_id: str) -> FakeNode | None:
        for n in self._nodes:
            if n.id == node_id:
                return n
        return None
