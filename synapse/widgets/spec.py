"""Widget-spec dataclasses — Qt-free, JSON-serializable.

A WidgetSpec describes a single node's UI shape once. Two renderers consume it:
  - synapse/widgets/pyside_renderer.py (Phase 1b) builds a QWidget tree
  - web/src/components/widgets/Renderer.tsx builds React components

Keep this module Qt-free. spec-only tests must run in CI environments
that don't have PySide6 installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class WidgetSpec:
    """Base class. Subclasses set ``kind`` via a class attribute.

    ``tab`` is an optional grouping hint: multiple sibling specs that share
    the same ``tab`` value should render inside one tab panel. Empty string
    means "main card" (no tab).
    """
    kind: str = field(init=False)
    tab: str = field(default="", kw_only=True)


# ---------------------------------------------------------------------------
# Leaf widgets — one property, one input surface
# ---------------------------------------------------------------------------

@dataclass
class ComboBox(WidgetSpec):
    prop: str
    label: str
    options: list[str]
    default: Optional[str] = None
    kind: str = field(default="ComboBox", init=False)


@dataclass
class NumberField(WidgetSpec):
    prop: str
    label: str
    min: float = 0.0
    max: float = 1e9
    step: float = 1.0
    decimals: int = 0           # 0 = integer spinbox behavior
    default: float = 0.0
    kind: str = field(default="NumberField", init=False)


@dataclass
class CheckBox(WidgetSpec):
    prop: str
    label: str
    default: bool = False
    kind: str = field(default="CheckBox", init=False)


@dataclass
class TextField(WidgetSpec):
    prop: str
    label: str
    default: str = ""
    placeholder: str = ""
    kind: str = field(default="TextField", init=False)


# TODO(web-phase-1c): add __post_init__ validation once spec_from_json
# is fed from untrusted HTTP input. Literal[...] is advisory at runtime.
@dataclass
class FilePath(WidgetSpec):
    prop: str
    label: str
    mode: Literal["server-browse", "upload", "either"] = "either"
    file_filter: str = "*"      # glob for server-browse / accept for upload
    default: str = ""
    kind: str = field(default="FilePath", init=False)


@dataclass
class Button(WidgetSpec):
    """A button that triggers a named action on the node (no property bound)."""
    action: str                 # node-method name, e.g. "reset_roi"
    label: str
    kind: str = field(default="Button", init=False)


@dataclass
class Progress(WidgetSpec):
    """A progress bar bound to a 0..1 property set by evaluate()."""
    prop: str                   # property the node updates during evaluate()
    label: str = ""
    kind: str = field(default="Progress", init=False)


# TODO(web-phase-1c): add __post_init__ validation once spec_from_json
# is fed from untrusted HTTP input. Literal[...] is advisory at runtime.
@dataclass
class Preview(WidgetSpec):
    """Display area for an output port's preview (image/table/figure)."""
    preview_kind: Literal["image", "table", "figure"]
    source: str                 # e.g. "output:image" or "output:table"
    kind: str = field(default="Preview", init=False)


@dataclass
class Custom(WidgetSpec):
    """Escape hatch — component_id names a hand-written widget on each side."""
    component_id: str
    props: dict[str, Any] = field(default_factory=dict)
    kind: str = field(default="Custom", init=False)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class VerticalLayout(WidgetSpec):
    children: list["WidgetSpec"]
    kind: str = field(default="VerticalLayout", init=False)


@dataclass
class HorizontalLayout(WidgetSpec):
    children: list["WidgetSpec"]
    kind: str = field(default="HorizontalLayout", init=False)


# ---------------------------------------------------------------------------
# (De)serialization
# ---------------------------------------------------------------------------

_KIND_MAP: dict[str, type[WidgetSpec]] = {
    cls.__name__: cls for cls in [
        ComboBox, NumberField, CheckBox, TextField, FilePath,
        Button, Progress, Preview, Custom,
        VerticalLayout, HorizontalLayout,
    ]
}


def spec_to_json(spec: WidgetSpec) -> dict:
    """Convert a WidgetSpec tree into a plain dict suitable for json.dumps.

    asdict recursively walks nested dataclasses; `kind` comes along
    because it's a regular dataclass field (just with init=False).
    """
    return asdict(spec)


def spec_from_json(blob: dict) -> WidgetSpec:
    """Inverse of spec_to_json. Raises ValueError on unknown ``kind``."""
    kind = blob.get("kind")
    cls = _KIND_MAP.get(kind)
    if cls is None:
        raise ValueError(f"unknown widget kind: {kind!r}")
    payload = {k: v for k, v in blob.items() if k != "kind"}
    if "children" in payload:
        payload["children"] = [spec_from_json(c) for c in payload["children"]]
    return cls(**payload)
