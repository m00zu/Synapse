"""Serialize node outputs into the session preview dir.

Called by the Executor after each node's evaluate() returns success. Inspects
``output_values`` and writes one file per compatible output:

  - ``ImageData``   → 256 px PNG via PIL (aspect-preserving)
  - ``TableData``   → JSON ``{columns: [...], rows: [...]}`` of head(50)
  - ``FigureData``  → matplotlib ``savefig(png, dpi=72, bbox_inches='tight')``

Writes are best-effort — a serialization failure must NOT fail the node run.
Each write returns a ``{"port": str, "kind": "image"|"table"|"figure"}`` so the
executor can publish a ``preview_available`` WS event per preview.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_MAX_TABLE_ROWS = 50
_MAX_IMAGE_EDGE_PX = 256


def write_previews(node_id: str, output_values: dict, preview_dir: Path) -> list[dict]:
    """Write previews for every compatible output value. Returns the list
    of ``{port, kind}`` records for the ones successfully written."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    for port, value in (output_values or {}).items():
        kind = _detect_kind(value)
        if kind is None:
            continue
        out_path = preview_dir / f"{node_id}__{port}.{'json' if kind == 'table' else 'png'}"
        try:
            if kind == "image":
                _write_image(value, out_path)
            elif kind == "table":
                _write_table(value, out_path)
            elif kind == "figure":
                _write_figure(value, out_path)
        except Exception as exc:
            logger.warning("preview: %s/%s %s — %s",
                           node_id, port, kind, exc)
            continue
        written.append({"port": port, "kind": kind})
    return written


def _detect_kind(value: Any) -> str | None:
    """Duck-typed kind detection. Imports data_models lazily so this module
    can be imported in environments where data_models raises at import time."""
    try:
        from synapse.data_models import ImageData, TableData, FigureData
    except Exception:
        return None
    if isinstance(value, ImageData):
        return "image"
    if isinstance(value, TableData):
        return "table"
    if isinstance(value, FigureData):
        return "figure"
    return None


def _write_image(value: Any, out: Path) -> None:
    """Write an ImageData payload as a 256 px PNG (aspect-preserving)."""
    import numpy as np
    from PIL import Image
    arr = value.payload
    if arr is None:
        raise ValueError("ImageData.payload is None")
    a = np.asarray(arr)
    # Normalize [0,1] floats to uint8 for Pillow.
    if a.dtype.kind == "f":
        a = (np.clip(a, 0.0, 1.0) * 255.0).astype("uint8")
    # Collapse multi-channel w/ only one to grayscale; trim RGBA alpha.
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    img = Image.fromarray(a)
    img.thumbnail((_MAX_IMAGE_EDGE_PX, _MAX_IMAGE_EDGE_PX))
    img.save(out, format="PNG", optimize=True)


def _write_table(value: Any, out: Path) -> None:
    """Write a TableData payload as ``head(50)`` JSON."""
    df = value.payload
    if df is None:
        raise ValueError("TableData.payload is None")
    head = df.head(_MAX_TABLE_ROWS)
    payload = {
        "columns": [str(c) for c in head.columns],
        "rows": head.astype(object).where(head.notna(), None).values.tolist(),
        "total_rows": int(len(df)),
    }
    out.write_text(json.dumps(payload), encoding="utf-8")


def _write_figure(value: Any, out: Path) -> None:
    """Save a matplotlib FigureData as PNG."""
    fig = value.payload
    if fig is None:
        raise ValueError("FigureData.payload is None")
    fig.savefig(out, format="png", dpi=72, bbox_inches="tight")
