"""read_node_output tool handler — compact peek at a node's last evaluated output."""
from __future__ import annotations

import base64
import io
from typing import Callable

THUMB_SIZE = 256


def _unwrap(value):
    if value is None:
        return None
    if hasattr(value, "df"):
        return value.df
    if hasattr(value, "payload"):
        return value.payload
    return value


def _image_thumbnail_b64(arr) -> str | None:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    a = arr
    if a.ndim == 2:
        img = Image.fromarray(_to_uint8(a))
    elif a.ndim == 3 and a.shape[-1] in (3, 4):
        img = Image.fromarray(_to_uint8(a))
    else:
        return None
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _to_uint8(a):
    import numpy as np
    if a.dtype == np.uint8:
        return a
    mn, mx = float(a.min()), float(a.max())
    if mx == mn:
        return (a * 0).astype(np.uint8)
    scaled = (a - mn) / (mx - mn) * 255.0
    return scaled.astype(np.uint8)


def make_read_node_output_handler(
    graph,
    supports_vision: Callable[[], bool],
):
    """Build a handler bound to a graph + a vision-capability probe."""

    def _handler(tool_input: dict) -> dict:
        node_id = (tool_input or {}).get("node_id")
        if not node_id:
            return {"error": "read_node_output requires 'node_id'."}
        node = next((n for n in graph.all_nodes() if n.id == node_id), None)
        if node is None:
            return {"error": f"No node with id: {node_id}"}
        if not getattr(node, "output_values", None):
            return {"kind": "empty", "node_id": node_id}
        first_port = next(iter(node.outputs()), None)
        if first_port is None:
            return {"kind": "empty", "node_id": node_id}
        raw = node.output_values.get(first_port)
        unwrapped = _unwrap(raw)
        if unwrapped is None:
            return {"kind": "empty", "node_id": node_id}
        try:
            import pandas as pd
            import numpy as np
        except Exception:
            pd = None; np = None

        if pd is not None and isinstance(unwrapped, pd.DataFrame):
            df = unwrapped
            return {
                "kind": "table",
                "node_id": node_id,
                "metadata": {"shape": list(df.shape), "columns": list(df.columns.astype(str))},
                "text_preview": df.head(10).to_markdown(index=False),
            }
        if np is not None and isinstance(unwrapped, np.ndarray):
            arr = unwrapped
            meta = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(arr.min()) if arr.size else None,
                "max": float(arr.max()) if arr.size else None,
                "nan_count": int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0,
            }
            out = {"kind": "image", "node_id": node_id, "metadata": meta, "text_preview": ""}
            if supports_vision():
                thumb = _image_thumbnail_b64(arr)
                if thumb:
                    out["thumbnail"] = thumb
            return out

        return {
            "kind": "other",
            "node_id": node_id,
            "metadata": {"type": type(unwrapped).__name__},
            "text_preview": repr(unwrapped)[:500],
        }

    return _handler
