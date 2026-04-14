"""read_node_output tool handler — compact peek at one or many nodes' last outputs."""
from __future__ import annotations

import base64
import io
import json
from typing import Callable

from synapse.ai.context import estimate_tokens

THUMB_SIZE = 256
MAX_BATCH = 8
THUMBNAIL_BATCH_LIMIT = 3  # include thumbnails only if image count in batch is strictly below this
DEFAULT_TOKEN_CAP = 2500   # per-call payload cap when multiple nodes are requested


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
        from PIL import Image
    except Exception:
        return None
    a = arr
    if a.size == 0:
        return None
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


def _classify(unwrapped) -> str:
    """Return kind string without building the full dict yet (for pre-counting images)."""
    if unwrapped is None:
        return "empty"
    try:
        import pandas as pd
        import numpy as np
    except Exception:
        pd = None; np = None
    if pd is not None and isinstance(unwrapped, pd.DataFrame):
        return "table"
    if np is not None and isinstance(unwrapped, np.ndarray):
        return "image"
    return "other"


def _read_one(node, include_thumbnail: bool) -> dict:
    """Build a result dict for a single node. ``include_thumbnail`` is AND-gated
    with vision-capability by the caller."""
    node_id = node.id
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
            "metadata": {"shape": list(df.shape),
                         "columns": list(df.columns.astype(str))},
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
        if include_thumbnail:
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


def make_read_node_output_handler(
    graph,
    supports_vision: Callable[[], bool],
    token_cap: int = DEFAULT_TOKEN_CAP,
):
    """Build a handler bound to a graph + a vision-capability probe.

    Accepts either ``node_id: str`` (single-node result dict, unchanged shape)
    or ``node_ids: [str]`` (batch; returns ``{"results": [...], "truncated": bool}``).
    """

    def _lookup(node_id: str):
        for n in graph.all_nodes():
            if getattr(n, "id", None) == node_id:
                return n
            if getattr(n, "_llm_id", None) == node_id:
                return n
        return None

    def _handler(tool_input: dict) -> dict:
        inp = tool_input or {}
        single_id = inp.get("node_id")
        batch_ids = inp.get("node_ids")

        # Validate: exactly one of the two must be present.
        if single_id and batch_ids:
            return {"error": "read_node_output: pass only one of 'node_id' or 'node_ids'."}
        if not single_id and not batch_ids:
            return {"error": "read_node_output: requires 'node_id' or 'node_ids'."}

        # Single-node path — preserves Phase 2a shape for existing callers.
        if single_id:
            node = _lookup(single_id)
            if node is None:
                return {"error": f"No node with id: {single_id}"}
            return _read_one(node, include_thumbnail=supports_vision())

        # Batch path.
        if not isinstance(batch_ids, list):
            return {"error": "'node_ids' must be an array of strings."}
        if len(batch_ids) == 0:
            return {"error": "'node_ids' must be non-empty."}
        if len(batch_ids) > MAX_BATCH:
            return {"error": f"'node_ids' exceeds batch limit ({MAX_BATCH})."}

        # Resolve nodes first so we can pre-count images for the thumbnail guard.
        resolved = []
        for nid in batch_ids:
            node = _lookup(nid)
            if node is None:
                resolved.append((nid, None))
            else:
                resolved.append((nid, node))

        # Pre-classify to decide thumbnail inclusion for the whole batch.
        image_count = 0
        for _, node in resolved:
            if node is None:
                continue
            first_port = next(iter(node.outputs()), None)
            if first_port is None:
                continue
            raw = node.output_values.get(first_port) if getattr(node, "output_values", None) else None
            unwrapped = _unwrap(raw)
            if _classify(unwrapped) == "image":
                image_count += 1
        include_thumb = supports_vision() and image_count < THUMBNAIL_BATCH_LIMIT

        results: list[dict] = []
        truncated = False
        for nid, node in resolved:
            if node is None:
                results.append({"error": f"No node with id: {nid}", "node_id": nid})
            else:
                results.append(_read_one(node, include_thumbnail=include_thumb))
            # Budget check — trim trailing thumbnails, then trailing text_preview,
            # then mark truncated and stop adding to the list.
            approx = estimate_tokens(json.dumps({"results": results}))
            if approx > token_cap:
                # Trim most recent: drop thumbnail, then preview, then pop entirely.
                last = results[-1]
                if "thumbnail" in last:
                    last.pop("thumbnail", None)
                    truncated = True
                    approx = estimate_tokens(json.dumps({"results": results}))
                if approx > token_cap and "text_preview" in last and last["text_preview"]:
                    last["text_preview"] = ""
                    truncated = True
                    approx = estimate_tokens(json.dumps({"results": results}))
                if approx > token_cap and len(results) > 1:
                    # Keep at least one result so the model has something to work
                    # with, even if it overshoots — never return an empty list.
                    results.pop()
                    truncated = True
                    break

        return {
            "results": results,
            "truncated": truncated,
            "included_thumbnails": include_thumb,
        }

    return _handler
