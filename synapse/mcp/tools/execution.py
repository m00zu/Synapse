"""Execution tools: run_node, get_node_status, get_node_output."""
from __future__ import annotations

from typing import Any

from ..controller import GraphController


def run_node(controller: GraphController, node_id: str) -> dict[str, Any]:
    """Evaluate a node, re-evaluating any dirty upstream first.

    Mirrors what the user gets when they click "Run" on the node in the
    Synapse UI.  Returns ``{success, message, duration_ms}``.  A node
    that succeeded but produced a status message (warnings, counts, etc.)
    still has ``success=True``; ``message`` carries the text.
    """
    try:
        controller.get_node(node_id)
    except KeyError:
        raise ValueError(
            f"Unknown node id: {node_id!r}. "
            f"Call describe_graph() to see current node ids.")
    return controller.run_node(node_id)


def get_node_status(controller: GraphController,
                    node_id: str) -> dict[str, Any]:
    """Return the last known status of a node without re-running it.

    Status values: ``'pending'`` (never evaluated since last dirty),
    ``'running'``, ``'clean'`` (last evaluate succeeded), ``'error'``
    (last evaluate raised or returned False).
    """
    try:
        rec = controller.get_node(node_id)
    except KeyError:
        raise ValueError(
            f"Unknown node id: {node_id!r}. "
            f"Call describe_graph() to see current node ids.")
    return {'node_id': rec.id, 'status': rec.status,
            'last_message': rec.last_message}


def get_node_output(controller: GraphController,
                    node_id: str,
                    port_name: str | None = None) -> dict[str, Any]:
    """Read the value emitted on a node's output port (preview).

    Returns a JSON-friendly summary of the data, scaled appropriately
    for chat context.  For tabular data this is shape + dtypes + head(10).
    For images: shape, dtype, basic stats.  For figures: dimensions.
    For models: type name + repr.  For scalars: the value directly.

    If ``port_name`` is None and the node has exactly one output port,
    that port is used automatically; otherwise pass the port name from
    ``describe_graph`` or ``describe_node``.

    Errors with a clear message if the node hasn't been evaluated yet.
    """
    try:
        rec = controller.get_node(node_id)
    except KeyError:
        raise ValueError(
            f"Unknown node id: {node_id!r}. "
            f"Call describe_graph() to see current node ids.")

    # Resolve port if not specified.
    if port_name is None:
        try:
            info = controller.describe_registered(rec.type_id)
            outs = info.output_ports
        except Exception:
            outs = []
        if len(outs) == 1:
            port_name = outs[0]
        elif len(outs) == 0:
            raise ValueError(
                f"Node {node_id!r} has no output ports.")
        else:
            raise ValueError(
                f"Node {node_id!r} has {len(outs)} output ports "
                f"({outs}); pass port_name to pick one.")

    try:
        value = controller.get_node_output(node_id, port_name)
    except KeyError as e:
        raise ValueError(
            f"{e.args[0]}. Call run_node() first, then retry.")

    return _summarize(value, port_name)


def _summarize(value: Any, port_name: str) -> dict[str, Any]:
    """Convert any node output into a JSON-safe preview dict."""
    # Unwrap typed-data containers (TableData, ImageData, etc.) — their
    # actual content lives on .payload by Synapse convention.
    payload = getattr(value, 'payload', value)

    out: dict[str, Any] = {'port': port_name,
                           'wrapper': type(value).__name__
                                       if value is not payload else None}

    # pandas DataFrame
    try:
        import pandas as pd
        if isinstance(payload, pd.DataFrame):
            df = payload
            out.update({
                'kind': 'table',
                'n_rows': int(df.shape[0]),
                'n_cols': int(df.shape[1]),
                'columns': list(df.columns.astype(str)),
                'dtypes': {str(c): str(df[c].dtype) for c in df.columns},
                'head': df.head(10).to_dict(orient='records'),
            })
            return out
    except Exception:
        pass

    # numpy ndarray
    try:
        import numpy as np
        if isinstance(payload, np.ndarray):
            arr = payload
            out.update({
                'kind': 'image' if arr.ndim in (2, 3) else 'array',
                'shape': list(arr.shape),
                'dtype': str(arr.dtype),
            })
            # Basic stats (best effort — strings/objects don't cooperate)
            try:
                out['min'] = float(arr.min())
                out['max'] = float(arr.max())
                out['mean'] = float(arr.mean())
            except Exception:
                pass
            return out
    except Exception:
        pass

    # PIL Image
    try:
        from PIL import Image as _PILImage
        if isinstance(payload, _PILImage.Image):
            out.update({
                'kind': 'image', 'size': list(payload.size),
                'mode': payload.mode,
            })
            return out
    except Exception:
        pass

    # matplotlib Figure
    try:
        import matplotlib  # noqa: F401
        from matplotlib.figure import Figure
        if isinstance(payload, Figure):
            w, h = payload.get_size_inches()
            out.update({
                'kind': 'figure',
                'width_inches': float(w), 'height_inches': float(h),
                'n_axes': len(payload.axes),
            })
            return out
    except Exception:
        pass

    # sklearn estimator (or anything with fit/predict)
    if (hasattr(payload, 'fit') and
            (hasattr(payload, 'predict') or hasattr(payload, 'transform'))):
        out.update({
            'kind': 'model',
            'type_name': type(payload).__name__,
            'str': repr(payload)[:500],
        })
        return out

    # Plain JSON-serializable types
    if isinstance(payload, (type(None), bool, int, float, str)):
        out.update({'kind': 'scalar', 'value': payload})
        return out

    if isinstance(payload, (list, tuple)):
        truncated = list(payload)[:50]
        out.update({'kind': 'json', 'value': truncated,
                    'total_length': len(payload),
                    'truncated': len(payload) > 50})
        return out

    if isinstance(payload, dict):
        truncated_keys = list(payload.keys())[:50]
        out.update({'kind': 'json',
                    'value': {k: payload[k] for k in truncated_keys},
                    'total_keys': len(payload),
                    'truncated': len(payload) > 50})
        return out

    # Fallback: opaque
    out.update({
        'kind': 'opaque',
        'type_name': type(payload).__name__,
        'str': repr(payload)[:500],
    })
    return out
