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


# ── Hard caps (not adjustable via tool params) ───────────────────────────────
_RANGE_CAP = 500
_COLUMNS_CAP = 500
_FILTER_CAP = 200
_FULL_LIMIT = 5000


def _describe_mode(df: Any, port_name: str) -> dict[str, Any]:
    try:
        import math
        summary = df.describe(include='all').to_dict()
        cleaned: dict = {}
        for col, stats in summary.items():
            cleaned[str(col)] = {
                str(k): (None if (isinstance(v, float) and math.isnan(v)) else
                          v.item() if hasattr(v, 'item') else v)
                for k, v in stats.items()
            }
    except Exception as e:
        raise ValueError(f"describe failed: {e}")
    return {'port': port_name, 'kind': 'describe',
            'n_rows': int(df.shape[0]), 'n_cols': int(df.shape[1]),
            'summary': cleaned}


def _range_mode(df: Any, port_name: str,
                start: int, end: int | None) -> dict[str, Any]:
    n = len(df)
    s = max(0, int(start))
    e = n if end is None else max(s, int(end))
    e = min(e, n, s + _RANGE_CAP)
    rows = df.iloc[s:e].to_dict(orient='records')
    return {'port': port_name, 'kind': 'range',
            'start': s, 'end': e, 'n_total': n,
            'n_returned': len(rows), 'rows': rows}


def _columns_mode(df: Any, port_name: str,
                  columns: list[str] | None,
                  sample: int) -> dict[str, Any]:
    if not columns:
        raise ValueError("mode='columns' requires the 'columns' parameter.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Unknown column(s): {missing}. Available: {list(df.columns)}")
    take = min(int(sample), _COLUMNS_CAP, len(df))
    rows = df[columns].head(take).to_dict(orient='records')
    return {'port': port_name, 'kind': 'columns',
            'columns': list(columns), 'n_rows': int(len(df)),
            'n_returned': len(rows), 'head': rows}


def _filter_mode(df: Any, port_name: str, query: str | None) -> dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("mode='filter' requires the 'query' parameter "
                          "(pandas df.query syntax, e.g. \"area > 1000\").")
    try:
        filtered = df.query(query)
    except Exception as e:
        raise ValueError(
            f"Invalid filter query {query!r}: {type(e).__name__}: {e}")
    n_matched = int(len(filtered))
    take = min(n_matched, _FILTER_CAP)
    rows = filtered.head(take).to_dict(orient='records')
    return {'port': port_name, 'kind': 'filter',
            'query': query, 'n_matched': n_matched,
            'n_returned': len(rows), 'rows': rows}


def _full_mode(df: Any, port_name: str) -> dict[str, Any]:
    n = int(len(df))
    if n > _FULL_LIMIT:
        raise ValueError(
            f"Table has {n} rows (> {_FULL_LIMIT}); refusing 'full' to "
            f"avoid blowing up chat context. Use mode='filter' with a "
            f"pandas-query predicate or mode='range' with start/end.")
    return {'port': port_name, 'kind': 'full',
            'n_rows': n, 'rows': df.to_dict(orient='records')}


def get_node_output(controller: GraphController,
                    node_id: str,
                    port_name: str | None = None,
                    mode: str = 'preview',
                    # mode='range' args:
                    start: int = 0,
                    end: int | None = None,
                    # mode='columns' args:
                    columns: list[str] | None = None,
                    sample: int = 20,
                    # mode='filter' args:
                    query: str | None = None,
                    ) -> dict[str, Any]:
    """Read a value emitted on a node's output port.

    Modes (DataFrame outputs only, except ``'preview'`` which handles any):

      - ``'preview'``  (default): shape + dtypes + first 10 rows / image
        stats / model summary / scalar value.
      - ``'describe'``: ``df.describe(include='all')`` -- per-column stats.
      - ``'range'``:    rows ``[start:end]`` (capped at 500).
      - ``'columns'``:  project to ``columns`` then sample (capped at 500).
      - ``'filter'``:   rows matching pandas ``df.query(query)`` (capped 200).
      - ``'full'``:     entire dataframe (errors if > 5000 rows; use filter).

    Use ``mode='preview'`` for non-table outputs (images, figures, models,
    scalars).
    """
    # ── 1. Resolve node + port ──────────────────────────────────────────
    try:
        rec = controller.get_node(node_id)
    except KeyError:
        raise ValueError(
            f"Unknown node id: {node_id!r}. "
            f"Call describe_graph() to see current node ids.")

    if port_name is None:
        try:
            info = controller.describe_registered(rec.type_id)
            outs = info.output_ports
        except Exception:
            outs = []
        if len(outs) == 1:
            port_name = outs[0]
        elif len(outs) == 0:
            raise ValueError(f"Node {node_id!r} has no output ports.")
        else:
            raise ValueError(
                f"Node {node_id!r} has {len(outs)} output ports "
                f"({outs}); pass port_name to pick one.")

    try:
        value = controller.get_node_output(node_id, port_name)
    except KeyError as e:
        raise ValueError(
            f"{e.args[0]}. Call run_node() first, then retry.")

    # ── 2. Preview short-circuit ────────────────────────────────────────
    if mode == 'preview':
        return _summarize(value, port_name)

    # ── 2b. Validate mode name before inspecting payload ────────────────
    _VALID_MODES = {'preview', 'describe', 'range', 'columns', 'filter', 'full'}
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. "
            f"Valid: 'preview', 'describe', 'range', 'columns', 'filter', 'full'.")

    # ── 3. Other modes require a DataFrame payload ──────────────────────
    payload = getattr(value, 'payload', value)
    try:
        import pandas as pd
    except ImportError:
        raise ValueError("pandas is required for non-preview modes.")
    if not isinstance(payload, pd.DataFrame):
        return {'port': port_name,
                'kind': 'unsupported_mode', 'mode': mode,
                'payload_type': type(payload).__name__,
                'hint': "mode='preview' works for non-table outputs"}

    # ── 4. Dispatch by mode ─────────────────────────────────────────────
    if mode == 'describe':
        return _describe_mode(payload, port_name)
    if mode == 'range':
        return _range_mode(payload, port_name, start, end)
    if mode == 'columns':
        return _columns_mode(payload, port_name, columns, sample)
    if mode == 'filter':
        return _filter_mode(payload, port_name, query)
    if mode == 'full':
        return _full_mode(payload, port_name)

    raise ValueError(
        f"Unknown mode {mode!r}. "
        f"Valid: 'preview', 'describe', 'range', 'columns', 'filter', 'full'.")


def _summarize(value: Any, port_name: str) -> dict[str, Any]:
    """Convert any node output into a JSON-safe preview dict."""
    # Unwrap typed-data containers (TableData, ImageData, etc.) -- their
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
            # Basic stats (best effort -- strings/objects don't cooperate)
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
                'hint': 'To VIEW this image, call '
                        'get_node_image(node_id, port_name) instead.',
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
                'hint': 'To VIEW this figure rendered as a PNG, call '
                        'get_node_image(node_id, port_name) instead.  '
                        'This preview gives only metadata.',
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
