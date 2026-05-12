"""Visual-content tool: get_node_image.

Returns a node's output port as a PNG image embedded in the MCP
response so the LLM can "see" the result via its vision model.
Supports PIL.Image, numpy ndarray (HxW or HxWxC), and matplotlib
Figure payloads.  Larger images are downsampled to keep response
size bounded.
"""
from __future__ import annotations

import base64
import io
from typing import Any

from ..controller import GraphController


# Hard cap on the long side to keep responses small enough for chat
# clients to render comfortably (Claude's vision input is ~5 MB max).
_MAX_DIMENSION = 1600


def get_node_image(controller: GraphController,
                   node_id: str,
                   port_name: str | None = None,
                   max_dim: int = 1024) -> list:
    """**Use this whenever the user wants to SEE a figure or image** —
    plots, masks, thresholded images, anything visual.  The output is
    embedded as an MCP ImageContent so your vision model receives the
    actual pixels, not metadata.

    Prefer this over ``get_node_output`` for any image/figure/mask
    output.  ``get_node_output(mode='preview')`` on a figure only
    returns width/height/n_axes — useless for actually seeing it.

    Supported payload types:
      - PIL.Image (returned as-is)
      - numpy.ndarray (2-D grayscale or 3-D HxWxC, dtype anything,
        bool masks auto-convert to 0/255)
      - matplotlib.figure.Figure (rasterised via savefig at adaptive DPI)
      - FigureData.svg_override (edited SVG from the SVG Editor node,
        rasterised via Qt's QSvgRenderer so user edits are preserved)

    ``max_dim`` caps the longest side (default 1024 px; hard ceiling
    1600).  Larger images are downsampled with Lanczos resampling.

    Returns a list of MCP content blocks (TextContent + ImageContent).

    Errors with a clear message for non-image outputs — fall back to
    ``get_node_output(mode='preview')`` for tables, models, scalars.
    """
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

    cap = min(int(max_dim), _MAX_DIMENSION) if max_dim else _MAX_DIMENSION

    # SVG Editor produces FigureData with `svg_override` set to the
    # edited SVG bytes — diverges from the underlying matplotlib
    # figure once the user has touched anything.  Render the SVG
    # directly via Qt so edits actually reach the LLM.
    svg_bytes = getattr(value, 'svg_override', None)
    if svg_bytes:
        png_bytes = _svg_to_png(svg_bytes, max_dim=cap)
    else:
        payload = getattr(value, 'payload', value)
        png_bytes = _render_to_png(payload, max_dim=cap)

    b64 = base64.b64encode(png_bytes).decode('ascii')
    # Return shape: FastMCP turns the dict into structured content, but
    # we also surface a content block of ImageContent so Claude's
    # vision model sees the image directly.  The exact return type
    # below is documented inline.
    from mcp.types import ImageContent, TextContent
    blocks = [
        TextContent(type='text',
                     text=f"Image from node {node_id} port {port_name!r} "
                          f"({len(png_bytes)} bytes PNG)."),
        ImageContent(type='image', data=b64, mimeType='image/png'),
    ]
    # FastMCP forwards a returned list[Content] verbatim.
    return blocks


# ── Rendering helpers ────────────────────────────────────────────────

def _render_to_png(payload: Any, max_dim: int) -> bytes:
    """Convert any supported payload type to PNG bytes."""
    # 1. PIL.Image
    try:
        from PIL import Image as _PILImage
        if isinstance(payload, _PILImage.Image):
            return _pil_to_png(payload, max_dim)
    except ImportError:
        pass

    # 2. numpy ndarray
    try:
        import numpy as np
        if isinstance(payload, np.ndarray):
            return _ndarray_to_png(payload, max_dim)
    except ImportError:
        pass

    # 3. matplotlib Figure
    try:
        from matplotlib.figure import Figure
        if isinstance(payload, Figure):
            return _figure_to_png(payload, max_dim)
    except ImportError:
        pass

    raise ValueError(
        f"Cannot render to image: payload type "
        f"{type(payload).__name__!r} is not a PIL.Image, ndarray, or "
        f"matplotlib.figure.Figure.  Use get_node_output(mode='preview') "
        f"for non-image outputs.")


def _downsample_pil(img, max_dim: int):
    """Lanczos downsample to fit longest side within max_dim."""
    from PIL import Image as _PILImage
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, _PILImage.Resampling.LANCZOS)


def _pil_to_png(img, max_dim: int) -> bytes:
    img = _downsample_pil(img, max_dim)
    # PNG doesn't support some modes natively; convert to RGB / RGBA.
    if img.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
        img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def _ndarray_to_png(arr, max_dim: int) -> bytes:
    """Encode a numpy image array as PNG."""
    import numpy as np
    from PIL import Image as _PILImage

    if arr.ndim not in (2, 3):
        raise ValueError(
            f"ndarray with shape {arr.shape} is not an image "
            f"(need 2-D grayscale or 3-D HxWxC).")

    # Normalize dtype for PIL: PIL accepts uint8 / uint16 / float32.
    # Bool masks → 0/255 uint8.  Float → scale to 0-255 (assuming 0-1
    # range; clip outliers).  Already-uint8 → as-is.
    a = arr
    if a.dtype == np.bool_:
        a = (a.astype(np.uint8)) * 255
    elif a.dtype.kind == 'f':
        # Assume 0-1 range; if max > 1, assume already 0-255 and clip.
        if float(a.max()) > 1.0001:
            a = np.clip(a, 0, 255).astype(np.uint8)
        else:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)

    img = _PILImage.fromarray(a)
    return _pil_to_png(img, max_dim)


def _figure_to_png(fig, max_dim: int) -> bytes:
    """Rasterise a matplotlib Figure to PNG."""
    # Cap DPI so the rendered file stays within max_dim on the longest side.
    w_in, h_in = fig.get_size_inches()
    longest_in = max(w_in, h_in)
    dpi = min(150, max(50, int(max_dim / longest_in)))
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()


def _svg_to_png(svg_bytes: bytes, max_dim: int) -> bytes:
    """Rasterise SVG bytes to PNG via Qt's built-in QSvgRenderer.

    Used for ``FigureData.svg_override`` payloads (the output of the
    interactive SVG editor node).  Zero new dependencies — PySide6 is
    already in the Synapse stack.
    """
    from PySide6 import QtCore, QtGui, QtSvg

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(bytes(svg_bytes)))
    if not renderer.isValid():
        raise ValueError("svg_override payload is not valid SVG.")

    # Native (vector-defined) size in pixels.  Downsample to max_dim
    # while preserving aspect ratio.
    native = renderer.defaultSize()
    w, h = max(1, native.width()), max(1, native.height())
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / longest
        w = max(1, int(w * scale))
        h = max(1, int(h * scale))

    img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(img)
    try:
        renderer.render(painter)
    finally:
        painter.end()

    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, 'PNG')
    return bytes(buf.data())
