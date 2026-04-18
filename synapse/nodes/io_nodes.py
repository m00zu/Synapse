"""
nodes/io_nodes.py
=================
Input / Output nodes for reading and writing files and folders.
"""
import os
import threading

import NodeGraphQt
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor

from ..data_models import TableData, ImageData, FigureData, CollectionData
from .base import (
    BaseExecutionNode, PORT_COLORS,
    NodeFileSelector, NodeFileSaver, NodeDirSelector,
    NodeChannelSelectorWidget,
)
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np


def _normalize_to_float(arr, bit_depth=None):
    """Normalize any-depth integer array to float32 [0, 1].
    If already float, pass through. Preserves shape."""
    if arr.dtype in (np.float32, np.float64):
        # Already float — ensure float32
        if arr.max() > 1.0:
            # Likely unnormalized float (e.g. from some readers)
            max_val = float((1 << bit_depth) - 1) if bit_depth and bit_depth > 8 else 255.0
            return (arr / max_val).astype(np.float32)
        return arr.astype(np.float32)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    # uint16, uint32, etc.
    max_val = float((1 << bit_depth) - 1) if bit_depth else float(arr.max() or 1)
    return arr.astype(np.float32) / max_val


def _denormalize_from_float(arr, bit_depth=8):
    """Convert float32 [0, 1] back to integer for saving.
    Returns uint8 for 8-bit, uint16 for 12/14/16-bit."""
    if bit_depth <= 8:
        return np.clip(arr * 255, 0, 255).astype(np.uint8)
    max_val = (1 << bit_depth) - 1
    return np.clip(arr * max_val, 0, max_val).astype(np.uint16)


def _guess_bit_depth(arr):
    """Guess source bit depth from array dtype and actual data range."""
    if arr.dtype == np.uint8:
        return 8
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        max_val = float(arr.max())
        if max_val <= 1.0:
            return 8  # likely normalized 0–1
        elif max_val <= 255:
            return 8
        elif max_val <= 4095:
            return 12
        elif max_val <= 16383:
            return 14
        return 16
    # uint16 or similar integer types
    max_val = int(arr.max())
    if max_val <= 255:
        return 8
    elif max_val <= 4095:
        return 12
    elif max_val <= 16383:
        return 14
    return 16


def parse_channels(channel_str):
    """Parses a string like '1,2' or '1-3' into a list of integers."""
    channels = []
    for part in channel_str.split(','):
        if '-' in part:
            try:
                start, end = part.split('-')
                channels.extend(range(int(start), int(end) + 1))
            except (ValueError, TypeError): pass
        else:
            try:
                channels.append(int(part))
            except (ValueError, TypeError): pass
    return sorted(list(set(channels))) if channels else [2]


class FileReadNode(BaseExecutionNode):
    """
    Reads a tabular file (CSV, TSV) using pandas and outputs a DataFrame.

    **file_path** — path to the input file (widget or upstream port).
    **separator** — column delimiter (default: `,`).

    Keywords: csv, tsv, read, load, import, 讀取, 匯入, 檔案, 資料表, 載入
    """
    __identifier__ = 'nodes.io'
    NODE_NAME = 'Table Reader'
    PORT_SPEC = {'inputs': ['path'], 'outputs': ['table']}
    _collection_aware = True

    def __init__(self):
        super(FileReadNode, self).__init__()
        self.add_input('file_path', color=PORT_COLORS.get('path'))
        self.add_output('out', multi_output=True, color=PORT_COLORS.get('table'))
        
        # Add the custom file selector widget to the node surface
        # and link it to the 'file_path' property
        file_selector = NodeFileSelector(self.view, name='file_path', label='File Path')
        self.add_custom_widget(
            file_selector, 
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, 
            tab='Properties'
        )
        
        self.add_text_input('separator', 'Separator', text=',')

    def evaluate(self):
        self.reset_progress()
        import pandas as pd
        import os
        
        # Check for input port first
        in_port = self.inputs().get('file_path')
        if in_port and in_port.connected_ports():
            connected = in_port.connected_ports()[0]
            upstream_node = connected.node()
            file_path = upstream_node.output_values.get(connected.name(), None)
        else:
            file_path = self.get_property("file_path")
            
        separator = self.get_property("separator")
        
        if not file_path or not os.path.exists(file_path):
            self.mark_error()
            return False, f"File not found: {file_path}"
            
        try:
            self.set_progress(10)
            # Treat empty separator field as comma by default
            sep = separator if separator else ','
            
            # Read DataFrame
            self.set_progress(30)
            df = pd.read_csv(file_path, sep=sep)
            
            self.set_progress(80)
            self.output_values['out'] = TableData(payload=df)
            
            self.mark_clean()
            self.set_progress(100)
            return True, None
            
        except Exception as e:
            self.mark_error()
            return False, str(e)


class FolderIteratorNode(BaseExecutionNode):
    """
    Selects a folder and file pattern for batch processing.

    The actual looping is managed by the Batch Runner in `main.py`.

    **folder_path** — directory to iterate over.
    **pattern** — glob pattern for matching files (default: `*.csv`).
    **iterate_mode** — iterate over *Files* or *Subdirectories*.

    Keywords: batch, loop, directory, folder, glob, 批次, 資料夾, 疊代, 迴圈, 目錄
    """
    __identifier__ = 'nodes.io'
    NODE_NAME = 'Folder Iterator'
    PORT_SPEC = {'inputs': [], 'outputs': ['path']}
    _collection_aware = True

    def __init__(self):
        super(FolderIteratorNode, self).__init__(use_progress=False)
        self.add_output('file_path', color=PORT_COLORS['path'])
        
        folder_selector = NodeDirSelector(self.view, name='folder_path', label='Folder')
        self.add_custom_widget(
            folder_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties'
        )
        self.add_text_input('pattern', 'File Pattern', text='*.csv')
        self.add_combo_menu('iterate_mode', 'Iterate', items=['Files', 'Subdirectories'])
        self.create_property('current_file', '') # Set by the batch runner

    def evaluate(self):
        file_path = self.get_property('current_file')
        if not file_path:
            # Not in batch mode — output the first matching file as preview
            import re
            from pathlib import Path
            folder = self.get_property('folder_path') or ''
            pattern = self.get_property('pattern') or '*'
            if folder and os.path.isdir(folder):
                def _nat_key(p):
                    return [int(s) if s.isdigit() else s.lower()
                            for s in re.split(r'(\d+)', p.name)]
                matches = sorted(Path(folder).glob(pattern), key=_nat_key)
                files = [p for p in matches if p.is_file()]
                if files:
                    file_path = str(files[0])
        self.output_values['file_path'] = file_path
        self.mark_clean()
        return True, None


class _RangeSlider(QtWidgets.QWidget):
    """Dual-handle range slider for selecting start/end within a range."""

    range_changed = QtCore.Signal(int, int)  # (low, high) — 0-based

    _TRACK_H  = 4
    _HANDLE_W = 10
    _HANDLE_H = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min = 0
        self._max = 0
        self._low = 0
        self._high = 0
        self._dragging = None  # 'low' | 'high' | None
        self.setFixedHeight(22)
        self.setMinimumWidth(100)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def set_range(self, minimum: int, maximum: int):
        self._min = minimum
        self._max = max(minimum, maximum)
        self._low = minimum
        self._high = self._max
        self.update()

    def set_values(self, low: int, high: int):
        self._low = max(self._min, min(low, self._max))
        self._high = max(self._low, min(high, self._max))
        self.update()

    @property
    def low(self): return self._low

    @property
    def high(self): return self._high

    def _val_to_x(self, val: int) -> float:
        rng = self._max - self._min
        if rng <= 0:
            return self._HANDLE_W
        usable = self.width() - 2 * self._HANDLE_W
        return self._HANDLE_W + (val - self._min) / rng * usable

    def _x_to_val(self, x: float) -> int:
        usable = self.width() - 2 * self._HANDLE_W
        if usable <= 0:
            return self._min
        ratio = (x - self._HANDLE_W) / usable
        return int(round(self._min + ratio * (self._max - self._min)))

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cy = self.height() // 2

        # Track background
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QColor(60, 60, 60))
        p.drawRoundedRect(self._HANDLE_W, cy - self._TRACK_H // 2,
                          self.width() - 2 * self._HANDLE_W, self._TRACK_H,
                          2, 2)

        # Selected range highlight
        lx = self._val_to_x(self._low)
        hx = self._val_to_x(self._high)
        p.setBrush(QColor(80, 160, 255))
        p.drawRect(int(lx), cy - self._TRACK_H // 2,
                   int(hx - lx), self._TRACK_H)

        # Handles
        for val, color in [(self._low, QColor(50, 200, 100)),
                           (self._high, QColor(255, 100, 80))]:
            hx = self._val_to_x(val)
            p.setBrush(color)
            p.setPen(QColor(200, 200, 200))
            p.drawRoundedRect(int(hx - self._HANDLE_W // 2),
                              cy - self._HANDLE_H // 2,
                              self._HANDLE_W, self._HANDLE_H, 3, 3)
        p.end()

    def mousePressEvent(self, event):
        x = event.position().x()
        lx = self._val_to_x(self._low)
        hx = self._val_to_x(self._high)
        dl = abs(x - lx)
        dh = abs(x - hx)
        self._dragging = 'low' if dl <= dh else 'high'

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        val = max(self._min, min(self._x_to_val(event.position().x()), self._max))
        if self._dragging == 'low':
            self._low = min(val, self._high)
        else:
            self._high = max(val, self._low)
        self.update()
        self.range_changed.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        if self._dragging is not None:
            self._dragging = None
            self.range_changed.emit(self._low, self._high)


class _VideoWidget(NodeBaseWidget):
    """Video preview with frame browser and range selection for batch iteration."""

    def __init__(self, parent=None, name='', label=''):
        super().__init__(parent, name, label)

        self._n_frames = 0
        self._reader = None
        self._video_path: str | None = None
        self._current_pil = None
        self._current_pixmap = None
        self._current_frame_idx = -1

        container = QtWidgets.QWidget()
        container.setMinimumWidth(340)
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(3)

        # ── Preview ────────────────────────────────────────────────
        self._preview = QtWidgets.QLabel("No video loaded")
        self._preview.setMinimumHeight(60)
        self._preview.setMaximumHeight(400)
        self._preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet(
            "border:1px solid #333; background:#1a1a1a; color:#aaa; font-size:9px;")
        lay.addWidget(self._preview)

        # ── Browse slider ──────────────────────────────────────────
        browse_row = QtWidgets.QHBoxLayout()
        browse_row.setSpacing(4)
        browse_row.setContentsMargins(0, 0, 0, 0)
        lbl = QtWidgets.QLabel("Frame")
        lbl.setFixedWidth(36)
        lbl.setStyleSheet("font-size:10px;")
        browse_row.addWidget(lbl)
        self._browse_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._browse_slider.setMinimum(1)
        self._browse_slider.setMaximum(1)
        browse_row.addWidget(self._browse_slider, 1)
        self._browse_spin = QtWidgets.QSpinBox()
        self._browse_spin.setMinimum(1)
        self._browse_spin.setMaximum(1)
        self._browse_spin.setFixedWidth(60)
        self._browse_spin.setStyleSheet("font-size:10px;")
        browse_row.addWidget(self._browse_spin)
        self._total_label = QtWidgets.QLabel("/ 0")
        self._total_label.setFixedWidth(45)
        self._total_label.setStyleSheet("font-size:10px;")
        browse_row.addWidget(self._total_label)
        lay.addLayout(browse_row)

        # ── Range slider (batch start/end) ─────────────────────────
        range_row = QtWidgets.QHBoxLayout()
        range_row.setSpacing(4)
        range_row.setContentsMargins(0, 0, 0, 0)
        lbl2 = QtWidgets.QLabel("Range")
        lbl2.setFixedWidth(36)
        lbl2.setStyleSheet("font-size:10px;")
        range_row.addWidget(lbl2)
        self._range_slider = _RangeSlider()
        range_row.addWidget(self._range_slider, 1)
        self._start_spin = QtWidgets.QSpinBox()
        self._start_spin.setMinimum(1)
        self._start_spin.setMaximum(1)
        self._start_spin.setFixedWidth(60)
        self._start_spin.setStyleSheet("font-size:10px;")
        self._start_spin.setToolTip("Batch start frame")
        range_row.addWidget(self._start_spin)
        dash = QtWidgets.QLabel("–")
        dash.setFixedWidth(8)
        dash.setStyleSheet("font-size:10px;")
        range_row.addWidget(dash)
        self._end_spin = QtWidgets.QSpinBox()
        self._end_spin.setMinimum(1)
        self._end_spin.setMaximum(1)
        self._end_spin.setFixedWidth(60)
        self._end_spin.setStyleSheet("font-size:10px;")
        self._end_spin.setToolTip("Batch end frame")
        range_row.addWidget(self._end_spin)
        lay.addLayout(range_row)

        lay.addStretch()
        self.set_custom_widget(container)

        # ── Connections ────────────────────────────────────────────
        self._browse_slider.valueChanged.connect(self._on_browse_slider)
        self._browse_spin.valueChanged.connect(self._on_browse_spin)
        self._range_slider.range_changed.connect(self._on_range_slider)
        self._start_spin.valueChanged.connect(self._on_start_spin)
        self._end_spin.valueChanged.connect(self._on_end_spin)

    # ── Browse slider sync ─────────────────────────────────────────
    def _on_browse_slider(self, val):
        self._browse_spin.blockSignals(True)
        self._browse_spin.setValue(val)
        self._browse_spin.blockSignals(False)
        self._load_frame(val - 1)

    def _on_browse_spin(self, val):
        self._browse_slider.blockSignals(True)
        self._browse_slider.setValue(val)
        self._browse_slider.blockSignals(False)
        self._load_frame(val - 1)

    # ── Range slider sync ──────────────────────────────────────────
    def _on_range_slider(self, low, high):
        # Range slider uses 0-based; spinboxes use 1-based
        self._start_spin.blockSignals(True)
        self._end_spin.blockSignals(True)
        self._start_spin.setValue(low + 1)
        self._end_spin.setValue(high + 1)
        self._start_spin.blockSignals(False)
        self._end_spin.blockSignals(False)
        self._emit_change()

    def _on_start_spin(self, val):
        if val > self._end_spin.value():
            self._end_spin.blockSignals(True)
            self._end_spin.setValue(val)
            self._end_spin.blockSignals(False)
        self._range_slider.set_values(val - 1, self._end_spin.value() - 1)
        self._emit_change()

    def _on_end_spin(self, val):
        if val < self._start_spin.value():
            self._start_spin.blockSignals(True)
            self._start_spin.setValue(val)
            self._start_spin.blockSignals(False)
        self._range_slider.set_values(self._start_spin.value() - 1, val - 1)
        self._emit_change()

    def _emit_change(self):
        self.value_changed.emit(self.get_name(), self.get_value())

    # ── Video loading ──────────────────────────────────────────────
    def open_video(self, video_path: str):
        self._close_reader()
        self._video_path = None
        self._n_frames = 0
        self._current_frame_idx = -1

        if not video_path or not os.path.exists(video_path):
            self._preview.setText("No video loaded")
            self._update_ranges(0)
            return

        try:
            import imageio
            self._reader = imageio.get_reader(video_path, 'ffmpeg')
            self._n_frames = self._reader.count_frames()
            self._video_path = video_path
            self._update_ranges(self._n_frames)
            QtCore.QTimer.singleShot(0, lambda: self._load_frame(0))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "Cannot open video '%s': %s", video_path, exc)
            self._preview.setText(f"Error: {exc}")
            self._update_ranges(0)

    def _close_reader(self):
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

    def _update_ranges(self, n: int):
        mx = max(1, n)
        for w in (self._browse_slider, self._browse_spin,
                  self._start_spin, self._end_spin):
            w.blockSignals(True)
            w.setMaximum(mx)
            w.blockSignals(False)
        self._browse_slider.setValue(1)
        self._browse_spin.setValue(1)
        self._start_spin.setValue(1)
        self._end_spin.setValue(mx)
        self._total_label.setText(f"/ {n}")
        self._range_slider.set_range(0, max(0, n - 1))

    def _load_frame(self, frame_idx: int):
        if self._reader is None or frame_idx < 0 or frame_idx >= self._n_frames:
            return
        if frame_idx == self._current_frame_idx and self._current_pil is not None:
            return

        try:
            from PIL import Image as PILImage
            from PySide6.QtGui import QImage, QPixmap
            import numpy as np

            frame_arr = self._reader.get_data(frame_idx)
            if frame_arr.ndim == 2:
                pil_img = PILImage.fromarray(frame_arr).convert('RGB')
            elif frame_arr.shape[2] == 4:
                pil_img = PILImage.fromarray(frame_arr[:, :, :3])
            else:
                pil_img = PILImage.fromarray(frame_arr)

            self._current_frame_idx = frame_idx
            self._current_pil = pil_img

            # Scale for preview
            display = pil_img.copy()
            display.thumbnail((480, 360), PILImage.Resampling.LANCZOS)
            rgb = np.ascontiguousarray(display.convert('RGB'))
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self._current_pixmap = pixmap  # prevent GC
            self._preview.setPixmap(pixmap)
            self._preview.setFixedHeight(h + 4)
            self._preview.setFixedWidth(w + 4)

            # Resize node
            if self.node and hasattr(self.node, 'view'):
                self.widget().adjustSize()
                QtCore.QTimer.singleShot(0, self._do_resize)
                QtCore.QTimer.singleShot(50, self._do_resize)

            # Update output
            if self.node:
                out_arr = np.asarray(pil_img).astype(np.float32) / 255.0
                self.node.output_values['image'] = ImageData(payload=out_arr, bit_depth=8)
                self.node.output_values['file_path'] = (
                    f"{os.path.basename(self._video_path)}:frame_{frame_idx}"
                    if self._video_path else '')

        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "Failed to read frame %d: %s", frame_idx, exc)

    def _do_resize(self):
        if self.node and hasattr(self.node, 'view') and hasattr(self.node.view, 'draw_node'):
            self.widget().adjustSize()
            self.node.view.draw_node()

    # ── Batch helpers ──────────────────────────────────────────────
    @property
    def batch_start_0(self) -> int:
        """0-based start frame for batch."""
        return max(0, self._start_spin.value() - 1)

    @property
    def batch_end_0(self) -> int:
        """0-based end frame (exclusive) for batch."""
        return min(self._n_frames, self._end_spin.value())

    # ── Serialisation ──────────────────────────────────────────────
    def get_value(self) -> str:
        import json
        return json.dumps({
            'start': self._start_spin.value(),
            'end': self._end_spin.value(),
            'browse': self._browse_spin.value(),
        })

    def set_value(self, value):
        import json
        if not value:
            return
        try:
            d = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            return
        if 'start' in d:
            self._start_spin.blockSignals(True)
            self._start_spin.setValue(int(d['start']))
            self._start_spin.blockSignals(False)
        if 'end' in d:
            self._end_spin.blockSignals(True)
            self._end_spin.setValue(int(d['end']))
            self._end_spin.blockSignals(False)
        if 'browse' in d:
            self._browse_slider.blockSignals(True)
            self._browse_spin.blockSignals(True)
            self._browse_slider.setValue(int(d['browse']))
            self._browse_spin.setValue(int(d['browse']))
            self._browse_slider.blockSignals(False)
            self._browse_spin.blockSignals(False)
        # Sync range slider
        self._range_slider.set_values(
            self._start_spin.value() - 1, self._end_spin.value() - 1)


class VideoIteratorNode(BaseExecutionNode):
    """
    Browses and iterates over frames of a video file.

    Preview any frame with the browse slider. Select a start/end range
    with the dual-handle range slider, then use Batch Run to process
    each frame through the downstream graph.

    **video_path** — path to the video file.

    Keywords: video, frames, batch, loop, mp4, avi, timelapse, preview, 影片, 視頻, 幀, 批次, 預覽
    """

    __identifier__ = 'nodes.io'
    NODE_NAME      = 'Video Iterator'
    PORT_SPEC      = {'inputs': [], 'outputs': ['image', 'path']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress',
        'video_widget_state',
    })

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_output('image', color=PORT_COLORS['image'])
        self.add_output('file_path', color=PORT_COLORS['path'])

        file_selector = NodeFileSelector(
            self.view, name='video_path', label='Video')
        self.add_custom_widget(
            file_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties',
        )

        self._video_widget = _VideoWidget(self.view, name='video_widget_state', label='')
        self.add_custom_widget(self._video_widget)

        self.create_property('current_file', '')

        self._reader = None
        self._video_path_cached: str | None = None

    def set_property(self, name, value, push_undo=True):
        super().set_property(name, value, push_undo)
        if name == 'video_path':
            self._video_widget.open_video(value)

    @staticmethod
    def _get_reader(video_path: str):
        import imageio
        return imageio.get_reader(video_path, 'ffmpeg')

    def get_batch_items(self) -> list[str]:
        video_path = self.get_property('video_path')
        if not video_path or not os.path.exists(video_path):
            return []

        w = self._video_widget
        start = w.batch_start_0
        end = w.batch_end_0

        if end <= start:
            return []
        return [str(i) for i in range(start, end)]

    def on_batch_start(self):
        video_path = self.get_property('video_path')
        try:
            self._reader = self._get_reader(video_path)
            self._video_path_cached = video_path
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "Failed to open video: %s", exc)
            self._reader = None

    def on_batch_end(self):
        w = self._video_widget
        self.model.set_property('current_file', str(w.batch_start_0))

        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        self._video_path_cached = None

    def evaluate(self):
        frame_idx_str = self.get_property('current_file')
        video_path = self.get_property('video_path')

        if not video_path or not os.path.exists(video_path):
            return False, f"Video not found: {video_path}"

        # Single run (not batch): use the browse slider's current frame
        if not frame_idx_str:
            w = self._video_widget
            if w._current_pil is not None:
                arr = np.asarray(w._current_pil).astype(np.float32) / 255.0
                self.output_values['image'] = ImageData(payload=arr, bit_depth=8)
                self.output_values['file_path'] = (
                    f"{os.path.basename(video_path)}:frame_{w._current_frame_idx}")
                self.mark_clean()
                return True, None
            frame_idx = max(0, w._browse_spin.value() - 1)
        else:
            try:
                frame_idx = int(frame_idx_str)
            except ValueError:
                return False, f"Invalid frame index: {frame_idx_str}"

        try:
            if self._reader is not None and self._video_path_cached == video_path:
                frame_arr = self._reader.get_data(frame_idx)
            else:
                reader = self._get_reader(video_path)
                frame_arr = reader.get_data(frame_idx)
                reader.close()

            if frame_arr.ndim == 2:
                frame_arr = np.stack([frame_arr]*3, axis=-1)
            elif frame_arr.shape[2] == 4:
                frame_arr = frame_arr[:, :, :3]
            out_arr = frame_arr.astype(np.float32) / 255.0

            self.output_values['image'] = ImageData(payload=out_arr, bit_depth=8)
            self.output_values['file_path'] = (
                f"{os.path.basename(video_path)}:frame_{frame_idx}")
            self.mark_clean()
            return True, None

        except Exception as exc:
            return False, f"Failed to read frame {frame_idx}: {exc}"


class BatchAccumulatorNode(BaseExecutionNode):
    """
    Collects the output of each batch iteration and merges them after the batch finishes.

    Connect upstream data to the `in` port; the `out` port emits the merged
    result only after the entire batch is complete.

    Batch context stamping:
    - Automatically adds `frame` and `file` metadata to each collected value.
    - For `TableData`, this lets downstream nodes identify which frame each row came from.

    Keywords: batch, collect, merge, accumulate, combine, 批次, 合併, 累積, 收集, 組合
    """
    __identifier__ = 'nodes.io'
    NODE_NAME = 'Batch Accumulator'
    PORT_SPEC = {'inputs': ['any'], 'outputs': ['any']}
    _is_accumulator = True

    def __init__(self):
        super(BatchAccumulatorNode, self).__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['any'])
        self.add_output('out', multi_output=True, color=PORT_COLORS['any'])
        self._collected = []
        self._iteration = 0
        self._batch_complete = False   # True after a successful batch run

    def mark_dirty(self):
        """Override to clear the batch-complete flag when upstream changes."""
        self._batch_complete = False
        super().mark_dirty()

    def on_batch_start(self):
        """Called by BatchGraphWorker before the first iteration."""
        self._collected = []
        self._iteration = 0

    def _find_batch_file(self) -> str:
        """Walk upstream to find the current batch file from an iterator node."""
        visited = set()
        queue = [self]
        while queue:
            node = queue.pop(0)
            nid = id(node)
            if nid in visited:
                continue
            visited.add(nid)
            cf = None
            try:
                cf = node.get_property('current_file')
            except Exception:
                pass
            if cf:
                # VideoIteratorNode: current_file is a frame index, not a path.
                # Build a meaningful label from the video filename + frame index.
                vp = None
                try:
                    vp = node.get_property('video_path')
                except Exception:
                    pass
                if vp and cf.isdigit():
                    stem = os.path.splitext(os.path.basename(vp))[0]
                    return f"{stem}_frame_{int(cf)+1}"
                return os.path.basename(str(cf))
            # Walk further upstream
            for port in node.inputs().values():
                for cp in port.connected_ports():
                    queue.append(cp.node())
        return ''

    def evaluate(self):
        """Collect one value from the upstream node per iteration.
        If a previous batch already completed and nothing upstream changed,
        return the cached merged output without re-collecting."""
        if self._batch_complete and 'out' in self.output_values:
            self.mark_clean()
            return True, None

        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input connected"

        connected = in_port.connected_ports()[0]
        upstream_node = connected.node()
        value = upstream_node.output_values.get(connected.name(), None)

        if value is not None:
            self._iteration += 1
            # Stamp batch context into metadata
            batch_file = self._find_batch_file()
            if hasattr(value, 'metadata'):
                value.metadata['frame'] = self._iteration
                value.metadata['file'] = batch_file
                # For future, just in case?
                value.metadata['batch_key'] = batch_file or str(self._iteration)
            if hasattr(value, 'source_path') and not value.source_path:
                value.source_path = batch_file
            self._collected.append(value)

        # During batch run, output is not set yet — deferred to on_batch_end
        self.mark_clean()
        return True, None

    def on_batch_end(self):
        """Called by BatchGraphWorker after all iterations. Merges and stores output."""
        if not self._collected:
            self.output_values['out'] = None
            return

        first = self._collected[0]
        try:
            merged = type(first).merge(self._collected)
        except (NotImplementedError, AttributeError):
            # Fallback: just return a plain list
            merged = self._collected

        self.output_values['out'] = merged
        self._batch_complete = True
        self.mark_clean()


# ── Pure-Python OIR reader (fallback when oir_reader_rs is not compiled) ───
def _py_get_meta_block(path):
    """Locate the XML metadata inside an OIR binary."""
    import mmap as _mmap
    with open(path, 'rb') as f, _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
        start = mm.find(b"<fileinfo")
        if start == -1:
            start = 0
        end = mm.find(b"<annotation", start)
        if end == -1:
            end = min(start + 40000, len(mm))
        return mm[start:end].decode("utf-8", errors="ignore")


def _py_extract_xml(xml, field, as_float=False, as_int=False):
    open_tag = f"<{field}>"
    close_tag = f"</{field}>"
    a = xml.find(open_tag)
    if a == -1:
        return None
    b = xml.find(close_tag, a + len(open_tag))
    if b == -1:
        return None
    inner = xml[a + len(open_tag):b].strip()
    if as_float:
        try:
            return float(inner)
        except ValueError:
            return None
    if as_int:
        try:
            return int(inner)
        except ValueError:
            return None
    return inner


def _py_read_oir_meta(path):
    """Parse OIR metadata: dimensions, channels, bit depth, line rate."""
    import re
    meta = _py_get_meta_block(path)
    p_res = meta.find('<lsmimage:scannerSettings type="Resonant">')
    p_gal = meta.find('<lsmimage:scannerSettings type="Galvano">')
    if p_res != -1 and p_gal != -1:
        scanner = _py_extract_xml(meta, 'lsmimage:scannerType')
        meta_scan = meta[p_res:] if scanner == 'Resonant' else meta[p_gal:p_res]
    elif p_res != -1:
        meta_scan = meta[p_res:]
    elif p_gal != -1:
        meta_scan = meta[p_gal:]
    else:
        meta_scan = meta

    bit_depth = _py_extract_xml(meta, 'commonphase:bitCounts', as_int=True) or 12
    size_x = _py_extract_xml(meta_scan, 'commonparam:width', as_int=True)
    size_y = _py_extract_xml(meta_scan, 'commonparam:height', as_int=True)
    line_rate = _py_extract_xml(meta_scan, 'commonparam:lineSpeed', as_float=True)

    # Count unique channel orders
    ch_orders = set(re.findall(r'<commonphase:channel\s+id="[^"]+"\s+order="(\d+)"', meta))
    n_channels = len(ch_orders) if ch_orders else 4

    if not all([size_x, size_y, line_rate]):
        raise ValueError("Could not parse OIR metadata (missing width/height/lineSpeed)")
    return size_x, size_y, n_channels, line_rate, bit_depth


def _py_read_oir_frames(path, size_x, size_y, n_ch, line_rate, bit_depth):
    """Decode tiled pixel data from an OIR file (pure Python + numpy)."""
    import numpy as np
    with open(path, 'rb') as f:
        raw = f.read()
    buf = np.frombuffer(raw, dtype=np.uint8)

    lines_per_tile = int(np.ceil(30.0 / line_rate))
    n_div = int(np.ceil(size_y / lines_per_tile))

    # Find tile markers
    pos_4 = np.flatnonzero(buf == 4).astype(np.int64)

    def literal_positions(lit):
        L = len(lit)
        if L == 0 or pos_4.size == 0:
            return np.array([], dtype=np.int64)
        p = pos_4.copy()
        mask = np.ones_like(p, dtype=bool)
        for a, byte in enumerate(lit):
            idx = p - (L + 4) + a
            ok = (idx >= 0) & (idx < buf.size) & (buf[idx] == byte)
            mask &= ok
        return p[mask]

    starts_per_div = []
    for i_div in range(1, n_div + 1):
        s = literal_positions(f"_{i_div - 1}".encode()) + 3
        s = s[s < buf.size]
        if s.size > 0:
            first = int(s[0])
            if b'REF' in raw[max(0, first - 99):first + 1]:
                s = s[n_ch:] if s.size >= n_ch else np.array([], dtype=np.int64)
        starts_per_div.append(s)

    # Keep raw values as uint16 — no normalization to 8-bit
    image = np.zeros((size_y, size_x, n_ch), dtype=np.uint16)

    for ch in range(n_ch):
        rows_written = 0
        for j in range(n_div):
            lines_this = min(lines_per_tile, size_y - j * lines_per_tile)
            starts = starts_per_div[j]
            if ch >= len(starts):
                rows_written += lines_this
                continue
            p = int(starts[ch])
            b0 = p + 1
            b1 = min(b0 + 2 * size_x * lines_this, len(raw))
            block = raw[b0:b1]
            if len(block) % 2 == 1:
                block = block[:-1]
            temp = np.frombuffer(block, dtype='<u2')
            need = size_x * lines_this
            if temp.size < need:
                temp = np.pad(temp, (0, need - temp.size))
            elif temp.size > need:
                temp = temp[:need]
            tile = temp.reshape((lines_this, size_x))
            image[rows_written:rows_written + lines_this, :, ch] = tile
            rows_written += lines_this

    return image


def _py_read_single_oir(path):
    """Read an OIR file, returning (H, W, C) uint16 array, n_channels, scale, and bit_depth."""
    size_x, size_y, n_ch, line_rate, bit_depth = _py_read_oir_meta(path)
    img = _py_read_oir_frames(path, size_x, size_y, n_ch, line_rate, bit_depth)
    scale_um = _extract_oir_scale(path)
    return img, n_ch, scale_um, bit_depth


def _extract_oir_scale(path):
    """Extract µm/pixel from OIR metadata (commonphase:length)."""
    import re
    try:
        meta = _py_get_meta_block(path)
        m = re.search(r'<commonphase:length>.*?<commonparam:x>([\d.]+)</commonparam:x>', meta, re.DOTALL)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def _extract_tiff_scale(path):
    """Extract µm/pixel from TIFF metadata.

    Checks in order: ImageJ description, OME-XML, standard resolution tags.
    """
    import re
    try:
        from PIL import Image
        img = Image.open(path)

        # 1. ImageJ description (tag 270)
        if hasattr(img, 'tag_v2'):
            desc = img.tag_v2.get(270, '')
            if isinstance(desc, str) and 'ImageJ' in desc:
                # Look for "spacing=X" (Z spacing) or "unit=um" style
                unit_m = re.search(r'unit=(\S+)', desc)
                unit = unit_m.group(1) if unit_m else ''
                # ImageJ stores resolution in XResolution tag when unit is set
                xres = img.tag_v2.get(282)
                res_unit = img.tag_v2.get(296, 1)
                if xres and unit.lower() in ('um', 'µm', 'micron'):
                    if isinstance(xres, tuple):
                        xres = xres[0] / xres[1] if len(xres) == 2 else xres[0]
                    # ImageJ: XResolution = pixels per unit
                    return 1.0 / float(xres) if xres > 0 else None

            # 2. OME-XML
            desc = img.tag_v2.get(270, '')
            if isinstance(desc, str) and 'PhysicalSizeX' in desc:
                m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', desc)
                if m:
                    return float(m.group(1))

            # 3. Standard TIFF resolution tags
            xres = img.tag_v2.get(282)
            res_unit = img.tag_v2.get(296, 1)  # 1=none, 2=inch, 3=cm
            if xres and res_unit in (2, 3):
                if isinstance(xres, tuple):
                    xres = xres[0] / xres[1] if len(xres) == 2 else xres[0]
                xres = float(xres)
                if xres > 0:
                    if res_unit == 3:  # cm
                        return 10000.0 / xres  # cm to µm
                    elif res_unit == 2:  # inch
                        return 25400.0 / xres  # inch to µm
    except Exception:
        pass
    return None


def _save_image_with_scale(pil_img, file_path, scale_um):
    """Save a PIL Image with scale metadata embedded if possible."""
    import struct
    ext = file_path.lower().rsplit('.', 1)[-1] if '.' in file_path else ''

    if ext in ('tif', 'tiff') and scale_um and scale_um > 0:
        # Write as TIFF with resolution tags (pixels per cm)
        px_per_cm = 10000.0 / scale_um
        pil_img.save(file_path, tiffinfo={
            282: px_per_cm,   # XResolution
            283: px_per_cm,   # YResolution
            296: 3,           # ResolutionUnit = centimeters
        })
    elif ext == 'png' and scale_um and scale_um > 0:
        # PNG pHYs chunk: pixels per meter
        px_per_m = 1_000_000.0 / scale_um
        ppi = int(px_per_m / 39.3701)  # approximate DPI for compatibility
        pil_img.save(file_path, dpi=(ppi, ppi))
    else:
        pil_img.save(file_path)


class ImageReadNode(BaseExecutionNode):
    """
    Reads an image file and outputs it as a float32 [0,1] numpy array.

    Supported formats:

    - *Standard* — JPEG, PNG, BMP, and other PIL-supported formats (8-bit)
    - *TIFF* — 8/12/14/16-bit microscopy TIFFs (bit depth preserved). Multi-page TIFFs output a CollectionData with one ImageData per page.
    - *OIR* — Olympus .oir files (Rust accelerated, with Python fallback)

    The original bit depth is stored as metadata for downstream nodes
    (threshold sliders, histogram, save). All processing uses float32 [0,1]
    internally.

    Options:

    - **channels** — comma-separated channel numbers (0-4, where 0 = black/pad).
      `2` for single grayscale channel,
      `1,2,3` for RGB,
      `2,3,4` to map channels 2/3/4 as R/G/B,
      `1,0,3` to map ch1 as red, black as green, ch3 as blue.

    Keywords: open, load, import, image, channel, oir, tiff, 讀取, 匯入, 影像, 開啟, 載入
    """
    __identifier__ = 'nodes.io'
    NODE_NAME = 'Image Reader'
    PORT_SPEC = {'inputs': ['path'], 'outputs': ['image']}
    _collection_aware = True

    def __init__(self):
        super(ImageReadNode, self).__init__()
        self.add_input('file_path', color=PORT_COLORS.get('path'))
        self.add_output('out', multi_output=True, color=PORT_COLORS.get('image'))

        # Add embedded file selector
        file_selector = NodeFileSelector(self.view, name='file_path', label='Image Path')
        self.add_custom_widget(
            file_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties'
        )
        # Channel selector: toggle channels 1-4, max 3.
        # e.g. "1,2,3" for RGB, "2" for single grayscale, "2,3,4" for custom RGB mapping.
        ch_widget = NodeChannelSelectorWidget(self.view, name='channels', label='Channels', text='1,2,3')
        self.add_custom_widget(
            ch_widget,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties'
        )

        self.output_values = {}

    def _parse_channels(self, ch_str):
        """Parse channel string like '2' or '1,0,3' into list of ints.
        Returns None for invalid input. Max 3 channels, values 0-4 (0=pad/black)."""
        parts = [p.strip() for p in str(ch_str).split(',') if p.strip()]
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            return None
        if not vals or len(vals) > 3:
            return None
        if any(v < 0 or v > 4 for v in vals):
            return None
        return vals

    def evaluate(self):
        self.reset_progress()
        from PIL import Image
        import numpy as np
        import os

        # Check for input port first
        in_port = self.inputs().get('file_path')
        if in_port and in_port.connected_ports():
            connected = in_port.connected_ports()[0]
            upstream_node = connected.node()
            file_path = upstream_node.output_values.get(connected.name(), None)
        else:
            file_path = self.get_property("file_path")

        if not file_path or not os.path.exists(file_path):
            self.mark_error()
            return False, f"File not found: {file_path}"

        ch_str = str(self.get_property('channels') or '').strip()
        if not ch_str:
            ch_str = '1,2,3'  # default: RGB
        channels = self._parse_channels(ch_str)
        if channels is None:
            self.mark_error()
            return False, (f"Invalid channel setting '{ch_str}'. "
                           "Use 1-4, e.g. '2' for grayscale, '1,3' for RG, or '2,3,4' for RGB mapping. "
                           "Max 3 channels.")

        try:
            self.set_progress(10)
            scale_um = None
            bit_depth = 8

            if file_path.lower().endswith('.oir'):
                # Try Rust extension first, fall back to pure Python
                try:
                    import oir_reader_rs
                    _name, img, _group, _isize = oir_reader_rs.read_oir_file(
                        file_path, list(range(1, 5)))
                    if img is None:
                        return False, f"Failed to read OIR: {file_path}"
                    scale_um = _extract_oir_scale(file_path)
                    # Both readers now output raw uint16
                    bit_depth = _guess_bit_depth(img)
                except ImportError:
                    img, _n_ch, scale_um, bit_depth = _py_read_single_oir(file_path)
                self.set_progress(50)
                out_arr = self._select_channels(img, channels)

            elif file_path.lower().endswith(('.tif', '.tiff')):
                n_pages = 1
                tiff_bps = None
                try:
                    import tifffile
                    with tifffile.TiffFile(file_path) as tif:
                        n_pages = len(tif.pages)
                        tiff_bps = tif.pages[0].bitspersample
                        arr = tif.asarray()
                except Exception:
                    arr = np.asarray(Image.open(file_path))
                scale_um = _extract_tiff_scale(file_path)
                self.set_progress(50)

                if n_pages > 1:
                    # Multi-page TIFF → CollectionData
                    bd = tiff_bps or _guess_bit_depth(arr[0])
                    items = {}
                    for i in range(n_pages):
                        page = arr[i]
                        if page.ndim == 2:
                            page_out = page
                        else:
                            page_out = self._select_channels(page, channels)
                        page_out = _normalize_to_float(page_out, bd)
                        items[f'page_{i}'] = ImageData(
                            payload=page_out, bit_depth=bd, scale_um=scale_um)
                    self.output_values['out'] = CollectionData(payload=items)
                    self.mark_clean()
                    self.set_progress(100)
                    return True, None

                bit_depth = tiff_bps or _guess_bit_depth(arr)
                if arr.ndim == 2:
                    out_arr = arr
                else:
                    out_arr = self._select_channels(arr, channels)
            else:
                pil_img = Image.open(file_path)
                arr = np.asarray(pil_img.convert('RGB'))
                bit_depth = 8
                self.set_progress(50)
                out_arr = self._select_channels(arr, channels)

            self.set_progress(80)
            # Normalize to float32 [0, 1] for consistent pipeline
            out_arr = _normalize_to_float(out_arr, bit_depth)
            self.output_values['out'] = ImageData(
                payload=out_arr, bit_depth=bit_depth, scale_um=scale_um)
            self.mark_clean()
            self.set_progress(100)
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)

    @staticmethod
    def _select_channels(img_arr, channels):
        """Select and reorder channels from a (H, W, C) array.

        channels: list of channel numbers (1-4, or 0 for black/pad).
        - 1 channel  -> grayscale numpy array (H, W)
        - 2-3 channels -> RGB numpy array (H, W, 3)
        Returns numpy array, preserving original dtype.
        """
        if img_arr.ndim == 2:
            return img_arr

        n_ch = img_arr.shape[2]

        if len(channels) == 1:
            ch = channels[0]
            if ch == 0:
                return np.zeros(img_arr.shape[:2], dtype=img_arr.dtype)
            ch_idx = ch - 1
            if ch_idx < n_ch:
                return img_arr[..., ch_idx]
            return np.zeros(img_arr.shape[:2], dtype=img_arr.dtype)

        # 2 or 3 channels -> (H, W, 3), preserving dtype
        rgb = np.zeros((*img_arr.shape[:2], 3), dtype=img_arr.dtype)
        for i, ch in enumerate(channels):
            if ch == 0:
                continue
            ch_idx = ch - 1
            if ch_idx < n_ch:
                rgb[..., i] = img_arr[..., ch_idx]
        return rgb

def write_pzfx(tables: dict, file_path: str):
    """
    Writes a dictionary of pandas DataFrames (or a single DataFrame) to a GraphPad Prism .pzfx file.
    Assumes a simple 'OneWay' (Column format) table where each DataFrame column becomes a Prism YColumn.
    """
    import pandas as pd
    root = ET.Element("GraphPadPrismFile", PrismXMLVersion="5.00")

    # minimal properties to be a valid file
    created = ET.SubElement(root, "Created")
    ET.SubElement(created, "OriginalVersion", CreatedByProgram="GraphPad Prism", CreatedByVersion="8.0.2.263", Login="user", DateTime="2024-01-01T00:00:00+00:00")
    
    table_seq = ET.SubElement(root, "TableSequence")
    
    for i, (table_name, df) in enumerate(tables.items()):
        ref = ET.SubElement(table_seq, "Ref", ID=f"Table{i}")
        if i == 0:
            ref.set("Selected", "1")
            
        table = ET.SubElement(root, "Table", ID=f"Table{i}", XFormat="none", TableType="OneWay", EVFormat="AsteriskAfterNumber")
        title_el = ET.SubElement(table, "Title")
        title_el.text = str(table_name)
        
        for col_name in df.columns:
            y_col = ET.SubElement(table, "YColumn", Width="80", Decimals="6", Subcolumns="1")
            col_title = ET.SubElement(y_col, "Title")
            
            # Optionally clean the name if it came from read_pzfx (e.g. NF_0 -> NF)
            clean_name = str(col_name)
            if "_" in clean_name and clean_name.split("_")[-1].isdigit():
                clean_name = "_".join(clean_name.split("_")[:-1])
                
            col_title.text = clean_name
            
            subcol = ET.SubElement(y_col, "Subcolumn")
            for val in df[col_name]:
                d_el = ET.SubElement(subcol, "d")
                if pd.isna(val):
                    pass # Empty element
                else:
                    d_el.text = str(val)
    
    # Format the XML string across multiple lines and add the header
    xml_string = ET.tostring(root, encoding="UTF-8")
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="UTF-8")

    with open(file_path, 'wb') as f:
        f.write(pretty_xml)

class SaveNode(BaseExecutionNode):
    """
    Saves incoming data to a file on disk.

    Supported output types:
    - *DataFrame* — saved as CSV, TSV, or `.pzfx` (GraphPad Prism)
    - *Figure* — saved as an image at the figure's native DPI
    - *Image* — saved via PIL in any format matching the file extension

    **file_path** — destination path (widget or upstream port).

    Keywords: save, write, export, csv, tsv, 儲存, 寫入, 匯出, 檔案, 輸出
    """
    __identifier__ = 'nodes.utility'
    NODE_NAME = 'Data Saver'
    PORT_SPEC = {'inputs': ['any', 'path'], 'outputs': []}
    _collection_aware = True

    def __init__(self):
        super(SaveNode, self).__init__()
        self.add_input('in', color=PORT_COLORS.get('any'))
        self.add_input('file_path_in', color=PORT_COLORS.get('path'))

        # Add embedded file selector
        file_selector = NodeFileSaver(self.view, name='file_path', label='Save Path')
        self.add_custom_widget(
            file_selector, 
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, 
            tab='Properties'
        )
    
    def evaluate(self):
        self.reset_progress()
        from PIL import Image
        import os
        
        # Check for path override from input port
        in_path_port = self.inputs().get('file_path_in')
        if in_path_port and in_path_port.connected_ports():
            connected = in_path_port.connected_ports()[0]
            upstream_node = connected.node()
            file_path = upstream_node.output_values.get(connected.name(), None)
        else:
            file_path = self.get_property("file_path")

        in_values = []
        in_raw = []  # keep original NodeData for metadata
        in_port = self.inputs().get('in')
        if in_port and in_port.connected_ports():
            for connected in in_port.connected_ports():
                upstream_node = connected.node()
                up_val = upstream_node.output_values.get(connected.name(), None)
                in_raw.append(up_val)
                if isinstance(up_val, TableData):
                    in_values.append(up_val.df)
                elif hasattr(up_val, 'payload'):
                    in_values.append(up_val.payload)
                else:
                    in_values.append(up_val)

        if not in_values or in_values[0] is None:
            self.mark_error()
            return False, "No input data"

        data = in_values[0]
        raw_data = in_raw[0] if in_raw else None

        try:
            self.set_progress(10)
            if not file_path:
                self.reset_progress()
                self.mark_error()
                return False, "File path not specified"
                
            import pandas as pd
            import matplotlib.figure
            
            if isinstance(data, pd.DataFrame):
                df = data

                self.set_progress(50)
                if file_path.lower().endswith('.pzfx'):
                    write_pzfx({"Data 1": df}, file_path)
                else:
                    sep = '\t' if file_path.lower().endswith('.tsv') else ','
                    df.to_csv(file_path, sep=sep, index=False)
                
            elif isinstance(data, matplotlib.figure.Figure):
                self.set_progress(50)
                data.tight_layout()
                data.savefig(file_path, bbox_inches='tight', dpi=float(data.get_dpi()))
                
            elif isinstance(data, np.ndarray):
                # Numpy array from the new pipeline
                self.set_progress(50)
                bit_depth = getattr(raw_data, 'bit_depth', 8) if raw_data else 8
                scale_um = getattr(raw_data, 'scale_um', None) if raw_data else None
                ext = file_path.lower().rsplit('.', 1)[-1] if '.' in file_path else ''

                if ext in ('tif', 'tiff'):
                    import tifffile
                    out_arr = _denormalize_from_float(data, bit_depth)
                    if scale_um and scale_um > 0:
                        px_per_cm = 10000.0 / scale_um
                        metadata = {'resolution': (px_per_cm, px_per_cm), 'resolutionunit': 3}
                        tifffile.imwrite(file_path, out_arr, **metadata)
                    else:
                        tifffile.imwrite(file_path, out_arr)
                else:
                    # PNG/JPEG/BMP — always 8-bit
                    out_arr = _denormalize_from_float(data, 8)
                    from PIL import Image as _PILImage
                    pil = _PILImage.fromarray(out_arr)
                    if scale_um and scale_um > 0:
                        ppi = int(1_000_000.0 / scale_um / 39.3701)
                        pil.save(file_path, dpi=(ppi, ppi))
                    else:
                        pil.save(file_path)

            elif isinstance(data, Image.Image):
                # Legacy PIL Image (backward compat)
                self.set_progress(50)
                scale_um = getattr(raw_data, 'scale_um', None) if raw_data else None
                _save_image_with_scale(data, file_path, scale_um)

            self.mark_clean()
            self.set_progress(100)
            return True, None

        except Exception as e:
            self.mark_error()
            return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# BatchGateNode  — pauses the batch loop after each iteration for user review
# ─────────────────────────────────────────────────────────────────────────────

class _BatchGateWidget(NodeBaseWidget):
    """Widget showing gate status + Next / Pass All / Refresh buttons."""

    next_clicked     = Signal()
    pass_all_clicked = Signal()
    refresh_clicked  = Signal()

    _state_signal = Signal(bool, str)   # (waiting, packed_info) → main thread

    def __init__(self, parent=None):
        super().__init__(parent, 'batch_gate')

        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)
        container = QtWidgets.QWidget()
        container.setLayout(root)

        self._status_lbl = QtWidgets.QLabel('Idle — not in a batch run')
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet('color:#aaa; font-size:10px; padding:2px;')
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setFixedHeight(8)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self._next_btn = QtWidgets.QPushButton('▶  Next')
        self._next_btn.setFixedHeight(28)
        self._next_btn.setEnabled(False)
        self._refresh_btn = QtWidgets.QPushButton('↻  Refresh')
        self._refresh_btn.setFixedHeight(28)
        self._refresh_btn.setEnabled(False)
        self._pass_all_btn = QtWidgets.QPushButton('Pass All')
        self._pass_all_btn.setFixedHeight(28)
        self._pass_all_btn.setEnabled(False)
        btn_row.addWidget(self._next_btn)
        btn_row.addWidget(self._refresh_btn)
        btn_row.addWidget(self._pass_all_btn)
        root.addLayout(btn_row)

        self.set_custom_widget(container)

        self._next_btn.clicked.connect(self.next_clicked)
        self._refresh_btn.clicked.connect(self.refresh_clicked)
        self._pass_all_btn.clicked.connect(self.pass_all_clicked)
        self._state_signal.connect(self._apply_state, Qt.ConnectionType.QueuedConnection)

    def get_value(self):         return ''
    def set_value(self, _value): pass

    def show_waiting(self, i: int, total: int, file_name: str):
        packed = f'{i}|{total}|{file_name}'
        if threading.current_thread() is threading.main_thread():
            self._apply_state(True, packed)
        else:
            self._state_signal.emit(True, packed)

    def show_idle(self):
        if threading.current_thread() is threading.main_thread():
            self._apply_state(False, '')
        else:
            self._state_signal.emit(False, '')

    def _apply_state(self, waiting: bool, info: str):
        if waiting:
            parts = info.split('|', 2)
            i, total = int(parts[0]), int(parts[1])
            name = parts[2] if len(parts) > 2 else ''
            self._status_lbl.setText(f'⏸  Paused  —  item {i} / {total}\n{name}')
            self._status_lbl.setStyleSheet(
                'color:#ffdd55; font-size:10px; font-weight:bold; padding:2px;')
            self._progress.setRange(0, total)
            self._progress.setValue(i)
            self._next_btn.setEnabled(True)
            self._refresh_btn.setEnabled(True)
            self._pass_all_btn.setEnabled(True)
            self._next_btn.setStyleSheet(
                'background:#2a7a44; color:white; font-weight:bold; border-radius:4px;')
            self._refresh_btn.setStyleSheet(
                'background:#3a5a8a; color:white; font-weight:bold; border-radius:4px;')
        else:
            self._status_lbl.setText('Idle — not in a batch run')
            self._status_lbl.setStyleSheet('color:#aaa; font-size:10px; padding:2px;')
            self._progress.setRange(0, 1)
            self._progress.setValue(0)
            self._next_btn.setEnabled(False)
            self._refresh_btn.setEnabled(False)
            self._pass_all_btn.setEnabled(False)
            self._next_btn.setStyleSheet('')
            self._refresh_btn.setStyleSheet('')


class BatchGateNode(BaseExecutionNode):
    """
    Pass-through gate that pauses the batch pipeline for user review.

    Wire between any two nodes using the single `any`-typed input/output.
    Blocking happens inside `evaluate()`, so multiple gates pause
    independently at their own step in the topological evaluation order.

    Controls:
    - *Next* — let this iteration continue past the gate
    - *Refresh* — re-evaluate upstream nodes and update previews
    - *Pass All* — stop pausing for the rest of this batch run

    Keywords: batch, pause, gate, review, step, 批次, 暫停, 閘門, 審查, 逐步
    """

    __identifier__ = 'nodes.io'
    NODE_NAME = 'Batch Gate'
    PORT_SPEC = {'inputs': ['any'], 'outputs': ['any']}

    def __init__(self):
        super().__init__()
        self.add_input('in',  color=PORT_COLORS['any'])
        self.add_output('out', color=PORT_COLORS['any'], multi_output=True)

        self._event    = threading.Event()
        self._refresh  = threading.Event()
        self._pass_all = False
        self._stopped  = False
        self._in_batch = False
        self._batch_item  = 0
        self._batch_total = 0
        self._batch_file  = ''

        self._widget = _BatchGateWidget(self.view)
        self._widget.next_clicked.connect(self._on_next)
        self._widget.refresh_clicked.connect(self._on_refresh)
        self._widget.pass_all_clicked.connect(self._on_pass_all)
        self.add_custom_widget(self._widget)

    def _on_next(self):
        self._event.set()

    def _on_refresh(self):
        self._refresh.set()

    def _on_pass_all(self):
        self._pass_all = True
        self._event.set()

    # ── batch lifecycle ──────────────────────────────────────────────────────

    def on_batch_start(self):
        self._in_batch    = True
        self._pass_all    = False
        self._stopped     = False
        self._batch_item  = 0
        self._batch_total = 0
        self._event.clear()
        self._widget.show_idle()

    def on_batch_end(self):
        self._in_batch = False
        self._stopped  = True
        self._event.set()   # unblock if currently waiting

    def set_batch_item(self, i: int, total: int, file_path: str):
        """Called by BatchGraphWorker before each iteration so the widget can
        show item progress when the gate activates."""
        self._batch_item  = i
        self._batch_total = total
        self._batch_file  = os.path.basename(str(file_path))

    # ── upstream refresh ────────────────────────────────────────────────────

    def _read_input(self):
        """Copy upstream output_values into our own output."""
        in_port = self.inputs().get('in')
        if in_port and in_port.connected_ports():
            cp = in_port.connected_ports()[0]
            self.output_values['out'] = cp.node().output_values.get(cp.name())

    def _collect_upstream_sorted(self):
        """Return upstream nodes in dependency order (leaves first)."""
        visited = set()
        order = []

        def _walk(node):
            nid = id(node)
            if nid in visited:
                return
            visited.add(nid)
            for in_p in node.inputs().values():
                for conn_p in in_p.connected_ports():
                    _walk(conn_p.node())
            order.append(node)

        for in_p in self.inputs().values():
            for conn_p in in_p.connected_ports():
                _walk(conn_p.node())
        return order

    def _refresh_upstream(self):
        """Re-evaluate all upstream nodes so their previews update,
        then re-read the input value.

        Skip other BatchGateNode instances — they've already passed through
        earlier in the topological evaluation order, so re-entering their
        evaluate() would just pause them again, blocking this refresh until
        the user clicks Next on the upstream gate too.
        """
        upstream = self._collect_upstream_sorted()
        for node in upstream:
            if isinstance(node, BatchGateNode):
                continue
            if hasattr(node, 'evaluate'):
                try:
                    node.evaluate()
                    if hasattr(node, 'mark_clean'):
                        node.mark_clean()
                except Exception:
                    pass
        self._read_input()

    # ── evaluate ─────────────────────────────────────────────────────────────

    def evaluate(self):
        # Pass input straight through
        self._read_input()

        # Block here until user clicks Next (only during batch, not manual runs)
        if self._in_batch and not self._pass_all and not self._stopped:
            self._widget.show_waiting(
                self._batch_item, self._batch_total, self._batch_file)
            self._event.clear()
            self._refresh.clear()
            while not self._event.wait(timeout=0.1):
                if self._stopped:
                    break
                if self._refresh.is_set():
                    self._refresh.clear()
                    self._refresh_upstream()
            self._widget.show_idle()

        return True, None
