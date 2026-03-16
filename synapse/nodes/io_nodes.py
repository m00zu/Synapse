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

from ..data_models import TableData, ImageData, FigureData
from .base import (
    BaseExecutionNode, PORT_COLORS,
    NodeFileSelector, NodeFileSaver, NodeDirSelector,
)
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


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
                self.node.output_values['image'] = ImageData(payload=pil_img)
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
        from PIL import Image as PILImage

        frame_idx_str = self.get_property('current_file')
        video_path = self.get_property('video_path')

        if not video_path or not os.path.exists(video_path):
            return False, f"Video not found: {video_path}"

        # Single run (not batch): use the browse slider's current frame
        if not frame_idx_str:
            w = self._video_widget
            if w._current_pil is not None:
                self.output_values['image'] = ImageData(payload=w._current_pil)
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
                pil_img = PILImage.fromarray(frame_arr).convert('RGB')
            elif frame_arr.shape[2] == 4:
                pil_img = PILImage.fromarray(frame_arr[:, :, :3])
            else:
                pil_img = PILImage.fromarray(frame_arr)

            self.output_values['image'] = ImageData(payload=pil_img)
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


class ImageReadNode(BaseExecutionNode):
    """
    Reads an image file and outputs it as a PIL Image.

    Supported formats:
    - *Standard* — JPEG, PNG, BMP, and other PIL-supported formats
    - *TIFF* — 12/16-bit microscopy TIFFs (auto-normalized to 8-bit)
    - *OIR* — Olympus .oir files (via `oir_reader_rs` extension)

    **file_path** — path to the image file (widget or upstream port).

    Keywords: open, load, import, image, 讀取, 匯入, 影像, 開啟, 載入
    """
    __identifier__ = 'nodes.io'
    NODE_NAME = 'Image Reader'
    PORT_SPEC = {'inputs': ['path'], 'outputs': ['image']}

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
        # self.add_text_input('channel', 'OIR Channel', text='2')
        
        self.output_values = {}
    
    def evaluate(self):
        self.reset_progress()
        from PIL import Image
        import os
        import sys
        from pathlib import Path
        
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

        try:
            self.set_progress(10)
            if file_path.lower().endswith('.oir'):
                # OIR handling via Rust extension
                import oir_reader_rs
                channels = [1, 2, 3]
                name, img, group, isize = oir_reader_rs.read_oir_file(file_path, channels)
                if img is None:
                    return False, f"Failed to read OIR: {file_path}"
                pil_img = Image.fromarray(img)
            elif file_path.lower().endswith(('.tif', '.tiff')):
                import numpy as np
                # Try PIL first (works in Nuitka builds), fall back to tifffile
                try:
                    pil_img = Image.open(file_path)
                    arr = np.asarray(pil_img)
                except Exception:
                    import tifffile
                    arr = tifffile.imread(file_path)
                # Normalize to uint8 for PIL (handle 12/16-bit microscopy TIFFs)
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.float32)
                    lo, hi = arr.min(), arr.max()
                    if hi > lo:
                        arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
                    else:
                        arr = np.zeros_like(arr, dtype=np.uint8)
                pil_img = Image.fromarray(arr).convert('RGB')
            else:
                pil_img = Image.open(file_path).convert('RGB')

            self.set_progress(80)
            self.output_values['out'] = ImageData(payload=pil_img)
            self.mark_clean()
            self.set_progress(100)
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)

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
        in_port = self.inputs().get('in')
        if in_port and in_port.connected_ports():
            for connected in in_port.connected_ports():
                upstream_node = connected.node()
                up_val = upstream_node.output_values.get(connected.name(), None)
                if isinstance(up_val, TableData):
                    up_val = up_val.df
                elif hasattr(up_val, 'payload'):
                    up_val = up_val.payload
                in_values.append(up_val)
        
        if not in_values or in_values[0] is None:
            self.mark_error()
            return False, "No input data"
        
        data = in_values[0]

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
                
            elif isinstance(data, Image.Image):
                self.set_progress(50)
                data.save(file_path)

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
        then re-read the input value."""
        upstream = self._collect_upstream_sorted()
        for node in upstream:
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
