"""
nodes/display_nodes.py
======================
Pop-up display and embedded cell viewer nodes.
"""
from ..data_models import TableData, FigureData
import pandas as pd
import numpy as np
from .base import (
    BaseExecutionNode, PORT_COLORS,
    NodeTableWidget, NodeImageWidget,
)


class DisplayNode(BaseExecutionNode):
    """
    Takes any input and pops up a preview window to inspect it.

    Supported data types:
    - *DataFrame* — shown as an editable table dialog
    - *Figure* — rendered to PNG and shown in a scrollable image dialog
    - *Image* — displayed as a scrollable PIL image dialog
    - *Other* — shown as a plain text message box

    Keywords: preview, inspect output, quick view, popup, debug display, 顯示, 預覽, 彈出視窗, 偵錯, 檢視
    """
    __identifier__ = 'nodes.display'
    NODE_NAME = 'Pop-up Display'
    PORT_SPEC = {'inputs': ['any'], 'outputs': []}

    def __init__(self):
        super(DisplayNode, self).__init__()
        self.add_input('in', multi_input=True, color=PORT_COLORS['any'])
        # Dedicated leaf node, intentionally omitting outward port to mark End Of Flow

    def evaluate(self):
        self.reset_progress()
        import traceback

        try:
            in_values = []
            for port in self.inputs().values():
                if port.connected_ports():
                    upstream_node = port.connected_ports()[0].node()
                    up_val = upstream_node.output_values.get(port.connected_ports()[0].name(), None)
                    in_values.append(up_val)

            if in_values:
                data = in_values[0]
                if hasattr(data, 'payload'):
                    data = data.payload
                elif hasattr(data, 'df'):
                    data = data.df

                self.set_display(data)

            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)

    def _display_ui(self, data):
        """Actual UI logic for display, runs on Main Thread."""
        import matplotlib.pyplot as plt
        import matplotlib.figure
        import tempfile
        import os
        import traceback
        import numpy as np
        from PIL import Image
        from PySide6 import QtWidgets, QtCore, QtGui

        try:
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    df = data
                    dialog = QtWidgets.QDialog()
                    dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
                    dialog.setWindowTitle(f"Output Table: {self.name()}")
                    layout = QtWidgets.QVBoxLayout(dialog)

                    table = QtWidgets.QTableWidget(df.shape[0], df.shape[1])
                    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
                    table.setVerticalHeaderLabels([str(i) for i in df.index])

                    for row in range(df.shape[0]):
                        for col in range(df.shape[1]):
                            item = QtWidgets.QTableWidgetItem(str(df.iat[row, col]))
                            table.setItem(row, col, item)

                    layout.addWidget(table)
                    dialog.resize(800, 600)

                    self._active_dialogs.append(dialog)
                    dialog.finished.connect(lambda r: self._active_dialogs.remove(dialog) if dialog in self._active_dialogs else None)

                    dialog.show()

                elif isinstance(data, matplotlib.figure.Figure):
                    fd, path = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    data.savefig(path, bbox_inches='tight', dpi=600)

                    dialog = QtWidgets.QDialog()
                    dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
                    dialog.setWindowTitle(f"Output Figure: {self.name()}")
                    layout = QtWidgets.QVBoxLayout(dialog)
                    scroll = QtWidgets.QScrollArea()
                    scroll.setWidgetResizable(True)
                    label = QtWidgets.QLabel()
                    pixmap = QtGui.QPixmap(path)

                    max_size = QtCore.QSize(1200, 800)
                    if pixmap.width() > max_size.width() or pixmap.height() > max_size.height():
                        pixmap = pixmap.scaled(max_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)

                    label.setPixmap(pixmap)
                    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    scroll.setWidget(label)
                    layout.addWidget(scroll)
                    dialog.resize(min(pixmap.width() + 40, 1240), min(pixmap.height() + 40, 840))

                    self._active_dialogs.append(dialog)

                    def _cleanup():
                        if dialog in self._active_dialogs:
                            self._active_dialogs.remove(dialog)
                        if os.path.exists(path):
                            os.remove(path)

                    dialog.finished.connect(_cleanup)
                    dialog.show()

                elif isinstance(data, np.ndarray):
                    from data_models import array_to_qpixmap
                    dialog = QtWidgets.QDialog()
                    dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
                    dialog.setWindowTitle(f"Output Image: {self.name()}")
                    layout = QtWidgets.QVBoxLayout(dialog)
                    scroll = QtWidgets.QScrollArea()
                    scroll.setWidgetResizable(True)
                    label = QtWidgets.QLabel()
                    pixmap = array_to_qpixmap(data)

                    max_size = QtCore.QSize(1200, 800)
                    if pixmap.width() > max_size.width() or pixmap.height() > max_size.height():
                        pixmap = pixmap.scaled(max_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)

                    label.setPixmap(pixmap)
                    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    scroll.setWidget(label)
                    layout.addWidget(scroll)
                    dialog.resize(min(pixmap.width() + 40, 1240), min(pixmap.height() + 40, 840))

                    self._active_dialogs.append(dialog)
                    dialog.finished.connect(lambda r: self._active_dialogs.remove(dialog) if dialog in self._active_dialogs else None)
                    dialog.show()

                elif isinstance(data, Image.Image):
                    from data_models import array_to_qpixmap
                    dialog = QtWidgets.QDialog()
                    dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
                    dialog.setWindowTitle(f"Output Image: {self.name()}")
                    layout = QtWidgets.QVBoxLayout(dialog)
                    scroll = QtWidgets.QScrollArea()
                    scroll.setWidgetResizable(True)
                    label = QtWidgets.QLabel()
                    pixmap = array_to_qpixmap(np.asarray(data))

                    max_size = QtCore.QSize(1200, 800)
                    if pixmap.width() > max_size.width() or pixmap.height() > max_size.height():
                        pixmap = pixmap.scaled(max_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)

                    label.setPixmap(pixmap)
                    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    scroll.setWidget(label)
                    layout.addWidget(scroll)
                    dialog.resize(min(pixmap.width() + 40, 1240), min(pixmap.height() + 40, 840))

                    self._active_dialogs.append(dialog)
                    dialog.finished.connect(lambda r: self._active_dialogs.remove(dialog) if dialog in self._active_dialogs else None)
                    dialog.show()
                else:
                    QtWidgets.QMessageBox.information(None, f"Output: {self.name()}", str(data))
        except Exception as e:
            print(f"Error in DisplayNode UI Handler: {e}")
            traceback.print_exc()


class DataTableCellNode(BaseExecutionNode):
    """
    Displays incoming DataFrame data directly on the node surface.

    Keywords: table viewer, inline dataframe, spreadsheet view, inspect table, 表格, 顯示, 資料框, 檢視, 內嵌
    """
    __identifier__ = 'nodes.display'
    NODE_NAME = 'Data Table Node'
    PORT_SPEC = {'inputs': ['table'], 'outputs': []}

    def __init__(self):
        super(DataTableCellNode, self).__init__(use_progress=False)
        self.add_input('in', multi_input=False, color=PORT_COLORS['table'])

        self._table_widget = NodeTableWidget(self.view)
        self.add_custom_widget(self._table_widget, tab='View')

    def evaluate(self):
        self.reset_progress()
        in_values = []
        in_port = self.inputs().get('in')
        if in_port and in_port.connected_ports():
            upstream_node = in_port.connected_ports()[0].node()
            up_val = upstream_node.output_values.get(in_port.connected_ports()[0].name(), None)
            if isinstance(up_val, TableData):
                up_val = up_val.df
            elif hasattr(up_val, 'payload'):
                up_val = up_val.payload
            in_values.append(up_val)

        data = in_values[0] if in_values else None
        self.set_display(data)
        self.mark_clean()
        return True, None

    def _display_ui(self, data):
        """Updates the embedded table widget (Main Thread only)."""
        from PySide6 import QtWidgets
        if isinstance(data, pd.DataFrame):
            # Display NaN as empty string — keep underlying data intact
            self._table_widget.set_value(data.fillna(''))
            self.view.draw_node()
        else:
            self._table_widget.set_value(None)


class DataFigureCellNode(BaseExecutionNode):
    """
    Displays incoming Image or Figure data directly on the node surface.

    Accepts `FigureData` (with optional SVG override) or raw matplotlib figures.

    Keywords: figure viewer, plot preview, inline chart, matplotlib display, 圖形, 顯示, 圖表預覽, 內嵌, 繪圖
    """
    __identifier__ = 'nodes.display'
    NODE_NAME = 'Data Figure Node'
    PORT_SPEC = {'inputs': ['figure'], 'outputs': []}

    def __init__(self):
        super(DataFigureCellNode, self).__init__(use_progress=False)
        self.add_input('in', multi_input=False, color=PORT_COLORS['figure'])

        self._figure_widget = NodeImageWidget(self.view)
        self.add_custom_widget(self._figure_widget, tab='View')

    def evaluate(self):
        self.reset_progress()
        in_port = self.inputs().get('in')
        if not in_port or not in_port.connected_ports():
            return False, 'No input connected'
        cp = in_port.connected_ports()[0]
        up_val = cp.node().output_values.get(cp.name())
        if up_val is None:
            return False, 'Input must be FigureData'

        # If upstream provides edited SVG, display that directly
        if isinstance(up_val, FigureData) and up_val.svg_override:
            self.set_display(up_val.svg_override)
            self.mark_clean()
            return True, None

        if isinstance(up_val, FigureData):
            data = up_val.fig
        elif hasattr(up_val, 'payload'):
            data = up_val.payload
        else:
            return False, 'Input must be FigureData'

        self.set_display(data)
        self.mark_clean()
        return True, None

    def _display_ui(self, data):
        """Updates the embedded figure widget (Main Thread only)."""
        import matplotlib.figure
        if isinstance(data, (bytes, bytearray)):
            # SVG bytes from SvgEditorNode
            self._figure_widget.set_value(data)
            self.view.draw_node()
        elif isinstance(data, matplotlib.figure.Figure):
            data.tight_layout()
            self._figure_widget.set_value(data)
            self.view.draw_node()
        else:
            self._figure_widget.set_value(None)


class ImageCellNode(BaseExecutionNode):
    """
    Displays a PIL Image directly on the node surface for quick inline inspection.

    Accepted input types:
    - *ImageData* — unwraps the payload
    - *LabelData* — uses the pre-generated colored visualization
    - *Raw PIL Image* — displayed as-is

    Keywords: image viewer, inline image, photo preview, mask preview, quick inspect, 影像, 顯示, 預覽, 遮罩, 內嵌
    """
    __identifier__ = 'nodes.display'
    NODE_NAME = 'Image Viewer'
    PORT_SPEC = {'inputs': ['image'], 'outputs': []}

    def __init__(self):
        super(ImageCellNode, self).__init__(use_progress=False)
        self.add_input('in', color=PORT_COLORS['image'])

        self._image_widget = NodeImageWidget(self.view)
        self.add_custom_widget(self._image_widget, tab='View')

    def evaluate(self):
        from data_models import ImageData, LabelData
        from PIL import Image
        import numpy as np

        in_port = self.inputs().get('in')
        data = None
        if in_port and in_port.connected_ports():
            upstream_node = in_port.connected_ports()[0].node()
            raw = upstream_node.output_values.get(in_port.connected_ports()[0].name(), None)
            if isinstance(raw, LabelData):
                data = raw.image  # pre-generated colored visualization
            elif isinstance(raw, ImageData):
                data = raw.payload  # numpy array
            elif isinstance(raw, np.ndarray):
                data = raw
            elif isinstance(raw, Image.Image):
                data = np.asarray(raw)
            elif hasattr(raw, 'payload'):
                inner = raw.payload
                if isinstance(inner, np.ndarray):
                    data = inner
                elif isinstance(inner, Image.Image):
                    data = np.asarray(inner)

        self.set_display(data)
        self.mark_clean()
        return True, None

    def _display_ui(self, data):
        """Renders image data into the embedded widget (Main Thread only)."""
        import numpy as np
        from PIL import Image
        if isinstance(data, (np.ndarray, Image.Image)):
            self._image_widget.set_value(data)
            self.view.draw_node()
        else:
            self._image_widget.set_value(None)
            self.view.draw_node()

