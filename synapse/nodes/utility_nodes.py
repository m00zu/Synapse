"""
nodes/utility_nodes.py
======================
General-purpose utility nodes.
"""
import NodeGraphQt
from ..data_models import TableData, ImageData, CollectionData
from PIL import Image
import pandas as pd
import numpy as np
from .base import (
    BaseExecutionNode, PORT_COLORS,
    NodeDirSelector,
)


class UniversalDataNode(BaseExecutionNode):
    """
    Executes arbitrary Python code to process multiple inputs and push results to outputs.

    Available variables in user code:
    - `inputs` — list of upstream data values
    - `output` — assign the result here (auto-wrapped into `TableData`, `ImageData`, or `FigureData`)
    - `pd`, `np`, `plt`, `sns` — pre-imported libraries

    Keywords: script, python, custom logic, arbitrary code, transform, 腳本, 工具, 自訂邏輯, 通用, 轉換
    """
    __identifier__ = 'nodes.data'
    NODE_NAME = 'Universal Node'
    PORT_SPEC = {'inputs': ['any'], 'outputs': ['any']}

    def __init__(self):
        super(UniversalDataNode, self).__init__()

        self.add_input('in', multi_input=True, color=PORT_COLORS['any'])
        self.add_output('out', multi_output=True, color=PORT_COLORS['any'])

        self.add_text_input('code', 'Python Code', tab='Execution')
        self.set_property('code', 'output = inputs[0]')
        
        self.output_values = {}

    def evaluate(self):
        """
        Executes the python code with 'inputs' representing a list of upstream data.
        'outputs' should be filled by the user's python code as a list.
        """
        self.reset_progress()
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
                
        local_scope = {
            "inputs": in_values, 
            "output": None
        }
        
        func_str = self.get_property("code")
        
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            local_scope['pd'] = pd
            local_scope['np'] = np
            local_scope['plt'] = plt
            local_scope['sns'] = sns
            
            exec(func_str, globals(), local_scope)
            
            if "output" in local_scope:
                res = local_scope["output"]
                if isinstance(res, (pd.DataFrame, pd.Series)):
                    from data_models import TableData
                    res = TableData(payload=res)
                elif isinstance(res, Image.Image):
                    res = ImageData(payload=np.asarray(res).astype(np.float32) / 255.0, bit_depth=8)
                elif isinstance(res, np.ndarray) and res.ndim in (2, 3):
                    res = ImageData(payload=res)
                elif hasattr(res, '__class__') and 'Figure' in type(res).__name__:
                    from data_models import FigureData
                    res = FigureData(payload=res)
                
                self.output_values['out'] = res
            
            self.mark_clean()
            self.set_progress(100)
            return True, None
            
        except Exception as e:
            self.mark_error()
            return False, str(e)


class PathModifierNode(BaseExecutionNode):
    """
    Takes a file path and modifies it by adding a suffix, changing the extension, or overriding the folder.

    **suffix** — string appended to the file stem (default: `_analyzed`).
    **ext** — replacement file extension (leave empty to keep original).
    **folder** — optional folder override for the output path.

    Keywords: path, filename, suffix, extension, rename, 路徑, 檔名, 副檔名, 工具, 重新命名
    """
    __identifier__ = 'nodes.utility'
    NODE_NAME = 'Path Modifier'
    PORT_SPEC = {'inputs': ['path'], 'outputs': ['path']}

    def __init__(self):
        super(PathModifierNode, self).__init__(use_progress=False)
        self.add_input('path', color=PORT_COLORS['path'])
        self.add_output('path', color=PORT_COLORS['path'])
        
        self.add_text_input('suffix', 'Suffix', text='_analyzed')
        self.add_text_input('ext', 'New Extension', text='')
        folder_selector = NodeDirSelector(self.view, name='folder', label='Folder Override')
        self.add_custom_widget(
            folder_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties'
        )

    def evaluate(self):
        from pathlib import Path
        
        in_port = self.inputs().get('path')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input path"
            
        connected = in_port.connected_ports()[0]
        upstream_node = connected.node()
        orig_path_val = upstream_node.output_values.get(connected.name(), None)
        
        if not orig_path_val:
            self.mark_error()
            return False, "Input path is empty"
        
        if isinstance(orig_path_val, str):
            orig_path_str = orig_path_val
        else:
            self.mark_error()
            return False, f"Unsupported path input type: {type(orig_path_val).__name__}"

        orig_path = Path(orig_path_str)
        suffix = self.get_property('suffix')
        new_ext = self.get_property('ext')
        folder_override = self.get_property('folder')
        
        stem = orig_path.stem
        ext = new_ext if new_ext else orig_path.suffix
        if ext and not ext.startswith('.'):
            ext = '.' + ext
            
        new_filename = f"{stem}{suffix}{ext}"
        
        if folder_override:
            new_path = Path(folder_override) / new_filename
        else:
            new_path = orig_path.parent / new_filename
            
        self.output_values['path'] = str(new_path)
        self.mark_clean()
        return True, None


# ===========================================================================
# Collect — pack multiple data items into a named CollectionData
# ===========================================================================

from PySide6 import QtWidgets, QtCore, QtGui
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget


class _CollectNamingWidget(NodeBaseWidget):
    """Editable list showing one row per connected input with a name field."""

    names_changed = QtCore.Signal()
    _update_sig = QtCore.Signal(list)  # thread-safe update

    def __init__(self, parent=None):
        super().__init__(parent, name='_collect_names', label='')
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._list = QtWidgets.QWidget()
        self._list_layout = QtWidgets.QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(2)
        layout.addWidget(self._list)

        self._edits: list[QtWidgets.QLineEdit] = []
        self.set_custom_widget(container)
        self._update_sig.connect(self._apply_update,
                                 QtCore.Qt.ConnectionType.QueuedConnection)

    def update_connections(self, port_names: list[str]):
        """Update the list to show one editable row per connected input.
        Thread-safe."""
        import threading
        if threading.current_thread() is threading.main_thread():
            self._apply_update(port_names)
        else:
            self._update_sig.emit(port_names)

    def _apply_update(self, port_names: list[str]):
        # Preserve existing names where possible
        old_names = self.get_value()

        # Clear existing widgets
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
        self._edits.clear()

        # Create rows — each is: "1. [name_field]"
        for i, upstream_name in enumerate(port_names):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(4)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QtWidgets.QLabel(f'{i+1}.')
            lbl.setStyleSheet('color:#aaa; font-size:10px;')
            lbl.setFixedWidth(18)
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(upstream_name or f'item_{i+1}')
            if i < len(old_names) and old_names[i]:
                edit.setText(old_names[i])
            else:
                edit.setText(upstream_name or f'item_{i+1}')
            edit.setStyleSheet('font-size:10px; padding: 2px 4px;')
            edit.setFixedHeight(22)
            edit.setMinimumWidth(120)
            edit.textChanged.connect(lambda _: self.names_changed.emit())
            row.addWidget(lbl)
            row.addWidget(edit, 1)
            self._list_layout.addLayout(row)
            self._edits.append(edit)

        # Set explicit height based on number of rows
        row_h = 26  # each row: 22px edit + 4px spacing
        total_h = max(30, len(self._edits) * row_h + 4)
        self._list.setFixedHeight(total_h)

        # Defer node redraw to after layout is fully computed
        node = getattr(self, '_node', None)
        if node and hasattr(node, 'view') and hasattr(node.view, 'draw_node'):
            QtCore.QTimer.singleShot(0, node.view.draw_node)

    def get_value(self) -> list[str]:
        return [e.text() for e in self._edits]

    def set_value(self, value):
        if isinstance(value, list):
            for i, v in enumerate(value[:len(self._edits)]):
                self._edits[i].blockSignals(True)
                self._edits[i].setText(str(v))
                self._edits[i].blockSignals(False)


class CollectNode(BaseExecutionNode):
    """
    Pack multiple data items into a named collection.

    Connect any number of items to the multi-input port. Each connection
    gets a name (auto-populated from the upstream port name, editable).
    The output is a single CollectionData that flows as one wire.

    Downstream nodes that expect a single item will automatically loop
    over all items in the collection and repack the results.

    Keywords: collect, pack, bundle, group, collection, batch, 收集, 打包, 集合
    """
    __identifier__ = 'nodes.utility'
    NODE_NAME      = 'Collect'
    PORT_SPEC      = {'inputs': ['any'], 'outputs': ['collection']}
    _handles_collection = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('in', multi_input=True, color=PORT_COLORS['any'])
        self.add_output('collection', color=PORT_COLORS['collection'])

        self._naming_widget = _CollectNamingWidget(self.view)
        self._naming_widget._node = self  # back-reference for node resize
        self._naming_widget.names_changed.connect(self._on_names_changed)
        self.add_custom_widget(self._naming_widget)

    def on_input_connected(self, in_port, out_port):
        """Update naming widget when a new wire is connected (main thread)."""
        super().on_input_connected(in_port, out_port)
        self._refresh_naming_widget()

    def on_input_disconnected(self, in_port, out_port):
        """Update naming widget when a wire is disconnected (main thread)."""
        super().on_input_disconnected(in_port, out_port)
        self._refresh_naming_widget()

    def _refresh_naming_widget(self):
        """Rebuild the naming list from current connections."""
        port = self.inputs().get('in')
        if not port:
            return
        port_names = [cp.name() for cp in port.connected_ports()]
        self._naming_widget.update_connections(port_names)

    def _on_names_changed(self):
        """User edited a name — re-evaluate to update output."""
        success, _ = self.evaluate()
        if success:
            self.mark_clean()

    def evaluate(self):
        port = self.inputs().get('in')
        if not port or not port.connected_ports():
            return False, "No inputs connected"

        connections = port.connected_ports()
        items = {}

        # Read names from widget (already populated by on_input_connected)
        names = self._naming_widget.get_value()

        # Collect items
        for i, cp in enumerate(connections):
            data = cp.node().output_values.get(cp.name())
            if data is None:
                continue
            name = names[i] if i < len(names) and names[i] else f'item_{i+1}'
            # Deduplicate names
            base = name
            counter = 2
            while name in items:
                name = f'{base}_{counter}'
                counter += 1
            items[name] = data

        if not items:
            return False, "No data from connected inputs"

        self.output_values['collection'] = CollectionData(payload=items)
        self.mark_clean()
        return True, None


# ===========================================================================
# Select Collection — extract one item from a CollectionData
# ===========================================================================

class _SelectDropdownWidget(NodeBaseWidget):
    """Dropdown that auto-populates with collection item names."""

    selection_changed = QtCore.Signal()
    _update_items_sig = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent, name='_select_key', label='')
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        lbl = QtWidgets.QLabel('Item:')
        lbl.setStyleSheet('color:#ccc; font-size:9px;')
        self._combo = QtWidgets.QComboBox()
        self._combo.setMinimumWidth(100)
        self._combo.currentTextChanged.connect(lambda _: self.selection_changed.emit())

        layout.addWidget(lbl)
        layout.addWidget(self._combo)
        layout.addStretch()
        self.set_custom_widget(container)

        self._update_items_sig.connect(self._apply_items,
                                       QtCore.Qt.ConnectionType.QueuedConnection)

    def update_items(self, names: list[str]):
        import threading
        if threading.current_thread() is threading.main_thread():
            self._apply_items(names)
        else:
            self._update_items_sig.emit(names)

    def _apply_items(self, names: list[str]):
        old = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(names)
        if old in names:
            self._combo.setCurrentText(old)
        self._combo.blockSignals(False)

    def get_value(self) -> str:
        return self._combo.currentText()

    def set_value(self, value):
        if isinstance(value, str):
            idx = self._combo.findText(value)
            if idx >= 0:
                self._combo.setCurrentText(value)


class SelectCollectionNode(BaseExecutionNode):
    """
    Extract a single item from a collection by name.

    Connect a CollectionData input, pick which item to extract from the
    dropdown. The dropdown auto-populates with available item names.

    Keywords: select, extract, unpack, pick, collection, 選擇, 提取, 集合
    """
    __identifier__ = 'nodes.utility'
    NODE_NAME      = 'Select Collection'
    PORT_SPEC      = {'inputs': ['collection'], 'outputs': ['any']}
    _handles_collection = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('collection', color=PORT_COLORS['collection'])
        self.add_output('out', color=PORT_COLORS['any'])

        self._select_widget = _SelectDropdownWidget(self.view)
        self._select_widget.selection_changed.connect(self._on_selection_changed)
        self.add_custom_widget(self._select_widget)

    def _on_selection_changed(self):
        success, _ = self.evaluate()
        if success:
            self.mark_clean()

    def evaluate(self):
        port = self.inputs().get('collection')
        if not port or not port.connected_ports():
            return False, "No collection connected"

        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())

        if not isinstance(data, CollectionData):
            return False, "Input must be a CollectionData"

        # Update dropdown with available names
        self._select_widget.update_items(data.names)

        key = self._select_widget.get_value()
        if not key:
            if data.names:
                key = data.names[0]
            else:
                return False, "Collection is empty"

        item = data.get(key)
        if item is None:
            return False, f"Item '{key}' not found in collection"

        self.output_values['out'] = item
        self.mark_clean()
        return True, None
