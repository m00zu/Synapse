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
    NodeDirSelector, NodeFileSaver,
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

        def _dedup(name, existing):
            base = name
            counter = 2
            while name in existing:
                name = f'{base}_{counter}'
                counter += 1
            return name

        # Read names from widget (already populated by on_input_connected)
        names = self._naming_widget.get_value()

        # Collect items — if an input is a CollectionData, merge its items in
        for i, cp in enumerate(connections):
            data = cp.node().output_values.get(cp.name())
            if data is None:
                continue
            if isinstance(data, CollectionData):
                # Merge all items from the incoming collection
                for key, val in data.payload.items():
                    key = _dedup(key, items)
                    items[key] = val
            else:
                name = names[i] if i < len(names) and names[i] else f'item_{i+1}'
                name = _dedup(name, items)
                items[name] = data

        if not items:
            return False, "No data from connected inputs"

        self.output_values['collection'] = CollectionData(payload=items)
        self.mark_clean()
        return True, None


# ===========================================================================
# Shared widgets for collection key selection
# ===========================================================================

class _KeySelectWidget(NodeBaseWidget):
    """Text input + dropdown menu for selecting a single collection key.

    Uses a QToolButton with ▼ that opens a QMenu (same as NodeColumnSelectorWidget).
    The text field is editable so users can pre-fill before data arrives.
    """

    selection_changed = QtCore.Signal()
    _update_items_sig = QtCore.Signal(list)

    def __init__(self, parent=None, name='_select_key', label='Key:'):
        super().__init__(parent, name=name, label='')
        self._items: list[str] = []

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet('color:#ccc; font-size:9px;')
        self._text = QtWidgets.QLineEdit()
        self._text.setPlaceholderText('item name')
        self._text.setMinimumWidth(80)
        self._text.editingFinished.connect(lambda: self.selection_changed.emit())

        self._btn = QtWidgets.QToolButton()
        self._btn.setText('▼')
        self._btn.setFixedWidth(22)
        self._btn.clicked.connect(self._show_menu)

        layout.addWidget(lbl)
        layout.addWidget(self._text, 1)
        layout.addWidget(self._btn)
        self.set_custom_widget(container)

        self._update_items_sig.connect(self._apply_items,
                                       QtCore.Qt.ConnectionType.QueuedConnection)

    def _show_menu(self):
        menu = QtWidgets.QMenu(QtWidgets.QApplication.activeWindow())
        for item in self._items:
            action = menu.addAction(item)
            action.triggered.connect(lambda checked, t=item: self._pick(t))
        if not self._items:
            a = menu.addAction('(run graph first)')
            a.setEnabled(False)
        menu.exec(QtGui.QCursor.pos())

    def _pick(self, text):
        self._text.setText(text)
        self.selection_changed.emit()

    def update_items(self, names: list[str]):
        import threading
        if threading.current_thread() is threading.main_thread():
            self._apply_items(names)
        else:
            self._update_items_sig.emit(names)

    def _apply_items(self, names: list[str]):
        self._items = list(names)
        if not self._text.text() and names:
            self._text.setText(names[0])

    def get_value(self) -> str:
        return self._text.text().strip()

    def set_value(self, value):
        if isinstance(value, str):
            self._text.setText(value)


class _MultiKeySelectWidget(NodeBaseWidget):
    """Text input (pipe-separated) + dropdown menu for selecting multiple keys.

    Uses a QToolButton with ▼ that opens a QMenu. Clicking a name toggles
    it in the text field. Pre-filling works: type names before data arrives.
    """

    selection_changed = QtCore.Signal()
    _update_items_sig = QtCore.Signal(list)

    def __init__(self, parent=None, name='_split_keys', label='Select:'):
        super().__init__(parent, name=name, label='')
        self._items: list[str] = []

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet('color:#ccc; font-size:9px;')
        self._text = QtWidgets.QLineEdit()
        self._text.setPlaceholderText('key1 | key2 | ...')
        self._text.setMinimumWidth(120)
        self._text.editingFinished.connect(lambda: self.selection_changed.emit())

        self._btn = QtWidgets.QToolButton()
        self._btn.setText('▼')
        self._btn.setFixedWidth(22)
        self._btn.clicked.connect(self._show_menu)

        layout.addWidget(lbl)
        layout.addWidget(self._text, 1)
        layout.addWidget(self._btn)
        self.set_custom_widget(container)

        self._update_items_sig.connect(self._apply_items,
                                       QtCore.Qt.ConnectionType.QueuedConnection)

    def _show_menu(self):
        existing = set(self.get_keys())
        menu = QtWidgets.QMenu(QtWidgets.QApplication.activeWindow())
        for item in self._items:
            action = menu.addAction(item)
            action.setCheckable(True)
            action.setChecked(item in existing)
            action.triggered.connect(lambda checked, t=item: self._toggle(t))
        if not self._items:
            a = menu.addAction('(run graph first)')
            a.setEnabled(False)
        menu.exec(QtGui.QCursor.pos())

    def _toggle(self, text):
        keys = self.get_keys()
        if text in keys:
            keys.remove(text)
        else:
            keys.append(text)
        self._text.setText(' | '.join(keys))
        self.selection_changed.emit()

    def update_items(self, names: list[str]):
        import threading
        if threading.current_thread() is threading.main_thread():
            self._apply_items(names)
        else:
            self._update_items_sig.emit(names)

    def _apply_items(self, names: list[str]):
        self._items = list(names)

    def get_keys(self) -> list[str]:
        raw = self._text.text().strip()
        if not raw:
            return []
        return [k.strip() for k in raw.split('|') if k.strip()]

    def get_value(self) -> str:
        return self._text.text().strip()

    def set_value(self, value):
        if isinstance(value, str):
            self._text.setText(value)


class SelectCollectionNode(BaseExecutionNode):
    """
    Extract a single item from a collection by name.

    Type a name or pick from the dropdown. The dropdown auto-populates
    with available item names when the collection is connected.

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

        self._select_widget = _KeySelectWidget(self.view, label='Item:')
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


# ===========================================================================
# Pop Collection — extract one item + output the remainder
# ===========================================================================

class PopCollectionNode(BaseExecutionNode):
    """
    Extract one item from a collection and output the rest separately.

    Two outputs: the extracted item on **item**, and a new collection
    without that item on **rest**.  Type a name or pick from the dropdown.

    Keywords: pop, remove, extract, collection, 彈出, 移除, 集合
    """
    __identifier__ = 'nodes.utility'
    NODE_NAME      = 'Pop Collection'
    PORT_SPEC      = {'inputs': ['collection'], 'outputs': ['any', 'collection']}
    _handles_collection = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('collection', color=PORT_COLORS['collection'])
        self.add_output('item', color=PORT_COLORS['any'])
        self.add_output('rest', color=PORT_COLORS['collection'])

        self._select_widget = _KeySelectWidget(self.view, label='Pop:')
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

        rest = {k: v for k, v in data.payload.items() if k != key}

        self.output_values['item'] = item
        self.output_values['rest'] = CollectionData(payload=rest)
        self.mark_clean()
        return True, None


class SplitCollectionNode(BaseExecutionNode):
    """
    Split a collection into two groups by selecting which items go to each output.

    Type item names separated by ' | ' or pick from the dropdown to add.
    Selected items go to **selected**, the rest go to **rest**.

    Keywords: split, partition, divide, collection, 分割, 分組, 集合
    """
    __identifier__ = 'nodes.utility'
    NODE_NAME      = 'Split Collection'
    PORT_SPEC      = {'inputs': ['collection'], 'outputs': ['collection', 'collection']}
    _handles_collection = True

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('collection', color=PORT_COLORS['collection'])
        self.add_output('selected', color=PORT_COLORS['collection'])
        self.add_output('rest', color=PORT_COLORS['collection'])

        self._select_widget = _MultiKeySelectWidget(self.view, label='Select:')
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

        self._select_widget.update_items(data.names)

        selected_keys = set(self._select_widget.get_keys())
        if not selected_keys:
            return False, "No items selected"

        selected = {}
        rest = {}
        for k, v in data.payload.items():
            if k in selected_keys:
                selected[k] = v
            else:
                rest[k] = v

        if not selected:
            return False, f"None of the selected keys found in collection"

        self.output_values['selected'] = CollectionData(payload=selected)
        self.output_values['rest'] = CollectionData(payload=rest)
        self.mark_clean()
        return True, None


class SaveCollectionNode(BaseExecutionNode):
    """
    Saves all items in a collection to disk.

    Each item is saved as a separate file using the item name as a suffix.
    Supports images (TIFF, PNG), tables (CSV, TSV), and figures.

    If a path is connected, it is used as the base — the item name is inserted
    before the extension.  Otherwise the folder + extension fields are used.

    Keywords: save, collection, batch save, export, 儲存, 集合, 批次儲存
    """
    __identifier__ = 'nodes.utility'
    NODE_NAME = 'Save Collection'
    PORT_SPEC = {'inputs': ['collection', 'path'], 'outputs': ['table']}
    _handles_collection = True

    def __init__(self):
        super().__init__()
        self.add_input('collection', color=PORT_COLORS.get('collection', (218, 165, 32)))
        self.add_input('file_path', color=PORT_COLORS.get('path'))
        self.add_output('status', color=PORT_COLORS.get('table'))

        folder_w = NodeDirSelector(self.view, name='folder', label='Output Folder')
        self.add_custom_widget(
            folder_w,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties',
        )
        self.add_text_input('prefix', 'Prefix', text='output')
        self.add_text_input('ext', 'Extension', text='.tif')

    def evaluate(self):
        import os
        from pathlib import Path

        self.reset_progress()

        # --- get collection ---
        col_port = self.inputs().get('collection')
        if not col_port or not col_port.connected_ports():
            self.mark_error()
            return False, "No collection connected"

        cp = col_port.connected_ports()[0]
        col = cp.node().output_values.get(cp.name())
        if not isinstance(col, CollectionData):
            self.mark_error()
            return False, "Input must be a CollectionData"

        if not col.payload:
            self.mark_error()
            return False, "Collection is empty"

        # --- resolve base path ---
        path_port = self.inputs().get('file_path')
        base_path = None
        if path_port and path_port.connected_ports():
            pc = path_port.connected_ports()[0]
            upstream = pc.node()
            port_name = pc.name()
            base_path = upstream.output_values.get(port_name)
            if hasattr(base_path, 'payload'):
                base_path = base_path.payload
            base_path = str(base_path) if base_path else None
        else:
            pass

        saved_rows = []
        items = list(col.payload.items())
        total = len(items)

        for idx, (name, item) in enumerate(items):
            self.set_progress(int((idx / total) * 100))

            # Build file path for this item
            if base_path:
                p = Path(base_path)
                file_path = str(p.parent / f"{p.stem}_{name}{p.suffix}")
            else:
                folder = self.get_property('folder') or '.'
                prefix = self.get_property('prefix') or 'output'
                ext = self.get_property('ext') or '.tif'
                if not ext.startswith('.'):
                    ext = '.' + ext
                file_path = os.path.join(folder, f"{prefix}_{name}{ext}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            try:
                self._save_item(item, file_path)
                saved_rows.append({'name': name, 'path': file_path, 'status': 'ok'})
            except Exception as e:
                saved_rows.append({'name': name, 'path': file_path, 'status': str(e)})

        self.output_values['status'] = TableData(payload=pd.DataFrame(saved_rows))
        ok = sum(1 for r in saved_rows if r['status'] == 'ok')
        errs = sum(1 for r in saved_rows if r['status'] != 'ok')
        self.mark_clean()
        self.set_progress(100)
        if errs:
            return True, f"Saved {ok} files, {errs} errors"
        return True, None

    def _save_item(self, item, file_path):
        """Save a single NodeData item to disk."""
        import matplotlib.figure

        ext = file_path.lower().rsplit('.', 1)[-1] if '.' in file_path else ''
        payload = item.payload if hasattr(item, 'payload') else item

        if isinstance(payload, pd.DataFrame):
            sep = '\t' if ext == 'tsv' else ','
            payload.to_csv(file_path, sep=sep, index=False)

        elif isinstance(payload, matplotlib.figure.Figure):
            payload.tight_layout()
            payload.savefig(file_path, bbox_inches='tight',
                            dpi=float(payload.get_dpi()))

        elif isinstance(payload, np.ndarray):
            bit_depth = getattr(item, 'bit_depth', 8) or 8
            scale_um = getattr(item, 'scale_um', None)

            if ext in ('tif', 'tiff'):
                import tifffile
                from ..nodes.io_nodes import _denormalize_from_float
                out_arr = _denormalize_from_float(payload, bit_depth)
                if scale_um and scale_um > 0:
                    px_per_cm = 10000.0 / scale_um
                    tifffile.imwrite(file_path, out_arr,
                                     resolution=(px_per_cm, px_per_cm),
                                     resolutionunit=3)
                else:
                    tifffile.imwrite(file_path, out_arr)
            else:
                from ..nodes.io_nodes import _denormalize_from_float
                out_arr = _denormalize_from_float(payload, 8)
                from PIL import Image as _PILImage
                pil = _PILImage.fromarray(out_arr)
                if scale_um and scale_um > 0:
                    ppi = int(1_000_000.0 / scale_um / 39.3701)
                    pil.save(file_path, dpi=(ppi, ppi))
                else:
                    pil.save(file_path)

        elif isinstance(payload, Image.Image):
            scale_um = getattr(item, 'scale_um', None)
            if scale_um and scale_um > 0:
                ppi = int(1_000_000.0 / scale_um / 39.3701)
                payload.save(file_path, dpi=(ppi, ppi))
            else:
                payload.save(file_path)
        else:
            raise ValueError(f"Unsupported data type: {type(payload).__name__}")
