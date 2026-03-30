# Nuitka runtime patch — fixes regionprops KeyError in frozen builds.
# Package bundling is handled by --include-package flags in the build command.
if "__compiled__" in globals():
    from . import skimage_nuitka_compat  # noqa: F401
import sys, time
import json

# Compatibility shims: external plugins and NodeGraphQt use bare imports like
# `from nodes.base import ...`, `from custom_nodes import ...`, etc.
# Register aliases so those resolve to synapse.* when installed via pip.
from synapse import nodes as _nodes_pkg, data_models as _dm_pkg, custom_nodes as _cn_pkg
from synapse.nodes import base as _nodes_base_pkg
sys.modules.setdefault('nodes', _nodes_pkg)
sys.modules.setdefault('nodes.base', _nodes_base_pkg)
sys.modules.setdefault('data_models', _dm_pkg)
sys.modules.setdefault('custom_nodes', _cn_pkg)
from PySide6 import QtCore, QtWidgets, QtGui
from NodeGraphQt import NodeGraph, PropertiesBinWidget, NodesTreeWidget
from .custom_nodes import (
    # Core I/O
    FileReadNode, FolderIteratorNode, VideoIteratorNode,
    ImageReadNode, SaveNode,
    BatchAccumulatorNode, BatchGateNode,
    # Core display
    DisplayNode, DataTableCellNode, DataFigureCellNode, ImageCellNode,
    # Core utility
    UniversalDataNode, PathModifierNode,
    CollectNode, SelectCollectionNode, PopCollectionNode,
    SplitCollectionNode, SaveCollectionNode,
    RenameCollectionNode, CollectionInfoNode, FilterCollectionNode, MapNamesNode,
)
import traceback
import os
from .nodes.base import BaseExecutionNode
from .i18n import tr, get_language, load_language, set_language
from .minimap import MinimapWidget

def clear_node_cache(graph, node):
    """
    Context menu function to clear a node's cache.
    """
    if hasattr(node, 'clear_cache'):
        node.clear_cache()
    # Also iterate over selected nodes in case multiple were selected
    for n in graph.selected_nodes():
        if hasattr(n, 'clear_cache'):
            n.clear_cache()


def toggle_node_disable(graph, node):
    """
    Context menu function to toggle a node's disabled state.
    Operates on all selected nodes; falls back to the right-clicked node.
    """
    targets = graph.selected_nodes() or [node]
    for n in targets:
        if not hasattr(n, 'is_disabled'):
            continue
        if n.is_disabled:
            n.mark_enabled()
        else:
            n.mark_disabled()

class GraphWorker(QtCore.QObject):
    """
    Worker object that runs the topological execution in a background thread.
    """
    finished = QtCore.Signal()
    error = QtCore.Signal(str)
    
    def __init__(self, sorted_nodes):
        super(GraphWorker, self).__init__()
        self.sorted_nodes = sorted_nodes
        self._stopped = False

    def stop(self):
        self._stopped = True
        BaseExecutionNode.request_cancel()

    def run(self):
        BaseExecutionNode.clear_cancel()
        try:
            # 1. Propagate dirty state DOWNSTREAM first
            for node in self.sorted_nodes:
                if hasattr(node, 'is_dirty'):
                    for in_port in node.inputs().values():
                        for connected_port in in_port.connected_ports():
                            upstream_node = connected_port.node()
                            if getattr(upstream_node, 'is_dirty', False):
                                node.mark_dirty()

            # 2. Execute in sorted order
            for node in self.sorted_nodes:
                if self._stopped:
                    self.finished.emit()
                    return
                if not hasattr(node, 'evaluate'):
                    continue
                # Skip execution if the node already computed its output and hasn't changed.
                if getattr(node, 'is_dirty', True) is False:
                    continue
                # Skip explicitly disabled nodes
                if getattr(node, 'is_disabled', False):
                    node.mark_skipped()
                    continue

                # Auto-loop: if the node receives a CollectionData but
                # doesn't handle collections natively, loop over each item.
                col_info = (node._check_collection_inputs()
                            if hasattr(node, '_check_collection_inputs') else None)
                if col_info and getattr(node, '_collection_aware', False):
                    success, err = node._evaluate_collection_loop(*col_info)
                else:
                    success, err = node.evaluate()
                if self._stopped:
                    self.finished.emit()
                    return
                if not success:
                    node.mark_error()
                    self.error.emit(f"Error in node '{node.name()}':\n{err}")
                    return
                else:
                    node.mark_clean()

            self.finished.emit()
        except Exception:
            self.error.emit(f"Fatal error during execution:\n{traceback.format_exc()}")


class BatchGraphWorker(QtCore.QObject):
    """
    Worker object that runs the topological execution multiple times for batch processing.
    """
    finished = QtCore.Signal()  # reverted: no run/skip count
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int, int) # Current file index, total files
    file_failed = QtCore.Signal(str, str)  # file name, error text
    
    def __init__(self, sorted_nodes, iterator_node, file_list):
        super(BatchGraphWorker, self).__init__()
        self.sorted_nodes = sorted_nodes
        self.iterator_node = iterator_node
        self.file_list = file_list
        self._stopped = False

    def stop(self):
        self._stopped = True
        BaseExecutionNode.request_cancel()

    def _run_downstream_only(self):
        """Skip the batch loop — just propagate from accumulators and
        evaluate only nodes downstream of accumulators.
        Returns True if stopped early."""
        # Collect nodes that are downstream of (or are) accumulators
        downstream = set()
        for node in self.sorted_nodes:
            if getattr(node, '_is_accumulator', False):
                downstream.add(id(node))
                continue
            # A node is downstream if any of its inputs come from a
            # node already in the downstream set
            for in_port in node.inputs().values():
                for cp in in_port.connected_ports():
                    if id(cp.node()) in downstream:
                        downstream.add(id(node))
                        break
                if id(node) in downstream:
                    break

        # Mark downstream (non-accumulator) nodes dirty
        for node in self.sorted_nodes:
            if id(node) in downstream and not getattr(node, '_is_accumulator', False):
                node.mark_dirty()

        # Evaluate only downstream nodes
        for node in self.sorted_nodes:
            if self._stopped:
                return True
            if id(node) not in downstream:
                continue
            if getattr(node, '_is_accumulator', False):
                node.mark_clean()
                continue
            if hasattr(node, 'evaluate'):
                if getattr(node, 'is_dirty', True) is False:
                    continue
                col_info = (node._check_collection_inputs()
                            if hasattr(node, '_check_collection_inputs') else None)
                if col_info and getattr(node, '_collection_aware', False):
                    success, err = node._evaluate_collection_loop(*col_info)
                else:
                    success, err = node.evaluate()
                if self._stopped:
                    return True
                if not success:
                    node.mark_error()
                    self.error.emit(
                        f"Error in node '{node.name()}':\n{err}")
                    return False
                else:
                    node.mark_clean()
        return False

    def run(self):
        BaseExecutionNode.clear_cancel()
        try:
            total = len(self.file_list)

            # Check if all accumulators already have completed batch data.
            # If so, skip the entire loop and jump to downstream evaluation.
            accumulators = [n for n in self.sorted_nodes
                           if getattr(n, '_is_accumulator', False)]
            all_complete = (accumulators
                           and all(getattr(a, '_batch_complete', False)
                                   for a in accumulators))

            if all_complete:
                # Mark accumulators dirty so downstream propagation works
                for a in accumulators:
                    a.is_dirty = True
                # Jump straight to post-batch downstream evaluation (step 5+)
                self._run_downstream_only()
                self.finished.emit()
                return

            # Lifecycle: notify accumulator nodes that batch is starting
            for node in self.sorted_nodes:
                if hasattr(node, 'on_batch_start'):
                    node.on_batch_start()

            # Pre-compute nodes DOWNSTREAM of accumulators — these must
            # NOT run during per-file iterations (accumulators defer
            # their merged output to on_batch_end).
            # Note: accumulators themselves ARE evaluated each iteration
            # (to collect values) — only nodes after them are deferred.
            _accumulator_ids = set()
            _post_batch_ids = set()
            for node in self.sorted_nodes:
                if getattr(node, '_is_accumulator', False):
                    _accumulator_ids.add(id(node))
                    continue
                for in_p in node.inputs().values():
                    for cp in in_p.connected_ports():
                        up_id = id(cp.node())
                        if up_id in _accumulator_ids or up_id in _post_batch_ids:
                            _post_batch_ids.add(id(node))
                            break
                    if id(node) in _post_batch_ids:
                        break

            for i, file_path in enumerate(self.file_list):
                if self._stopped:
                    break
                self.progress.emit(i + 1, total)

                # Inform gate nodes of current item so they can show progress info
                for node in self.sorted_nodes:
                    if hasattr(node, 'set_batch_item'):
                        node.set_batch_item(i + 1, total, file_path)

                # 1. Update the iterator node with current file
                # Use str() for serialization compatibility
                # Bypass the UI redraw by setting the model property directly
                self.iterator_node.model.set_property('current_file', str(file_path))

                # 2. Mark everything dirty EXCEPT accumulators and
                #    nodes downstream of accumulators
                for node in self.sorted_nodes:
                    nid = id(node)
                    if nid not in _accumulator_ids and nid not in _post_batch_ids:
                        node.mark_dirty()

                # 3. Propagate dirty state DOWNSTREAM
                for node in self.sorted_nodes:
                    if id(node) in _post_batch_ids:
                        continue
                    if hasattr(node, 'is_dirty'):
                        for in_port in node.inputs().values():
                            for connected_port in in_port.connected_ports():
                                upstream_node = connected_port.node()
                                if getattr(upstream_node, 'is_dirty', False):
                                    node.mark_dirty()

                # 4. Execute in sorted order (skip post-batch nodes)
                file_failed = False
                for node in self.sorted_nodes:
                    if self._stopped:
                        break
                    if id(node) in _post_batch_ids:
                        continue
                    if hasattr(node, 'evaluate'):
                        if getattr(node, 'is_dirty', True) is False:
                            continue

                        col_info = (node._check_collection_inputs()
                                    if hasattr(node, '_check_collection_inputs') else None)
                        if col_info and getattr(node, '_collection_aware', False):
                            success, err = node._evaluate_collection_loop(*col_info)
                        else:
                            success, err = node.evaluate()
                        if self._stopped:
                            break
                        if not success:
                            node.mark_error()
                            self.file_failed.emit(
                                os.path.basename(file_path),
                                f"Error in node '{node.name()}':\n{err}",
                            )
                            file_failed = True
                            break
                        else:
                            node.mark_clean()
                if self._stopped:
                    break
                if file_failed:
                    continue

            if not self._stopped:
                # Lifecycle: notify accumulator nodes that batch is finished, let them merge
                for node in self.sorted_nodes:
                    if hasattr(node, 'on_batch_end'):
                        node.on_batch_end()

                # 5. Propagate dirty state from accumulators DOWNSTREAM
                for node in self.sorted_nodes:
                    if hasattr(node, 'is_dirty'):
                        for in_port in node.inputs().values():
                            for connected_port in in_port.connected_ports():
                                upstream_node = connected_port.node()
                                if getattr(upstream_node, '_is_accumulator', False) or getattr(upstream_node, 'is_dirty', False):
                                    node.mark_dirty()

                # 6. Execute downstream nodes (like tables) that were waiting on accumulators
                for node in self.sorted_nodes:
                    if self._stopped:
                        break
                    # Accumulators are already done via on_batch_end, skip them here
                    if hasattr(node, 'evaluate') and not getattr(node, '_is_accumulator', False):
                        if getattr(node, 'is_dirty', True) is False:
                            continue

                        col_info = (node._check_collection_inputs()
                                    if hasattr(node, '_check_collection_inputs') else None)
                        if col_info and getattr(node, '_collection_aware', False):
                            success, err = node._evaluate_collection_loop(*col_info)
                        else:
                            success, err = node.evaluate()
                        if self._stopped:
                            break
                        if not success:
                            node.mark_error()
                            self.error.emit(f"Error in node '{node.name()}' during final post-batch completion:\n{err}")
                            self.finished.emit()
                            return
                        else:
                            node.mark_clean()

            self.finished.emit()
        except Exception:
            self.error.emit(f"Fatal error during batch execution:\n{traceback.format_exc()}")

# ── Theme ──────────────────────────────────────────────────────────────────

_DARK_STYLESHEET = """
QWidget                          { color: #ffffff; }
QMainWindow, QDialog             { background-color: #353535; }
QDockWidget                      { background-color: #353535; color: #ffffff; }
QDockWidget::title               { background-color: #404040; padding: 4px; }
QToolBar                         { background-color: #353535; border: none; spacing: 2px; }
QToolBar QToolButton             { background-color: transparent; color: #ffffff; padding: 3px 6px; border: none; border-radius: 3px; }
QToolBar QToolButton:hover       { background-color: #4a4a4a; }
QToolBar QToolButton:pressed     { background-color: #2e2e2e; }
QMenuBar                         { background-color: #353535; color: #ffffff; }
QMenuBar::item:selected          { background-color: #2a82da; }
QMenu                            { background-color: #353535; color: #ffffff; border: 1px solid #555555; }
QMenu::item:selected             { background-color: #2a82da; }
QMenu::separator                 { background-color: #555555; height: 1px; }
QStatusBar                       { background-color: #353535; color: #ffffff; }
QSplitter::handle                { background-color: #555555; }
QScrollBar:vertical              { background-color: #353535; width: 12px; border: none; margin: 0; }
QScrollBar::handle:vertical      { background-color: #606060; border-radius: 6px; min-height: 20px; margin: 2px; }
QScrollBar::handle:vertical:hover{ background-color: #808080; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical    { height: 0; border: none; }
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical    { background: none; }
QScrollBar:horizontal            { background-color: #353535; height: 12px; border: none; margin: 0; }
QScrollBar::handle:horizontal    { background-color: #606060; border-radius: 6px; min-width: 20px; margin: 2px; }
QScrollBar::handle:horizontal:hover{ background-color: #808080; }
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal  { width: 0; border: none; }
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal  { background: none; }
QLineEdit                        { background-color: #191919; color: #ffffff; border: 1px solid #555555; border-radius: 3px; padding: 2px 6px; selection-background-color: #2a82da; }
QTextEdit, QPlainTextEdit        { background-color: #191919; color: #ffffff; border: 1px solid #555555; selection-background-color: #2a82da; }
QPushButton                      { background-color: #454545; color: #ffffff; border: 1px solid #606060; border-radius: 3px; padding: 4px 12px; min-height: 22px; }
QPushButton:hover                { background-color: #505050; }
QPushButton:pressed              { background-color: #303030; }
QPushButton:disabled             { background-color: #404040; color: #808080; border-color: #505050; }
QComboBox                        { background-color: #454545; color: #ffffff; border: 1px solid #606060; border-radius: 3px; padding: 2px 8px; min-height: 22px; }
QComboBox::drop-down             { border: none; width: 20px; }
QComboBox::down-arrow            { width: 10px; height: 10px; }
QComboBox QAbstractItemView      { background-color: #353535; color: #ffffff; selection-background-color: #2a82da; border: 1px solid #555555; }
QLabel                           { background-color: transparent; color: #ffffff; }
QCheckBox                        { color: #ffffff; background-color: transparent; }
QCheckBox::indicator             { width: 14px; height: 14px; border: 1px solid #606060; background-color: #353535; border-radius: 2px; }
QCheckBox::indicator:checked     { background-color: #2a82da; border-color: #2a82da; }
QSpinBox, QDoubleSpinBox         { background-color: #191919; color: #ffffff; border: 1px solid #555555; border-radius: 3px; padding: 2px 6px; }
QGroupBox                        { color: #ffffff; border: 1px solid #555555; border-radius: 4px; margin-top: 8px; padding-top: 4px; }
QGroupBox::title                 { subcontrol-origin: margin; left: 8px; }
QHeaderView::section             { background-color: #404040; color: #ffffff; border: 1px solid #555555; padding: 2px 4px; }
QTreeWidget, QTreeView           { background-color: #252525; color: #ffffff; border: none; outline: 0; }
QTreeWidget::item:hover,
QTreeView::item:hover            { background-color: #3a4a6a; }
QTreeWidget::item:selected,
QTreeView::item:selected         { background-color: #2a82da; color: #ffffff; }
QAbstractItemView                { background-color: #252525; color: #ffffff; alternate-background-color: #2d2d2d; }
QTabWidget::pane                 { border: 1px solid #555555; }
QTabBar::tab                     { background-color: #454545; color: #ffffff; border: 1px solid #555555; padding: 4px 8px; }
QTabBar::tab:selected            { background-color: #353535; border-bottom: none; }
QToolTip                         { background-color: #2b2b2b; color: #ffffff; border: 1px solid #767676; padding: 1px; }
QPushButton[compact="true"]      { padding: 0px; min-height: 0px; min-width: 0px; }
QPushButton[pathButton="true"]   { padding: 0px; min-height: 0px; min-width: 0px; background-color: transparent; border: 1px solid transparent; border-radius: 3px; }
QPushButton[pathButton="true"]:hover   { background-color: rgba(255, 255, 255, 0.08); border-color: #666666; }
QPushButton[pathButton="true"]:pressed { background-color: rgba(255, 255, 255, 0.14); border-color: #777777; }
"""

_LIGHT_STYLESHEET = """
QWidget                          { color: #1a1a1a; }
QMainWindow, QDialog             { background-color: #f2f2f2; }
QDockWidget                      { background-color: #f2f2f2; color: #1a1a1a; }
QDockWidget::title               { background-color: #e0e0e0; padding: 4px; }
QToolBar                         { background-color: #f2f2f2; border: none; spacing: 2px; }
QToolBar QToolButton             { background-color: transparent; color: #1a1a1a; padding: 3px 6px; border: none; border-radius: 3px; }
QToolBar QToolButton:hover       { background-color: #dcdcdc; }
QToolBar QToolButton:pressed     { background-color: #c8c8c8; }
QMenuBar                         { background-color: #f2f2f2; color: #1a1a1a; }
QMenuBar::item:selected          { background-color: #2a82da; color: #ffffff; }
QMenu                            { background-color: #f2f2f2; color: #1a1a1a; border: 1px solid #cccccc; }
QMenu::item:selected             { background-color: #2a82da; color: #ffffff; }
QMenu::separator                 { background-color: #cccccc; height: 1px; }
QStatusBar                       { background-color: #f2f2f2; color: #1a1a1a; }
QSplitter::handle                { background-color: #cccccc; }
QScrollBar:vertical              { background-color: #f0f0f0; width: 12px; border: none; margin: 0; }
QScrollBar::handle:vertical      { background-color: #b0b0b0; border-radius: 6px; min-height: 20px; margin: 2px; }
QScrollBar::handle:vertical:hover{ background-color: #909090; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical    { height: 0; border: none; }
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical    { background: none; }
QScrollBar:horizontal            { background-color: #f0f0f0; height: 12px; border: none; margin: 0; }
QScrollBar::handle:horizontal    { background-color: #b0b0b0; border-radius: 6px; min-width: 20px; margin: 2px; }
QScrollBar::handle:horizontal:hover{ background-color: #909090; }
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal  { width: 0; border: none; }
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal  { background: none; }
QLineEdit                        { background-color: #ffffff; color: #1a1a1a; border: 1px solid #cccccc; border-radius: 3px; padding: 2px 6px; selection-background-color: #2a82da; selection-color: #ffffff; }
QTextEdit, QPlainTextEdit        { background-color: #ffffff; color: #1a1a1a; border: 1px solid #cccccc; selection-background-color: #2a82da; selection-color: #ffffff; }
QPushButton                      { background-color: #e0e0e0; color: #1a1a1a; border: 1px solid #cccccc; border-radius: 3px; padding: 4px 12px; min-height: 22px; }
QPushButton:hover                { background-color: #d0d0d0; }
QPushButton:pressed              { background-color: #c0c0c0; }
QPushButton:disabled             { background-color: #e8e8e8; color: #a0a0a0; border-color: #d0d0d0; }
QComboBox                        { background-color: #ffffff; color: #1a1a1a; border: 1px solid #cccccc; border-radius: 3px; padding: 2px 8px; min-height: 22px; }
QComboBox::drop-down             { border: none; width: 20px; }
QComboBox::down-arrow            { width: 10px; height: 10px; }
QComboBox QAbstractItemView      { background-color: #ffffff; color: #1a1a1a; selection-background-color: #2a82da; selection-color: #ffffff; border: 1px solid #cccccc; }
QLabel                           { background-color: transparent; color: #1a1a1a; }
QCheckBox                        { color: #1a1a1a; background-color: transparent; }
QCheckBox::indicator             { width: 14px; height: 14px; border: 1px solid #aaaaaa; background-color: #ffffff; border-radius: 2px; }
QCheckBox::indicator:checked     { background-color: #2a82da; border-color: #2a82da; }
QSpinBox, QDoubleSpinBox         { background-color: #ffffff; color: #1a1a1a; border: 1px solid #cccccc; border-radius: 3px; padding: 2px 6px; }
QGroupBox                        { color: #1a1a1a; border: 1px solid #cccccc; border-radius: 4px; margin-top: 8px; padding-top: 4px; }
QGroupBox::title                 { subcontrol-origin: margin; left: 8px; }
QHeaderView::section             { background-color: #e0e0e0; color: #1a1a1a; border: 1px solid #cccccc; padding: 2px 4px; }
QTreeWidget, QTreeView           { background-color: #ffffff; color: #1a1a1a; border: none; outline: 0; }
QTreeWidget::item:hover,
QTreeView::item:hover            { background-color: #dde8f5; }
QTreeWidget::item:selected,
QTreeView::item:selected         { background-color: #2a82da; color: #ffffff; }
QAbstractItemView                { background-color: #ffffff; color: #1a1a1a; alternate-background-color: #f5f5f5; }
QTabWidget::pane                 { border: 1px solid #cccccc; }
QTabBar::tab                     { background-color: #e0e0e0; color: #1a1a1a; border: 1px solid #cccccc; padding: 4px 8px; }
QTabBar::tab:selected            { background-color: #f2f2f2; border-bottom: none; }
QToolTip                         { background-color: #fffac0; color: #1a1a1a; border: 1px solid #bbbbbb; padding: 1px; }
QPushButton[compact="true"]      { padding: 0px; min-height: 0px; min-width: 0px; }
QPushButton[pathButton="true"]   { padding: 0px; min-height: 0px; min-width: 0px; background-color: transparent; border: 1px solid transparent; border-radius: 3px; }
QPushButton[pathButton="true"]:hover   { background-color: rgba(0, 0, 0, 0.06); border-color: #bbbbbb; }
QPushButton[pathButton="true"]:pressed { background-color: rgba(0, 0, 0, 0.10); border-color: #aaaaaa; }
"""


def _make_dark_palette() -> QtGui.QPalette:
    p = QtGui.QPalette()
    W = QtCore.Qt.GlobalColor.white
    p.setColor(QtGui.QPalette.ColorRole.Window,          QtGui.QColor(53, 53, 53))
    p.setColor(QtGui.QPalette.ColorRole.WindowText,      W)
    p.setColor(QtGui.QPalette.ColorRole.Base,            QtGui.QColor(25, 25, 25))
    p.setColor(QtGui.QPalette.ColorRole.AlternateBase,   QtGui.QColor(53, 53, 53))
    p.setColor(QtGui.QPalette.ColorRole.ToolTipBase,     W)
    p.setColor(QtGui.QPalette.ColorRole.ToolTipText,     W)
    p.setColor(QtGui.QPalette.ColorRole.Text,            W)
    p.setColor(QtGui.QPalette.ColorRole.Button,          QtGui.QColor(53, 53, 53))
    p.setColor(QtGui.QPalette.ColorRole.ButtonText,      W)
    p.setColor(QtGui.QPalette.ColorRole.BrightText,      QtCore.Qt.GlobalColor.red)
    p.setColor(QtGui.QPalette.ColorRole.Link,            QtGui.QColor(42, 130, 218))
    p.setColor(QtGui.QPalette.ColorRole.Highlight,       QtGui.QColor(42, 130, 218))
    p.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black)
    return p


def _make_light_palette() -> QtGui.QPalette:
    p = QtGui.QPalette()
    D = QtGui.QColor(26, 26, 26)
    p.setColor(QtGui.QPalette.ColorRole.Window,          QtGui.QColor(242, 242, 242))
    p.setColor(QtGui.QPalette.ColorRole.WindowText,      D)
    p.setColor(QtGui.QPalette.ColorRole.Base,            QtGui.QColor(255, 255, 255))
    p.setColor(QtGui.QPalette.ColorRole.AlternateBase,   QtGui.QColor(230, 230, 230))
    p.setColor(QtGui.QPalette.ColorRole.ToolTipBase,     QtGui.QColor(255, 250, 200))
    p.setColor(QtGui.QPalette.ColorRole.ToolTipText,     D)
    p.setColor(QtGui.QPalette.ColorRole.Text,            D)
    p.setColor(QtGui.QPalette.ColorRole.Button,          QtGui.QColor(224, 224, 224))
    p.setColor(QtGui.QPalette.ColorRole.ButtonText,      D)
    p.setColor(QtGui.QPalette.ColorRole.BrightText,      QtGui.QColor(200, 0, 0))
    p.setColor(QtGui.QPalette.ColorRole.Link,            QtGui.QColor(0, 102, 204))
    p.setColor(QtGui.QPalette.ColorRole.Highlight,       QtGui.QColor(42, 130, 218))
    p.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
    return p


def _lerp_color(c1: QtGui.QColor, c2: QtGui.QColor, t: float) -> QtGui.QColor:
    return QtGui.QColor(
        round(c1.red()   + (c2.red()   - c1.red())   * t),
        round(c1.green() + (c2.green() - c1.green()) * t),
        round(c1.blue()  + (c2.blue()  - c1.blue())  * t),
    )


def _smoothstep(t: float) -> float:
    """Ease-in-out so the transition accelerates then decelerates."""
    return t * t * (3.0 - 2.0 * t)


def _lerp_palette(p1: QtGui.QPalette, p2: QtGui.QPalette, t: float) -> QtGui.QPalette:
    out = QtGui.QPalette()
    n_roles = QtGui.QPalette.ColorRole.NColorRoles.value
    groups  = (QtGui.QPalette.ColorGroup.Active,
               QtGui.QPalette.ColorGroup.Inactive,
               QtGui.QPalette.ColorGroup.Disabled)
    for role_val in range(n_roles):
        role = QtGui.QPalette.ColorRole(role_val)
        for group in groups:
            c = _lerp_color(p1.color(group, role), p2.color(group, role), t)
            out.setColor(group, role, c)
    return out


class ThemeManager(QtCore.QObject):
    """Manages light/dark theme switching with a smooth animated transition."""

    theme_changed = QtCore.Signal(bool)   # emits True=dark, False=light

    _DURATION_MS = 260
    _FPS         = 30

    def __init__(self, app: QtWidgets.QApplication):
        super().__init__(app)
        self._app     = app
        self._is_dark = True
        self._dark_pal  = _make_dark_palette()
        self._light_pal = _make_light_palette()
        self._current_pal = self._dark_pal   # always tracks the live palette we last applied
        self._from_pal: QtGui.QPalette | None = None
        self._to_pal:   QtGui.QPalette | None = None
        self._t   = 0.0
        self._dt  = 1.0 / (self._DURATION_MS / 1000.0 * self._FPS)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1000 // self._FPS)
        self._timer.timeout.connect(self._step)
        # Apply initial dark palette
        self._app.setPalette(self._dark_pal)

    @property
    def is_dark(self) -> bool:
        return self._is_dark

    @property
    def current_palette(self) -> QtGui.QPalette:
        return self._current_pal

    def toggle(self):
        self._timer.stop()
        self._is_dark  = not self._is_dark
        self._from_pal = self._current_pal   # use our tracked palette, not app.palette()
        self._to_pal   = self._dark_pal if self._is_dark else self._light_pal
        self._t        = 0.0
        self._timer.start()
        self.theme_changed.emit(self._is_dark)

    def _step(self):
        self._t = min(self._t + self._dt, 1.0)
        self._current_pal = _lerp_palette(self._from_pal, self._to_pal, _smoothstep(self._t))
        self._app.setPalette(self._current_pal)
        if self._t >= 1.0:
            self._timer.stop()


class NodeExecutionWindow(QtWidgets.QMainWindow):
    _RECENT_MAX = 10
    _AUTOSAVE_INTERVAL_MS = 120000

    def __init__(self, theme_manager: 'ThemeManager | None' = None):
        super(NodeExecutionWindow, self).__init__()
        self.theme_manager = theme_manager or ThemeManager(QtWidgets.QApplication.instance())
        self.settings = QtCore.QSettings("Synapse", "Synapse")
        self._current_workflow_path = ""
        self._recent_workflows: list[str] = []
        self._manual_dirty = False
        self._undo_dirty = False
        self._autosave_path = self._get_recovery_file_path()
        self._load_recent_workflows()
        self._pipe_settings = self._load_pipe_settings()
        self._per_pipe_settings: dict = {}   # keyed by _pipe_key(); per-pipe overrides
        # self.setGeometry(0, 0, 1400, 900)

        # Initialize NodeGraph
        # from NodeGraphQt.constants import LayoutDirectionEnum
        self.graph = NodeGraph()
        # self.graph.set_layout_direction(LayoutDirectionEnum.VERTICAL.value)
        
        # Clear default registered nodes (like Backdrop) if you want a clean Node Explorer
        self.graph.node_factory.clear_registered_nodes()
        self._register_core_nodes()

        # Load plugins — must happen before NodesTreeWidget is constructed so
        # that plugin nodes appear in the Node Explorer tree automatically.
        from .plugin_loader import load_plugins, get_plugin_dir
        self._plugin_results = load_plugins(self.graph)
        self._plugin_dir = get_plugin_dir()

        # Context Menu: Add "Clear Cache" and "Disable/Enable" for nodes
        nodes_menu = self.graph.get_context_menu('nodes')
        nodes_menu.add_command('Clear Node Cache', clear_node_cache, node_class=BaseExecutionNode)
        nodes_menu.add_command('Disable / Enable Node', toggle_node_disable, node_class=BaseExecutionNode)
        
        # Set central widget
        self.setCentralWidget(self.graph.widget)

        # Add properties bin
        self.properties_bin = PropertiesBinWidget(node_graph=self.graph)
        self.properties_bin.setWindowFlags(QtCore.Qt.WindowType.Tool)
        
        # Setup dock widget for properties
        self.dockWidgetProperties = QtWidgets.QDockWidget("Properties")
        self.dockWidgetProperties.setWidget(self.properties_bin)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dockWidgetProperties)
        self.dockWidgetProperties.hide()
        
        # Setup node tree (palette)
        self.nodes_tree = NodesTreeWidget(node_graph=self.graph)
        
        # Set the display order of categories
        self.nodes_tree.set_category_order([
            'nodes.io',
            'nodes.display',
            'nodes.Collection',
            'nodes.utility',
            'nodes.dataframe',
            'nodes.plotting',
            'nodes.analysis',
            'nodes.image_process',
            'nodes.data',
            'plugins.Plugins',
            'plugins.Plugins.Segmentation',
            'plugins.Plugins.VideoAnalysis',
            'plugins.Plugins.confocal',
            'plugins.Plugins.filopodia',
        ])
        
        # Define human-readable labels for categories
        self.nodes_tree.set_category_label('nodes.dataframe', tr('Table Processing'))
        self.nodes_tree.set_category_label('nodes.analysis', tr('Statistical Analysis'))
        self.nodes_tree.set_category_label('nodes.plotting', tr('Visualization'))
        self.nodes_tree.set_category_label('plugins.Plugins', tr('Plugins'))
        self.nodes_tree.set_category_label('plugins.Plugins.Segmentation', tr('Segmentation'))
        self.nodes_tree.set_category_label('plugins.Plugins.VideoAnalysis', tr('Video Analysis'))
        self.nodes_tree.set_category_label('plugins.Plugins.confocal', tr('Confocal Analysis'))
        self.nodes_tree.set_category_label('nodes.io', tr('Input / Output'))
        self.nodes_tree.set_category_label('nodes.utility', tr('Common Utilities'))
        self.nodes_tree.set_category_label('nodes.Collection', tr('Collection'))
        self.nodes_tree.set_category_label('nodes.display', tr('Display'))
        self.nodes_tree.set_category_label('nodes.image_process', tr('Image Processing'))
        self.nodes_tree.set_category_label('nodes.image_process.color', tr('Color'))
        self.nodes_tree.set_category_label('nodes.image_process.adjust', tr('Adjust/Contrast'))
        self.nodes_tree.set_category_label('nodes.image_process.filter', tr('Filters'))
        self.nodes_tree.set_category_label('nodes.image_process.threshold', tr('Thresholding'))
        self.nodes_tree.set_category_label('nodes.image_process.morphology', tr('Morphology'))
        self.nodes_tree.set_category_label('nodes.image_process.math', tr('Math'))
        self.nodes_tree.set_category_label('plugins.Plugins.filopodia', tr('Filopodia Analysis'))
        self.nodes_tree.set_category_label('nodes.image_process.geometry', tr('Geometry'))
        self.nodes_tree.set_category_label('nodes.image_process.measure', tr('Measure'))
        self.nodes_tree.set_category_label('nodes.data', tr('Misc'))
        
        self.dockWidgetNodes = QtWidgets.QDockWidget(tr("Node Explorer"))
        self.dockWidgetNodes.setWidget(self.nodes_tree)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dockWidgetNodes)

        self._batch_progress = QtWidgets.QProgressBar(self)
        self._batch_progress.setRange(0, 100)
        self._batch_progress.setValue(0)
        self._batch_progress.setFixedWidth(180)
        self._batch_progress.hide()
        self.statusBar().addPermanentWidget(self._batch_progress)

        self._batch_meta_label = QtWidgets.QLabel("", self)
        self._batch_meta_label.hide()
        self.statusBar().addPermanentWidget(self._batch_meta_label)

        self._is_batch_running = False
        self._batch_total = 0
        self._batch_fail_count = 0
        self._batch_failures: list[tuple[str, str]] = []

        # AI Assistant dock (hidden by default; toggle via View menu)
        from .llm_assistant import LLMAssistantPanel
        self.llm_panel = LLMAssistantPanel(self.graph)
        self.dockWidgetLLM = QtWidgets.QDockWidget(tr("AI Assistant"))
        self.dockWidgetLLM.setWidget(self.llm_panel)
        self.dockWidgetLLM.setMinimumWidth(280)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dockWidgetLLM)
        self.dockWidgetLLM.hide()

        # ── Node Help Dock ──
        self._help_browser = QtWidgets.QTextBrowser()
        self._help_browser.setOpenExternalLinks(False)
        self._help_browser.setReadOnly(True)
        self._help_browser.setPlaceholderText(tr("Select a node to see its documentation."))
        self._help_browser.setHtml(
            "<p style='color:gray;'>Select a node to see its documentation.</p>")
        self.dockWidgetHelp = QtWidgets.QDockWidget(tr("Node Help"))
        self.dockWidgetHelp.setWidget(self._help_browser)
        self.dockWidgetHelp.setMinimumWidth(240)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
                           self.dockWidgetHelp)
        self.tabifyDockWidget(self.dockWidgetLLM, self.dockWidgetHelp)
        self.dockWidgetHelp.raise_()

        # ── Execution Order Dock ──
        self._exec_order_widget = QtWidgets.QWidget()
        exec_lay = QtWidgets.QVBoxLayout(self._exec_order_widget)
        exec_lay.setContentsMargins(4, 4, 4, 4)
        exec_lay.setSpacing(2)

        self._exec_order_list = QtWidgets.QListWidget()
        self._exec_order_list.setStyleSheet(
            "QListWidget { font-size: 11px; }"
            "QListWidget::item { padding: 3px 6px; }"
        )
        self._exec_order_list.itemClicked.connect(self._on_exec_order_item_clicked)

        btn_row = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton(tr("Refresh"))
        btn_refresh.clicked.connect(self._refresh_execution_order)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()

        exec_lay.addWidget(self._exec_order_list, 1)
        exec_lay.addLayout(btn_row)

        self.dockWidgetExecOrder = QtWidgets.QDockWidget(tr("Execution Order"))
        self.dockWidgetExecOrder.setWidget(self._exec_order_widget)
        self.dockWidgetExecOrder.setMinimumWidth(200)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
                           self.dockWidgetExecOrder)
        self.dockWidgetExecOrder.hide()

        # Update tooltip stylesheet to match current theme, and keep in sync on toggle
        self._update_theme_stylesheet()
        self.theme_manager.theme_changed.connect(lambda _: self._update_theme_stylesheet())

        # Show properties bin on double click
        self.graph.node_double_clicked.connect(self.display_properties_bin)
        # Update help panel when node selection changes
        self.graph.node_selection_changed.connect(self._on_node_selection_for_help)
        self.graph.node_created.connect(lambda _n: self._mark_manual_dirty())
        self.graph.node_created.connect(lambda _n: self._update_order_badges())
        self.graph.nodes_deleted.connect(self._on_nodes_deleted)
        self.graph.nodes_deleted.connect(lambda _ids: self._update_order_badges())
        self.graph.port_connected.connect(lambda _a, _b: self._mark_manual_dirty())
        self.graph.port_connected.connect(self._on_port_connected_style)
        self.graph.port_connected.connect(lambda _a, _b: self._update_order_badges())
        self.graph.port_disconnected.connect(lambda _a, _b: self._mark_manual_dirty())
        self.graph.port_disconnected.connect(lambda _a, _b: self._update_order_badges())
        self.graph.property_changed.connect(lambda _n, _p, _v: self._mark_manual_dirty())
        self.graph.undo_stack().indexChanged.connect(self._on_undo_stack_changed)

        self.worker_thread = None
        self.worker = None

        # Handle macOS Backspace for node deletion + pipe right-click context menu
        self.graph.viewer().installEventFilter(self)
        # ContextMenu events go to the viewport widget, not the view — intercept both
        self.graph.viewer().viewport().installEventFilter(self)
        # Re-apply per-pipe widths after any selection change (pipe.reset() hardcodes width=2)
        self.graph.viewer().scene().selectionChanged.connect(
            lambda: QtCore.QTimer.singleShot(0, self._apply_pipe_style_to_all)
        )
        
        # Minimap overlay (bottom-right of the graph viewer)
        self._minimap = MinimapWidget(self.graph.viewer(), parent=self.graph.widget)
        self._minimap.set_visible(False)
        self.graph.widget.installEventFilter(self._minimap)

        # Thread-safe UI Dispatcher
        from .custom_nodes import NODE_SIGNALS
        NODE_SIGNALS.progress_updated.connect(lambda nid, val: self._dispatch_ui_signal(nid, 'progress', val))
        NODE_SIGNALS.status_updated.connect(lambda nid, val: self._dispatch_ui_signal(nid, 'status', val))
        NODE_SIGNALS.display_requested.connect(lambda nid, val: self._dispatch_ui_signal(nid, 'display', val))

        self._apply_pipe_layout()     # apply saved layout style (no pipes exist yet, but sets the default)
        self.setup_toolbar()
        self.setup_menus()

        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setInterval(self._AUTOSAVE_INTERVAL_MS)
        self._autosave_timer.timeout.connect(self._autosave_if_needed)
        self._autosave_timer.start()
        
        # Create an initial node for demonstration
        # n1 = self.graph.create_node('custom.nodes.UniversalDataNode', name='Input Data', pos=[-300, 0])
        # n1.set_property('code', 'output = [10, 20, 30]')

        self._mark_clean()
        self._maybe_recover_session()
        self.statusBar().showMessage(tr("Ready"))
        self._update_window_title()
        self.showMaximized()

    def _get_recovery_file_path(self) -> str:
        base_dir = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.AppDataLocation
        )
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), ".synapse")
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, "recovery_autosave.json")

    def _workflow_dialog_start_dir(self) -> str:
        if self._current_workflow_path:
            return os.path.dirname(self._current_workflow_path)
        return "workflows"

    def _load_recent_workflows(self):
        values = self.settings.value("recent_workflows", [], type=list) or []
        self._recent_workflows = [
            p for p in values if isinstance(p, str) and os.path.isfile(p)
        ][:self._RECENT_MAX]

    def _save_recent_workflows(self):
        self.settings.setValue("recent_workflows", self._recent_workflows)
        self.settings.setValue("last_workflow", self._current_workflow_path or "")

    def _rebuild_recent_menu(self):
        if not hasattr(self, "_recent_menu"):
            return
        self._recent_menu.clear()
        if not self._recent_workflows:
            action = self._recent_menu.addAction(tr("No recent workflows"))
            action.setEnabled(False)
            return
        for path in self._recent_workflows:
            action = self._recent_menu.addAction(os.path.basename(path))
            action.setToolTip(path)
            action.triggered.connect(
                lambda checked=False, p=path: self._load_example(p, prompt_unsaved=True)
            )
        self._recent_menu.addSeparator()
        clear_action = self._recent_menu.addAction(tr("Clear Recent"))
        clear_action.triggered.connect(self._clear_recent_workflows)

    def _clear_recent_workflows(self):
        self._recent_workflows = []
        self._save_recent_workflows()
        self._rebuild_recent_menu()

    def _record_recent_workflow(self, file_path: str):
        file_path = os.path.abspath(file_path)
        self._recent_workflows = [p for p in self._recent_workflows if p != file_path]
        self._recent_workflows.insert(0, file_path)
        self._recent_workflows = self._recent_workflows[:self._RECENT_MAX]
        self._save_recent_workflows()
        self._rebuild_recent_menu()

    def _reopen_last_workflow(self):
        last_path = self.settings.value("last_workflow", "", type=str) or ""
        if not last_path or not os.path.isfile(last_path):
            QtWidgets.QMessageBox.information(self, tr("Reopen Last"), tr("No previous workflow found."))
            return
        self._load_example(last_path, prompt_unsaved=True)

    def _update_window_title(self):
        name = os.path.basename(self._current_workflow_path) if self._current_workflow_path else "Untitled"
        marker = "*" if (self._manual_dirty or self._undo_dirty) else ""
        self.setWindowTitle(f"Synapse - {name}{marker}")

    def _refresh_dirty_state(self):
        self._update_window_title()

    def _mark_manual_dirty(self):
        self._manual_dirty = True
        self._refresh_dirty_state()

    def _on_nodes_deleted(self, _ids):
        """Mark workflow dirty and purge stale node graphics left in scene."""
        self._mark_manual_dirty()
        QtCore.QTimer.singleShot(0, self._purge_orphan_node_items)

    def _purge_orphan_node_items(self):
        """
        Remove orphan NodeGraphQt graphics items that can remain in rare
        thread/contention cases (visible but not selectable "ghost" nodes).
        """
        try:
            from NodeGraphQt.qgraphics.node_abstract import AbstractNodeItem
            scene = self.graph.viewer().scene()
            valid_views = {
                n.view for n in self.graph.all_nodes()
                if getattr(n, 'view', None) is not None
            }
            removed = 0
            for item in list(scene.items()):
                if (
                    isinstance(item, AbstractNodeItem)
                    and item.parentItem() is None
                    and item not in valid_views
                ):
                    scene.removeItem(item)
                    removed += 1
            if removed:
                scene.update()
        except Exception:
            # Non-fatal guard: cleanup should never break regular workflows.
            pass

    def _on_undo_stack_changed(self, _index: int):
        try:
            self._undo_dirty = not self.graph.undo_stack().isClean()
        except RuntimeError:
            return
        self._refresh_dirty_state()

    def _mark_clean(self):
        self._manual_dirty = False
        self._undo_dirty = False
        self.graph.undo_stack().setClean()
        self._refresh_dirty_state()

    def _clear_recovery_file(self):
        try:
            if os.path.exists(self._autosave_path):
                os.remove(self._autosave_path)
        except OSError:
            pass

    def _autosave_if_needed(self):
        if not (self._manual_dirty or self._undo_dirty):
            return
        self._write_recovery_snapshot()

    def _write_recovery_snapshot(self) -> bool:
        if not self.graph.all_nodes():
            return False
        payload = {
            "timestamp": int(time.time()),
            "source_path": self._current_workflow_path,
            "layout": self.graph.serialize_session(),
        }
        tmp_path = f"{self._autosave_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_path, self._autosave_path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False

    def _autosave_now(self):
        ok = self._write_recovery_snapshot()
        if ok:
            self.statusBar().showMessage(tr("Autosave snapshot created."), 3000)
        else:
            self.statusBar().showMessage(tr("Autosave skipped (graph is empty)."), 3000)

    def _maybe_recover_session(self):
        if not os.path.isfile(self._autosave_path):
            return
        try:
            with open(self._autosave_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            self._clear_recovery_file()
            return

        ts = payload.get("timestamp", 0)
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "unknown time"
        reply = QtWidgets.QMessageBox.question(
            self,
            tr("Recover Autosave"),
            f"An autosaved workflow from {when} was found.\nRecover it now?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            self._clear_recovery_file()
            return

        try:
            layout = payload.get("layout") or {}
            self.graph.deserialize_session(layout, clear_session=True, clear_undo_stack=True)
            src = payload.get("source_path", "") or ""
            self._current_workflow_path = src if isinstance(src, str) else ""
            self._manual_dirty = True
            self._undo_dirty = False
            self._refresh_dirty_state()
            self.statusBar().showMessage("Recovered autosaved workflow.", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("Recovery Failed"), str(exc))
            self._clear_recovery_file()

    def _confirm_discard_unsaved(self, action_text: str) -> bool:
        if not (self._manual_dirty or self._undo_dirty):
            return True
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(tr("Unsaved Changes"))
        box.setText(f"You have unsaved changes.\nDo you want to save before {action_text}?")
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        save_btn = box.addButton(tr("Save"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton(tr("Discard"), QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked == save_btn:
            return self._save_workflow()
        if clicked == discard_btn:
            self._clear_recovery_file()
            return True
        return False

    def _register_core_nodes(self):
        """Register core Synapse nodes into the node factory.
        Domain-specific nodes (image processing, stats, plotting, etc.)
        are loaded as plugins via load_plugins() instead.
        """
        for node_cls in (
            # I/O & batch
            FileReadNode, FolderIteratorNode, VideoIteratorNode,
            ImageReadNode, SaveNode,
            BatchAccumulatorNode, BatchGateNode,
            # Display
            DisplayNode, DataTableCellNode, DataFigureCellNode,
            ImageCellNode,
            # Utility
            UniversalDataNode, PathModifierNode,
            CollectNode, SelectCollectionNode, PopCollectionNode,
    SplitCollectionNode, SaveCollectionNode,
    RenameCollectionNode, CollectionInfoNode, FilterCollectionNode, MapNamesNode,
            # DataFrame operations — moved to data_processing plugin
        ):
            self.graph.register_node(node_cls)

    def _set_batch_progress_visible(self, visible: bool):
        self._batch_progress.setVisible(visible)
        self._batch_meta_label.setVisible(visible)
        if not visible:
            self._batch_progress.setValue(0)
            self._batch_meta_label.setText("")

    def _update_batch_progress_ui(self, current: int, total: int):
        total = max(total, 1)
        percent = int((current / total) * 100)
        self._batch_progress.setValue(min(max(percent, 0), 100))
        elapsed = max(time.perf_counter() - self._batch_start, 0.0)
        eta_text = "ETA --:--"
        if current > 0:
            remaining = max(total - current, 0)
            eta_seconds = int((elapsed / current) * remaining)
            mins, secs = divmod(max(eta_seconds, 0), 60)
            eta_text = f"ETA {mins:02d}:{secs:02d}"
        self._batch_meta_label.setText(
            f"{current}/{total}  |  {eta_text}  |  Failed {self._batch_fail_count}"
        )

    def _reload_plugins(self):
        """Reload plugin files and refresh node explorer without restarting."""
        from .plugin_loader import load_plugins, get_plugin_dir
        self._plugin_dir = get_plugin_dir()
        self.graph.node_factory.clear_registered_nodes()
        self._register_core_nodes()
        self._plugin_results = load_plugins(self.graph)
        self.nodes_tree.update()
        ok_count = sum(1 for r in self._plugin_results if r.get('nodes') and not r.get('error'))
        err_count = sum(1 for r in self._plugin_results if r.get('error'))
        self.statusBar().showMessage(
            f"Plugins reloaded. Active files: {ok_count}, errors: {err_count}.", 6000
        )

    def eventFilter(self, obj, event):
        viewer = self.graph.viewer()
        # ContextMenu events arrive at the viewport widget, intercept there
        if (obj is viewer.viewport() or obj is viewer) and \
                event.type() == QtCore.QEvent.Type.ContextMenu:
            pos  = viewer.mapToScene(event.pos())
            from NodeGraphQt.qgraphics.pipe import PipeItem
            near = QtCore.QRectF(pos.x() - 6, pos.y() - 6, 12, 12)
            pipes = [i for i in viewer.scene().items(near)
                     if isinstance(i, PipeItem)
                     and i not in (viewer._LIVE_PIPE, viewer._SLICER_PIPE)]
            if pipes:
                self._show_pipe_context_menu(pipes[0], event.globalPos())
                return True   # consume — don't open the normal graph context menu

        if obj is viewer and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
                focus_item = self.graph.scene().focusItem()
                if isinstance(focus_item, (QtWidgets.QGraphicsProxyWidget,
                                           QtWidgets.QGraphicsTextItem)):
                    selected_views = {n.view for n in self.graph.selected_nodes()}
                    item = focus_item
                    while item is not None:
                        if item in selected_views:
                            return False   # editing inside a selected node
                        item = item.parentItem()
                    # Focus is inside an unselected node's widget.
                    # Only treat as stale when there are selected nodes to delete;
                    # otherwise the user is just editing that field — pass through.
                    if not selected_views:
                        return False
                    self.graph.scene().clearFocus()

                selected = self.graph.selected_nodes()
                if selected:
                    self.graph.delete_nodes(selected)
                    return True
        return super(NodeExecutionWindow, self).eventFilter(obj, event)
    
    def setup_toolbar(self):
        toolbar = self.addToolBar(tr("Execution"))

        self.btn_run = QtGui.QAction(tr("Run Graph"), self)
        self.btn_run.setShortcut("Ctrl+W")
        self.btn_run.setToolTip(tr("Run Graph") + "  [Ctrl+W]")
        self.btn_run.triggered.connect(self.execute_graph)
        toolbar.addAction(self.btn_run)

        self.btn_batch = QtGui.QAction(tr("Batch Run"), self)
        self.btn_batch.setShortcut("Ctrl+B")
        self.btn_batch.setToolTip(tr("Batch Run") + "  [Ctrl+B]")
        self.btn_batch.triggered.connect(self.execute_batch)
        toolbar.addAction(self.btn_batch)

        self.btn_stop = QtGui.QAction(tr("Stop"), self)
        self.btn_stop.triggered.connect(self.stop_execution)
        self.btn_stop.setEnabled(False)
        # Use a built-in icon if available or just text
        self.btn_stop.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserStop))
        toolbar.addAction(self.btn_stop)

        toolbar.addSeparator()

        self.btn_clear_selected = QtGui.QAction(tr("Clear Selected Caches"), self)
        self.btn_clear_selected.triggered.connect(self.clear_selected_caches)
        toolbar.addAction(self.btn_clear_selected)

        self.btn_clear_all = QtGui.QAction(tr("Clear All Caches"), self)
        self.btn_clear_all.triggered.connect(self.clear_all_caches)
        toolbar.addAction(self.btn_clear_all)

        toolbar.addSeparator()
        self._theme_action = QtGui.QAction(tr("Light Mode"), self)
        self._theme_action.setToolTip("Toggle light / dark theme")
        self._theme_action.triggered.connect(self._toggle_theme)
        toolbar.addAction(self._theme_action)

    def clear_selected_caches(self):
        """Toolbar action to clear memory for only the selected nodes."""
        selected = self.graph.selected_nodes()
        if not selected:
            self.statusBar().showMessage(tr("No nodes selected to clear."), 3000)
            return
        for node in selected:
            if hasattr(node, 'clear_cache'):
                node.clear_cache()
        self.statusBar().showMessage(f"Cleared cache for {len(selected)} nodes.", 3000)

    def clear_all_caches(self):
        """Toolbar action to clear memory for every node in the graph."""
        all_nodes = self.graph.all_nodes()
        for node in all_nodes:
            if hasattr(node, 'clear_cache'):
                node.clear_cache()
        self.statusBar().showMessage(tr("All node caches cleared."), 3000)

    def _toggle_theme(self):
        self.theme_manager.toggle()
        is_dark = self.theme_manager.is_dark
        self._theme_action.setText(tr("Light Mode") if is_dark else tr("Dark Mode"))
        # Keep the graph canvas consistent with the panel theme
        scene = self.graph.scene()
        if is_dark:
            scene.background_color = (35, 35, 35)
            scene.grid_color       = (45, 45, 45)
        else:
            scene.background_color = (205, 215, 220)
            scene.grid_color       = (182, 192, 197)
        scene.update()
        # Explicitly push the final palette to dock widgets after animation ends.
        # Qt palette inheritance can be interrupted by any widget that has an explicit
        # palette or stylesheet set — this guarantees the change always propagates.
        QtCore.QTimer.singleShot(self.theme_manager._DURATION_MS + 40,
                                 self._force_palette_refresh)

    def _force_palette_refresh(self):
        """Push the current palette explicitly to all dock widget contents."""
        pal = self.theme_manager.current_palette
        for w in (self.dockWidgetNodes, self.dockWidgetProperties,
                  self.dockWidgetLLM, self.nodes_tree, self.properties_bin):
            w.setPalette(pal)
            w.update()

    def _update_theme_stylesheet(self):
        """Apply comprehensive stylesheet so all widget types reflect the current theme."""
        app = QtWidgets.QApplication.instance()
        app.setStyleSheet(_DARK_STYLESHEET if self.theme_manager.is_dark else _LIGHT_STYLESHEET)

    def setup_menus(self):
        """Setup the main window menus."""
        menubar = self.menuBar()
        
        # Edit Menu
        edit_menu = menubar.addMenu(tr("&Edit"))

        copy_action = QtGui.QAction(tr("&Copy Nodes"), self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self._copy_nodes)
        edit_menu.addAction(copy_action)

        paste_action = QtGui.QAction(tr("&Paste Nodes"), self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self._paste_nodes)
        edit_menu.addAction(paste_action)

        select_all_action = QtGui.QAction(tr("Select &All Nodes"), self)
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(self._select_all_nodes)
        edit_menu.addAction(select_all_action)

        focus_search_action = QtGui.QAction(tr("&Focus Node Search"), self)
        focus_search_action.setShortcut("Ctrl+F")
        focus_search_action.triggered.connect(self._focus_search_bar)
        edit_menu.addAction(focus_search_action)

        # Workflows Menu
        workflow_menu = menubar.addMenu(tr("&Workflows"))

        save_action = QtGui.QAction(tr("&Save Workflow..."), self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_workflow)
        workflow_menu.addAction(save_action)

        open_action = QtGui.QAction(tr("&Open Workflow..."), self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_workflow)
        workflow_menu.addAction(open_action)

        append_open_action = QtGui.QAction(tr("Append Workflow..."), self)
        append_open_action.triggered.connect(self._append_workflow)
        workflow_menu.addAction(append_open_action)

        workflow_menu.addSeparator()

        autosave_now_action = QtGui.QAction(tr("Autosave Now"), self)
        autosave_now_action.triggered.connect(self._autosave_now)
        workflow_menu.addAction(autosave_now_action)

        reopen_last_action = QtGui.QAction(tr("Reopen Last Workflow"), self)
        reopen_last_action.triggered.connect(self._reopen_last_workflow)
        workflow_menu.addAction(reopen_last_action)

        self._recent_menu = workflow_menu.addMenu(tr("Open Recent"))
        self._rebuild_recent_menu()

        workflow_menu.addSeparator()

        # Examples Submenu
        examples_menu = workflow_menu.addMenu(tr("&Examples"))
        self._populate_examples(examples_menu, is_append=False)

        # Append Examples Submenu
        append_examples_menu = workflow_menu.addMenu(tr("Append Example"))
        self._populate_examples(append_examples_menu, is_append=True)

        # View Menu
        view_menu = menubar.addMenu(tr("&View"))
        view_menu.addAction(self.dockWidgetProperties.toggleViewAction())
        view_menu.addAction(self.dockWidgetNodes.toggleViewAction())
        view_menu.addAction(self.dockWidgetLLM.toggleViewAction())
        view_menu.addAction(self.dockWidgetExecOrder.toggleViewAction())
        self._show_order_badges = False
        self._order_badge_items: list = []
        badge_action = QtGui.QAction(tr("Show Execution Order Badges"), self)
        badge_action.setCheckable(True)
        badge_action.toggled.connect(self._toggle_order_badges)
        view_menu.addAction(badge_action)

        minimap_action = QtGui.QAction(tr("Minimap"), self)
        minimap_action.setCheckable(True)
        minimap_action.setChecked(False)
        minimap_action.toggled.connect(self._minimap.set_visible)
        view_menu.addAction(minimap_action)

        view_menu.addSeparator()
        install_plugin_action = QtGui.QAction(tr("Install Plugin..."), self)
        install_plugin_action.triggered.connect(self._install_plugin)
        view_menu.addAction(install_plugin_action)
        reload_plugin_action = QtGui.QAction(tr("Reload Plugins"), self)
        reload_plugin_action.triggered.connect(self._reload_plugins)
        view_menu.addAction(reload_plugin_action)
        plugin_mgr_action = QtGui.QAction(tr("Plugin Manager..."), self)
        plugin_mgr_action.triggered.connect(self._show_plugin_manager)
        view_menu.addAction(plugin_mgr_action)

        # Pipe style
        pipe_style_action = QtGui.QAction(tr("Pipe Style..."), self)
        pipe_style_action.triggered.connect(self._show_pipe_style_dialog)
        view_menu.addAction(pipe_style_action)

        # Language submenu
        view_menu.addSeparator()
        lang_menu = view_menu.addMenu(tr("Language"))
        action_en = QtGui.QAction("English", self)
        action_en.setCheckable(True)
        action_en.setChecked(get_language() == 'en')
        action_en.triggered.connect(lambda: self._set_language('en'))
        lang_menu.addAction(action_en)
        action_zh = QtGui.QAction("繁體中文", self)
        action_zh.setCheckable(True)
        action_zh.setChecked(get_language() == 'zh_TW')
        action_zh.triggered.connect(lambda: self._set_language('zh_TW'))
        lang_menu.addAction(action_zh)

        # ── Help menu ──
        help_menu = menubar.addMenu(tr("&Help"))
        open_manual_action = QtGui.QAction(tr("Open Manual (Offline)"), self)
        open_manual_action.setShortcut("F1")
        open_manual_action.triggered.connect(self._open_manual)
        help_menu.addAction(open_manual_action)

        open_online_action = QtGui.QAction(tr("Open Online Manual"), self)
        open_online_action.triggered.connect(self._open_online_manual)
        help_menu.addAction(open_online_action)

        help_menu.addSeparator()
        toggle_help_action = QtGui.QAction(tr("Node Help Panel"), self)
        toggle_help_action.setCheckable(True)
        toggle_help_action.setChecked(True)
        help_menu.addAction(toggle_help_action)
        # Connect after help dock is created (deferred in _setup_help_dock)
        self._toggle_help_action = toggle_help_action
        toggle_help_action.toggled.connect(self.dockWidgetHelp.setVisible)
        self.dockWidgetHelp.visibilityChanged.connect(toggle_help_action.setChecked)

    def _open_online_manual(self):
        """Open the online manual (GitHub Pages) in the default browser."""
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl("https://m00zu.github.io/Synapse/"))

    def _open_manual(self):
        """Open the bundled HTML manual in the default browser."""
        candidates = [
            os.path.join(os.path.dirname(__file__), "site", "index.html"),  # pip / source
            os.path.join(os.path.dirname(sys.executable), "site", "index.html"),  # Nuitka
        ]
        for site_index in candidates:
            if os.path.isfile(site_index):
                url = QtCore.QUrl.fromLocalFile(os.path.abspath(site_index))
                QtGui.QDesktopServices.openUrl(url)
                return
        # No bundled docs — fall back to online
        self._open_online_manual()

    @staticmethod
    def _docstring_to_html(doc: str) -> str:
        """Convert a markdown-ish docstring to HTML for the help panel.

        Supports: paragraphs, ``code blocks``, `inline code`, **bold**,
        *italic*, markdown tables (| col | col |), and bullet lists (- item).
        """
        import re

        def _inline(t):
            """Apply inline formatting: code, bold, italic."""
            t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
            t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
            t = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', t)
            return t

        def _is_bullet(line):
            return bool(re.match(r'^[-*]\s', line.strip()))

        def _render_bullets(bullet_lines):
            items = ''.join(
                f"<li>{_inline(re.sub(r'^[-*]\s+', '', l.strip()))}</li>"
                for l in bullet_lines)
            return f"<ul style='margin:4px 0;'>{items}</ul>"

        blocks = re.split(r'\n{2,}', doc)
        parts = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.split('\n')
            # Code block (all lines indented 4+ spaces or start with ```)
            if all(l.startswith('    ') or l.startswith('\t') for l in lines):
                code = '\n'.join(l[4:] if l.startswith('    ') else l[1:] for l in lines)
                parts.append(f"<pre style='background:#2a2a2a;color:#ccc;"
                             f"padding:8px;border-radius:4px;font-size:12px;"
                             f"overflow-x:auto;'>{code}</pre>")
                continue
            # Table (lines starting with |)
            if all(l.strip().startswith('|') for l in lines):
                rows = []
                for i, l in enumerate(lines):
                    cells = [c.strip() for c in l.strip().strip('|').split('|')]
                    # Skip separator rows (|---|---|)
                    if all(set(c) <= {'-', ':', ' '} for c in cells):
                        continue
                    tag = 'th' if i == 0 else 'td'
                    style = "padding:3px 8px;border:1px solid #555;"
                    row = ''.join(f"<{tag} style='{style}'>{_inline(c)}</{tag}>" for c in cells)
                    rows.append(f"<tr>{row}</tr>")
                if rows:
                    parts.append("<table style='border-collapse:collapse;"
                                 "margin:6px 0;font-size:13px;'>"
                                 + ''.join(rows) + "</table>")
                continue
            # Pure bullet list (all lines are bullets)
            if all(_is_bullet(l) for l in lines):
                parts.append(_render_bullets(lines))
                continue
            # Mixed block: label line(s) followed by bullet items
            bullet_start = None
            for i, l in enumerate(lines):
                if _is_bullet(l):
                    bullet_start = i
                    break
            if bullet_start is not None and bullet_start > 0 and all(
                _is_bullet(l) for l in lines[bullet_start:]
            ):
                label = ' '.join(l.strip() for l in lines[:bullet_start])
                parts.append(f"<p>{_inline(label)}</p>")
                parts.append(_render_bullets(lines[bullet_start:]))
                continue
            # Heading (### or ##)
            if lines[0].startswith('#'):
                level = len(lines[0]) - len(lines[0].lstrip('#'))
                level = min(max(level, 2), 4)
                text = lines[0].lstrip('# ').strip()
                rest = ' '.join(l.strip() for l in lines[1:]).strip()
                parts.append(f"<h{level}>{_inline(text)}</h{level}>")
                if rest:
                    parts.append(f"<p>{_inline(rest)}</p>")
                continue
            # Normal paragraph — apply inline formatting
            text = ' '.join(l.strip() for l in lines)
            parts.append(f"<p>{_inline(text)}</p>")
        return '\n'.join(parts)

    def _on_node_selection_for_help(self, selected, _deselected):
        """Update the Node Help dock when a node is selected."""
        if not selected:
            return
        node = selected[0]
        cls = type(node)
        name = getattr(cls, 'NODE_NAME', node.name())
        doc = cls.__doc__ or ""

        # Clean up docstring
        import textwrap
        doc = textwrap.dedent(doc).strip()
        # Remove keyword lines
        lines = []
        for line in doc.split('\n'):
            stripped = line.strip()
            if stripped.lower().startswith('keywords:') or stripped.lower().startswith('keyword:'):
                continue
            lines.append(line)
        doc = '\n'.join(lines).strip()

        # Build HTML
        html_parts = [
            f"<h2 style='margin-top:4px;'>{name}</h2>",
        ]

        if doc:
            # Convert markdown-ish docstring to HTML
            html_parts.append(self._docstring_to_html(doc))

        # Ports — show name + data type (reverse-mapped from port color)
        from .nodes.base import PORT_COLORS
        _color_to_type = {}
        for tname, rgb in PORT_COLORS.items():
            key = rgb[:3]
            if key not in _color_to_type:
                _color_to_type[key] = tname

        def _port_type(port):
            try:
                c = port.color
                rgb = (c[0], c[1], c[2]) if len(c) >= 3 else c
                return _color_to_type.get(rgb, '')
            except Exception:
                return ''

        in_ports = list(node.inputs().values()) if node.inputs() else []
        out_ports = list(node.outputs().values()) if node.outputs() else []
        if in_ports or out_ports:
            html_parts.append("<h3>Ports</h3>")
            _s = "padding:2px 8px;border:1px solid #555;"
            html_parts.append("<table cellpadding='3' cellspacing='0' "
                              "style='border-collapse:collapse;'>")
            html_parts.append(
                f"<tr><th style='{_s}'>Direction</th>"
                f"<th style='{_s}'>Port</th>"
                f"<th style='{_s}'>Type</th></tr>")
            for port in in_ports:
                dtype = _port_type(port)
                html_parts.append(
                    f"<tr><td style='{_s}'><b>Input</b></td>"
                    f"<td style='{_s}'><code>{port.name()}</code></td>"
                    f"<td style='{_s}'>{dtype}</td></tr>")
            for port in out_ports:
                dtype = _port_type(port)
                html_parts.append(
                    f"<tr><td style='{_s}'><b>Output</b></td>"
                    f"<td style='{_s}'><code>{port.name()}</code></td>"
                    f"<td style='{_s}'>{dtype}</td></tr>")
            html_parts.append("</table>")

        # Properties (from the node's model)
        try:
            props = node.model.custom_properties
            if props:
                # Filter out internal/UI properties
                ui_props = getattr(cls, '_UI_PROPS', frozenset())
                visible = {k: v for k, v in props.items()
                           if k not in ui_props and not k.startswith('_')}
                if visible:
                    html_parts.append("<h3>Properties</h3>")
                    html_parts.append("<ul>")
                    for k, v in visible.items():
                        val_str = str(v)
                        if len(val_str) > 60:
                            val_str = val_str[:60] + "..."
                        html_parts.append(
                            f"<li><code>{k}</code> = {val_str}</li>")
                    html_parts.append("</ul>")
        except Exception:
            pass

        self._help_browser.setHtml("\n".join(html_parts))

    def _install_plugin(self):
        """Quick-install: pick a .py, .zip, or .synpkg file and install it."""
        from .plugin_loader import install_plugin_file, install_plugin_package, install_synpkg
        from pathlib import Path
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, tr("Install Plugin"), str(Path.home()),
            tr("Plugin Files (*.py *.zip *.synpkg)")
        )
        if not paths:
            return
        installed = []
        for path_str in paths:
            src = Path(path_str)
            suffix = src.suffix.lower()

            if suffix == '.synpkg':
                pkg_name = src.stem.split('-')[0]
                dst = self._plugin_dir / pkg_name
                if dst.exists():
                    reply = QtWidgets.QMessageBox.question(
                        self, tr("Overwrite Plugin?"),
                        f"'{pkg_name}' is already installed. Overwrite it?",
                        QtWidgets.QMessageBox.StandardButton.Yes |
                        QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                        continue
                ok, msg = install_synpkg(src, self._plugin_dir, overwrite=True)
                if ok:
                    installed.append(pkg_name)
                else:
                    QtWidgets.QMessageBox.warning(self, tr("Install Failed"), msg)
            elif suffix == '.zip':
                ok, msg = install_plugin_package(src, self._plugin_dir, overwrite=True)
                if ok:
                    installed.append(src.stem)
                else:
                    QtWidgets.QMessageBox.warning(self, tr("Install Failed"), msg)
            else:
                dst = self._plugin_dir / src.name
                if dst.exists():
                    reply = QtWidgets.QMessageBox.question(
                        self, tr("Overwrite Plugin?"),
                        f"'{src.name}' is already installed. Overwrite it?",
                        QtWidgets.QMessageBox.StandardButton.Yes |
                        QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                        continue
                ok, msg = install_plugin_file(src, self._plugin_dir, overwrite=True)
                if ok:
                    installed.append(src.name)
                else:
                    QtWidgets.QMessageBox.warning(self, tr("Install Failed"), msg)
        if installed:
            names = ', '.join(installed)
            self.statusBar().showMessage(
                f"Installed: {names} — restart to activate.", 8000
            )

    def _show_plugin_manager(self):
        """Open the Plugin Manager dialog."""
        from .plugin_loader import PluginManagerDialog
        dlg = PluginManagerDialog(self._plugin_results, self._plugin_dir, self)
        dlg.exec()

    def _set_language(self, lang: str):
        set_language(lang)
        self.statusBar().showMessage(tr("Restart required to apply language change."), 6000)

    # ── Pipe style ────────────────────────────────────────────────────────────

    def _load_pipe_settings(self) -> dict:
        raw = self.settings.value('pipe_style', {})
        if not isinstance(raw, dict):
            raw = {}
        defaults = {'layout': 1, 'width': 1.2, 'draw_type': 0, 'color': None}
        defaults.update(raw)
        return defaults

    def _save_pipe_settings(self):
        self.settings.setValue('pipe_style', self._pipe_settings)

    def _pipe_color(self) -> tuple:
        c = self._pipe_settings.get('color')
        if c:
            return tuple(c)
        from NodeGraphQt.constants import PipeEnum
        return PipeEnum.COLOR.value

    def _pipe_key(self, pipe) -> tuple | None:
        """Stable key for a pipe: (out_node_id, out_port, in_node_id, in_port)."""
        try:
            op = pipe.output_port
            ip = pipe.input_port
            if op is None or ip is None:
                return None
            return (op.node.id, op.name, ip.node.id, ip.name)  # .name is a property
        except Exception:
            return None

    def _apply_pipe_layout(self):
        self.graph.set_pipe_style(int(self._pipe_settings.get('layout', 1)))
    
    def _apply_pipe_style_to_all(self):
        self._apply_pipe_layout()
        g_color     = self._pipe_color()
        g_width     = float(self._pipe_settings.get('width', 1.2))
        g_draw_type = int(self._pipe_settings.get('draw_type', 0))
        for pipe in self.graph.viewer().all_pipes():
            key      = self._pipe_key(pipe)
            override = self._per_pipe_settings.get(key) if key else None
            redraw   = False
            if override:
                color     = tuple(override['color']) if override.get('color') else g_color
                width     = float(override.get('width', g_width))
                draw_type = int(override.get('draw_type', g_draw_type))
                if 'layout' in override:
                    lv = int(override['layout'])
                    pipe.viewer_pipe_layout = lambda _lv=lv: _lv
                    redraw = True
                elif 'viewer_pipe_layout' in pipe.__dict__:
                    del pipe.viewer_pipe_layout
                    redraw = True
            else:
                color, width, draw_type = g_color, g_width, g_draw_type
                if 'viewer_pipe_layout' in pipe.__dict__:
                    del pipe.viewer_pipe_layout
                    redraw = True
            if redraw and pipe.input_port and pipe.output_port:
                pipe.draw_path(pipe.input_port, pipe.output_port)
            # Set backing attributes so pipe.reset() (called on deselect) uses our values
            pipe.color = color
            pipe.style = draw_type
            if not pipe.highlighted() and not pipe.active():
                pipe.set_pipe_styling(color=color, width=width, style=draw_type)

    def _on_port_connected_style(self, *_):
        """Apply current pipe style to all pipes whenever a new connection is made."""
        self._apply_pipe_style_to_all()

    def _show_pipe_context_menu(self, pipe, global_pos):
        """Right-click context menu shown when the user right-clicks on a pipe."""
        key          = self._pipe_key(pipe)
        has_override = bool(key and key in self._per_pipe_settings)

        menu         = QtWidgets.QMenu(self)
        act_style    = menu.addAction("Style This Pipe…")
        act_reset    = menu.addAction("Reset to Global Style")
        act_reset.setEnabled(has_override)

        chosen = menu.exec(global_pos)
        if chosen == act_style:
            self._show_single_pipe_dialog(pipe)
        elif chosen == act_reset:
            if key:
                self._per_pipe_settings.pop(key, None)
            self._apply_pipe_style_to_all()

    def _show_single_pipe_dialog(self, pipe):
        """Style dialog scoped to one specific pipe."""
        from NodeGraphQt.constants import PipeEnum

        key      = self._pipe_key(pipe)
        override = self._per_pipe_settings.get(key, {}) if key else {}

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Style This Pipe")
        dlg.setMinimumWidth(300)
        form = QtWidgets.QFormLayout(dlg)
        form.setSpacing(10)
        form.setContentsMargins(16, 16, 16, 12)

        # Shape (per-pipe: monkey-patches viewer_pipe_layout on the instance)
        combo_layout = QtWidgets.QComboBox()
        combo_layout.addItems(["Straight", "Curved", "Angled"])
        combo_layout.setCurrentIndex(int(override.get('layout', self._pipe_settings.get('layout', 1))))
        form.addRow("Shape:", combo_layout)

        # Width
        spin_width = QtWidgets.QDoubleSpinBox()
        spin_width.setRange(0.5, 8.0)
        spin_width.setSingleStep(0.5)
        spin_width.setDecimals(1)
        spin_width.setValue(float(override.get('width', self._pipe_settings.get('width', 1.2))))
        form.addRow("Width:", spin_width)

        # Draw style
        combo_draw = QtWidgets.QComboBox()
        combo_draw.addItems(["Solid", "Dashed", "Dotted"])
        combo_draw.setCurrentIndex(int(override.get('draw_type', self._pipe_settings.get('draw_type', 0))))
        form.addRow("Style:", combo_draw)

        # Color
        color_row = QtWidgets.QWidget()
        color_h   = QtWidgets.QHBoxLayout(color_row)
        color_h.setContentsMargins(0, 0, 0, 0)
        color_h.setSpacing(8)

        _saved     = override.get('color') or self._pipe_settings.get('color')
        _sel_color = list(_saved) if _saved else list(PipeEnum.COLOR.value)
        color_btn  = QtWidgets.QPushButton()
        color_btn.setFixedSize(40, 24)
        chk_default = QtWidgets.QCheckBox("Use global")
        chk_default.setChecked(not override.get('color'))

        def _refresh_btn():
            color_btn.setEnabled(not chk_default.isChecked())
            r, g, b = (_sel_color[:3] if not chk_default.isChecked()
                       else list(self._pipe_color())[:3])
            color_btn.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); "
                "border: 1px solid #555; border-radius: 2px;"
            )

        def _pick_color():
            nonlocal _sel_color
            r, g, b = _sel_color[:3]
            a = _sel_color[3] if len(_sel_color) > 3 else 255
            c = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(r, g, b, a), dlg, "Pipe Color",
                QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel,
            )
            if c.isValid():
                _sel_color = [c.red(), c.green(), c.blue(), c.alpha()]
                chk_default.setChecked(False)
                _refresh_btn()

        color_btn.clicked.connect(_pick_color)
        chk_default.toggled.connect(_refresh_btn)
        _refresh_btn()
        color_h.addWidget(color_btn)
        color_h.addWidget(chk_default)
        form.addRow("Color:", color_row)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel |
            QtWidgets.QDialogButtonBox.StandardButton.Apply,
        )
        form.addRow(btns)

        def _apply():
            layout    = combo_layout.currentIndex()
            width     = spin_width.value()
            draw_type = combo_draw.currentIndex()
            color     = None if chk_default.isChecked() else list(_sel_color)
            if key:
                self._per_pipe_settings[key] = {
                    'layout': layout, 'width': width, 'draw_type': draw_type, 'color': color,
                }
            # Apply shape override via monkey-patch
            pipe.viewer_pipe_layout = lambda _lv=layout: _lv
            if pipe.input_port and pipe.output_port:
                pipe.draw_path(pipe.input_port, pipe.output_port)
            # Apply immediately to the pipe
            g_color = self._pipe_color()
            c = tuple(color) if color else g_color
            pipe.color = c
            pipe.style = draw_type
            pipe.set_pipe_styling(color=c, width=width, style=draw_type)
        
        btns.accepted.connect(lambda: (_apply(), dlg.accept()))
        btns.rejected.connect(dlg.reject)
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(_apply)
        dlg.exec()

    def _show_pipe_style_dialog(self):
        from NodeGraphQt.constants import PipeEnum

        viewer    = self.graph.viewer()
        sel_pipes = viewer.selected_pipes()

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(tr("Pipe Style...").rstrip('.…'))
        dlg.setMinimumWidth(360)

        outer = QtWidgets.QVBoxLayout(dlg)
        outer.setSpacing(10)
        outer.setContentsMargins(16, 16, 16, 12)

        # ── Scope ─────────────────────────────────────────────────────────────
        scope_box = QtWidgets.QGroupBox("Apply to:")
        scope_h   = QtWidgets.QHBoxLayout(scope_box)
        scope_h.setSpacing(16)
        radio_all = QtWidgets.QRadioButton("All pipes")
        sel_label = f"Selected ({len(sel_pipes)})" if sel_pipes else "Selected (none)"
        radio_sel = QtWidgets.QRadioButton(sel_label)
        radio_sel.setEnabled(bool(sel_pipes))
        if sel_pipes:
            radio_sel.setChecked(True)
        else:
            radio_all.setChecked(True)
        scope_h.addWidget(radio_all)
        scope_h.addWidget(radio_sel)
        scope_h.addStretch()
        outer.addWidget(scope_box)

        # ── Style fields ──────────────────────────────────────────────────────
        form = QtWidgets.QFormLayout()
        form.setSpacing(10)
        outer.addLayout(form)

        # Shape (global only — disabled in "selected" mode)
        combo_layout = QtWidgets.QComboBox()
        combo_layout.addItems(["Straight", "Curved", "Angled"])
        combo_layout.setCurrentIndex(int(self._pipe_settings.get('layout', 1)))
        form.addRow("Shape:", combo_layout)

        # Width
        spin_width = QtWidgets.QDoubleSpinBox()
        spin_width.setRange(0.5, 8.0)
        spin_width.setSingleStep(0.5)
        spin_width.setDecimals(1)
        spin_width.setValue(float(self._pipe_settings.get('width', 1.2)))
        form.addRow("Width:", spin_width)

        # Draw style
        combo_draw = QtWidgets.QComboBox()
        combo_draw.addItems(["Solid", "Dashed", "Dotted"])
        combo_draw.setCurrentIndex(int(self._pipe_settings.get('draw_type', 0)))
        form.addRow("Style:", combo_draw)

        # Color
        color_row = QtWidgets.QWidget()
        color_h   = QtWidgets.QHBoxLayout(color_row)
        color_h.setContentsMargins(0, 0, 0, 0)
        color_h.setSpacing(8)

        _saved     = self._pipe_settings.get('color')
        _sel_color = list(_saved) if _saved else list(PipeEnum.COLOR.value)
        color_btn  = QtWidgets.QPushButton()
        color_btn.setFixedSize(40, 24)
        chk_default = QtWidgets.QCheckBox("Use default")
        chk_default.setChecked(_saved is None)

        def _refresh_btn():
            color_btn.setEnabled(not chk_default.isChecked())
            r, g, b = (_sel_color[:3] if not chk_default.isChecked()
                       else list(PipeEnum.COLOR.value)[:3])
            color_btn.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); "
                "border: 1px solid #555; border-radius: 2px;"
            )

        def _pick_color():
            nonlocal _sel_color
            r, g, b = _sel_color[:3]
            a = _sel_color[3] if len(_sel_color) > 3 else 255
            c = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(r, g, b, a), dlg, "Pipe Color",
                QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel,
            )
            if c.isValid():
                _sel_color = [c.red(), c.green(), c.blue(), c.alpha()]
                chk_default.setChecked(False)
                _refresh_btn()

        color_btn.clicked.connect(_pick_color)
        chk_default.toggled.connect(_refresh_btn)
        _refresh_btn()
        color_h.addWidget(color_btn)
        color_h.addWidget(chk_default)
        form.addRow("Color:", color_row)

        # Shape works in both modes: global (set_pipe_style) or per-pipe (monkey-patch)

        # ── Bottom button row ─────────────────────────────────────────────────
        btn_row     = QtWidgets.QHBoxLayout()
        btn_rst_sel = QtWidgets.QPushButton("Reset Selected")
        btn_rst_sel.setEnabled(bool(sel_pipes))
        btn_rst_all = QtWidgets.QPushButton("Reset All")
        btn_row.addWidget(btn_rst_sel)
        btn_row.addWidget(btn_rst_all)
        btn_row.addStretch()

        std_btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel |
            QtWidgets.QDialogButtonBox.StandardButton.Apply,
        )
        btn_row.addWidget(std_btns)
        outer.addLayout(btn_row)

        # ── Logic ─────────────────────────────────────────────────────────────
        def _apply():
            width     = spin_width.value()
            draw_type = combo_draw.currentIndex()
            color     = None if chk_default.isChecked() else list(_sel_color)

            if radio_all.isChecked():
                self._pipe_settings['layout']    = combo_layout.currentIndex()
                self._pipe_settings['width']     = width
                self._pipe_settings['draw_type'] = draw_type
                self._pipe_settings['color']     = color
                self._save_pipe_settings()
                self._apply_pipe_style_to_all()
            else:
                # Store per-pipe overrides and apply immediately
                layout  = combo_layout.currentIndex()
                g_color = self._pipe_color()
                for pipe in sel_pipes:
                    key = self._pipe_key(pipe)
                    if key:
                        self._per_pipe_settings[key] = {
                            'layout': layout, 'width': width, 'draw_type': draw_type, 'color': color,
                        }
                    pipe.viewer_pipe_layout = lambda _lv=layout: _lv
                    if pipe.input_port and pipe.output_port:
                        pipe.draw_path(pipe.input_port, pipe.output_port)
                    c = tuple(color) if color else g_color
                    pipe.color = c
                    pipe.style = draw_type
                    pipe.set_pipe_styling(color=c, width=width, style=draw_type)

        def _reset_selected():
            for pipe in sel_pipes:
                key = self._pipe_key(pipe)
                if key:
                    self._per_pipe_settings.pop(key, None)
            self._apply_pipe_style_to_all()

        def _reset_all():
            self._per_pipe_settings.clear()
            self._apply_pipe_style_to_all()

        btn_rst_sel.clicked.connect(_reset_selected)
        btn_rst_all.clicked.connect(_reset_all)
        std_btns.accepted.connect(lambda: (_apply(), dlg.accept()))
        std_btns.rejected.connect(dlg.reject)
        std_btns.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(_apply)
        dlg.exec()

    def _populate_examples(self, menu, is_append=False):
        """Scan workflows directory and populate the examples menu."""
        workflow_dir = os.path.join(os.path.dirname(__file__), 'workflows')
        if not os.path.exists(workflow_dir):
            return
            
        files = [f for f in os.listdir(workflow_dir) if f.endswith('.json')]
        if not files:
            action = menu.addAction(tr("No examples found"))
            action.setEnabled(False)
            return
            
        for file in sorted(files):
            file_path = os.path.join(workflow_dir, file)
            name = file.replace('.json', '').replace('_', ' ').title()
            action = menu.addAction(name)
            
            if is_append:
                action.triggered.connect(lambda checked=False, p=file_path: self._append_example(p))
            else:
                action.triggered.connect(lambda checked=False, p=file_path: self._load_example(p))

    def _copy_nodes(self):
        """Action handler for copying selected nodes."""
        self.graph.copy_nodes()

    def _paste_nodes(self):
        """Action handler for pasting nodes from clipboard."""
        pasted = self.graph.paste_nodes()
        if pasted:
            self.statusBar().showMessage(f"Pasted {len(pasted)} nodes.", 2000)

    def _select_all_nodes(self):
        """Select every node on the canvas (Ctrl/Cmd+A)."""
        self.graph.select_all()

    def _focus_search_bar(self):
        """Reveal and focus the Node Explorer search bar (Ctrl/Cmd+F)."""
        # Make sure the Node Explorer dock is visible
        if not self.dockWidgetNodes.isVisible():
            self.dockWidgetNodes.show()
        self.dockWidgetNodes.raise_()
        # Focus the search bar so the user can start typing immediately
        self.nodes_tree._search_bar.setFocus()
        self.nodes_tree._search_bar.selectAll()

    def _save_workflow(self):
        """Save the current graph session."""
        suggested = self._current_workflow_path or self._workflow_dialog_start_dir()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Save Workflow"), suggested, tr("JSON Files (*.json)")
        )
        if file_path:
            try:
                self.graph.save_session(file_path)
                self._current_workflow_path = os.path.abspath(file_path)
                self._record_recent_workflow(self._current_workflow_path)
                self._mark_clean()
                self._clear_recovery_file()
                self.statusBar().showMessage(f"Workflow saved to {file_path}", 3000)
                return True
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, tr("Save Error"), f"Could not save workflow: {str(e)}")
                return False
        return False

    def _export_as_script(self):
        """Export the current graph as a standalone Python script."""
        from .export_script import export_graph_to_script
        script = export_graph_to_script(self.graph)
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Export as Python Script"), "workflow.py",
            tr("Python Files (*.py)")
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(script)
                self.statusBar().showMessage(
                    tr("Script exported to") + f" {file_path}", 5000)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, tr("Export Error"),
                    f"Could not export script: {str(e)}")

    def _open_workflow(self):
        """Open a graph session from a file."""
        if not self._confirm_discard_unsaved(tr("opening another workflow")):
            return
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Open Workflow"), self._workflow_dialog_start_dir(), tr("JSON Files (*.json)")
        )
        if file_path:
            self._load_example(file_path, prompt_unsaved=False)
            
    def _append_workflow(self):
        """Append a graph session from a file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Append Workflow"), self._workflow_dialog_start_dir(), tr("JSON Files (*.json)")
        )
        if file_path:
            self._append_example(file_path)

    @staticmethod
    def _migrate_layout_node_types(layout_data: dict) -> int:
        """
        Migrate deprecated node types in serialized workflow data.
        Returns number of migrated nodes.
        """
        if not isinstance(layout_data, dict):
            return 0
        nodes = layout_data.get('nodes', {})
        if not isinstance(nodes, dict):
            return 0
        migrated = 0
        for n_data in nodes.values():
            if not isinstance(n_data, dict):
                continue
            t = str(n_data.get('type_', ''))
            if t.endswith('.GlobalMaskPropsNode'):
                n_data['type_'] = t[:-len('GlobalMaskPropsNode')] + 'ImageStatsNode'
                custom = n_data.get('custom') if isinstance(n_data.get('custom'), dict) else {}
                custom.setdefault('per_channel', False)
                n_data['custom'] = custom
                migrated += 1
        return migrated

    def _load_example(self, file_path, prompt_unsaved=True):
        """Load a graph session and handle any errors."""
        if prompt_unsaved and not self._confirm_discard_unsaved("loading another workflow"):
            return False
        try:
            with open(file_path, 'r', encoding='utf-8') as data_file:
                layout_data = json.load(data_file)
            migrated = self._migrate_layout_node_types(layout_data)
            self.graph.deserialize_session(layout_data, clear_session=True, clear_undo_stack=True)
            self._current_workflow_path = os.path.abspath(file_path)
            self._record_recent_workflow(self._current_workflow_path)
            self._mark_clean()
            self._clear_recovery_file()
            if migrated > 0:
                self.statusBar().showMessage(
                    f"Loaded workflow: {os.path.basename(file_path)} "
                    f"(migrated {migrated} Global Mask Props node(s) to Image Stats)",
                    5000
                )
            else:
                self.statusBar().showMessage(f"Loaded workflow: {os.path.basename(file_path)}", 3000)
            # Center the graph after loading
            self.graph.viewer().center_selection()
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, tr("Load Error"), f"Could not load workflow: {str(e)}")
            return False

    def _append_example(self, file_path):
        """Append a graph session visually adjacent to current nodes."""
        try:
            import json
            with open(file_path, encoding='utf-8') as data_file:
                layout_data = json.load(data_file)
            migrated = self._migrate_layout_node_types(layout_data)
            
            if not layout_data or 'nodes' not in layout_data:
                return

            # Analyze current nodes bounding box
            all_nodes = self.graph.all_nodes()
            offset_x, offset_y = 0, 0
            
            if all_nodes:
                min_x = min(n.pos()[0] for n in all_nodes)
                max_x = max(n.pos()[0] + getattr(n.view, 'width', 200) for n in all_nodes)
                min_y = min(n.pos()[1] for n in all_nodes)
                max_y = max(n.pos()[1] + getattr(n.view, 'height', 200) for n in all_nodes)
                
                width = max_x - min_x
                height = max_y - min_y
                padding = 100
                
                if width > height:
                    # Place below
                    target_x = min_x
                    target_y = max_y + padding
                else:
                    # Place right
                    target_x = max_x + padding
                    target_y = min_y
                    
                # Calculate import bounding box
                valid_nodes = [nd for nd in layout_data['nodes'].values() if 'pos' in nd]
                if valid_nodes:
                    import_min_x = min(nd['pos'][0] for nd in valid_nodes)
                    import_min_y = min(nd['pos'][1] for nd in valid_nodes)
                    
                    offset_x = target_x - import_min_x
                    offset_y = target_y - import_min_y
            
            # Apply offset to layout_data
            for n_data in layout_data.get('nodes', {}).values():
                if 'pos' in n_data:
                    n_data['pos'][0] += offset_x
                    n_data['pos'][1] += offset_y
                    
            # Actually deserialize the nodes without clearing the board
            # NodeGraphQt dynamically assigns fresh UUIDs to all created objects internally
            self.graph.deserialize_session(layout_data, clear_session=False, clear_undo_stack=False)
            self._mark_manual_dirty()
            if migrated > 0:
                self.statusBar().showMessage(
                    f"Appended workflow: {os.path.basename(file_path)} "
                    f"(migrated {migrated} Global Mask Props node(s) to Image Stats)",
                    5000
                )
            else:
                self.statusBar().showMessage(f"Appended workflow: {os.path.basename(file_path)}", 3000)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, tr("Append Error"), f"Could not append workflow: {str(e)}\n\n{traceback.format_exc()}")

    def display_properties_bin(self, node):
        if self.dockWidgetProperties.isHidden():
            self.dockWidgetProperties.show()
        if not self.properties_bin.isVisible():
            self.properties_bin.show()

    def _dispatch_ui_signal(self, node_id, signal_type, value):
        """
        Master dispatcher for node UI updates. 
        Guaranteed to run on the Main Thread because this window is a QObject.
        """
        # Find the node in the graph by its ID
        target_node = None
        for node in self.graph.all_nodes():
            if node.id == node_id:
                target_node = node
                break
        
        if not target_node:
            # Signal arrived after node deletion; clean any orphan visuals.
            QtCore.QTimer.singleShot(0, self._purge_orphan_node_items)
            return
            
        # Dispatch to the node's UI helpers
        if signal_type == 'progress':
            if hasattr(target_node, '_set_progress_ui'):
                target_node._set_progress_ui(value)
        elif signal_type == 'status':
            if value == 'clean' and hasattr(target_node, '_mark_clean_ui'):
                target_node._mark_clean_ui()
            elif value == 'dirty' and hasattr(target_node, '_mark_dirty_ui'):
                target_node._mark_dirty_ui()
            elif value == 'error' and hasattr(target_node, '_mark_error_ui'):
                target_node._mark_error_ui()
            elif value == 'skipped' and hasattr(target_node, '_mark_skipped_ui'):
                target_node._mark_skipped_ui()
            elif value == 'disabled' and hasattr(target_node, '_mark_disabled_ui'):
                target_node._mark_disabled_ui()
            elif value == 'enabled' and hasattr(target_node, '_mark_enabled_ui'):
                target_node._mark_enabled_ui()
        elif signal_type == 'display':
            if hasattr(target_node, '_display_ui'):
                target_node._display_ui(value)

    def execute_graph(self):
        # Guard: don't start if already running
        try:
            if self.worker_thread and self.worker_thread.isRunning():
                return
        except RuntimeError:
            pass
        # Warn if a FolderIteratorNode is present — user may have meant Batch Run
        # But skip the warning if all accumulators already have cached results
        # (the batch loop will be skipped anyway).
        for node in self.graph.all_nodes():
            if isinstance(node, FolderIteratorNode):
                accumulators = [n for n in self.graph.all_nodes()
                                if getattr(n, '_is_accumulator', False)]
                all_complete = (accumulators
                                and all(getattr(a, '_batch_complete', False)
                                        for a in accumulators))
                if all_complete:
                    # Batch already done — silently use batch path which
                    # will skip the loop and only run downstream nodes.
                    self.execute_batch()
                    return
                btn = QtWidgets.QMessageBox.question(
                    self,
                    tr("Run Graph"),
                    tr("A 'Folder Iterator' node is in the graph.\n\n"
                       "Did you mean to use Batch Run (Ctrl+B) to process all files?\n\n"
                       "Click 'Yes' to switch to Batch Run, or 'No' to run once."),
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes,
                )
                if btn == QtWidgets.QMessageBox.Yes:
                    self.execute_batch()
                    return
                break

        # Gather all nodes
        all_nodes = self.graph.all_nodes()
        if not all_nodes:
            return

        # Perform topological sort with cycle detection
        visited = set()
        in_stack = set()
        sorted_nodes = []
        _cycle_found = False

        def visit(node):
            nonlocal _cycle_found
            if _cycle_found or node in visited:
                return
            if node in in_stack:
                _cycle_found = True
                return
            in_stack.add(node)
            for port in node.connected_input_nodes().values():
                for upstream_node in port:
                    visit(upstream_node)
            in_stack.discard(node)
            visited.add(node)
            sorted_nodes.append(node)

        for node in all_nodes:
            visit(node)

        if _cycle_found:
            self.statusBar().showMessage(tr("Error: circular connection detected. Remove the cycle and try again."), 8000)
            return

        # UI State
        self.btn_run.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage(tr("Executing graph..."))

        # Setup Thread and Worker
        self.worker_thread = QtCore.QThread()
        self.worker = GraphWorker(sorted_nodes)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        
        self.worker.finished.connect(self._on_execution_finished)
        self.worker.error.connect(self._on_execution_error)

        self._batch_start = time.perf_counter()
        self.worker_thread.start()

    def execute_batch(self):
        # Guard: don't start if already running
        try:
            if self.worker_thread and self.worker_thread.isRunning():
                return
        except RuntimeError:
            pass
        # 1. Find a batch iterator node (FolderIterator or any node with get_batch_items)
        iter_node = None
        files = None
        for node in self.graph.all_nodes():
            if isinstance(node, FolderIteratorNode):
                iter_node = node
                break
            if hasattr(node, 'get_batch_items') and callable(node.get_batch_items):
                iter_node = node
                break

        if not iter_node:
            QtWidgets.QMessageBox.warning(self, tr("Batch Run"), tr("No iterator node found in the graph (Folder Iterator, Video Iterator, etc.)."))
            return

        # 2. Collect items to iterate over
        if isinstance(iter_node, FolderIteratorNode):
            folder = iter_node.get_property('folder_path')
            pattern = iter_node.get_property('pattern')

            if not folder or not os.path.exists(folder):
                QtWidgets.QMessageBox.warning(self, tr("Batch Run"), f"Invalid folder path in 'Folder Iterator': {folder}")
                return

            from pathlib import Path
            iterate_mode = iter_node.get_property('iterate_mode')

            import re
            def _nat_key(p):
                return [int(s) if s.isdigit() else s.lower()
                        for s in re.split(r'(\d+)', p.name)]
            all_matches = sorted(list(Path(folder).glob(pattern)), key=_nat_key)
            if iterate_mode == 'Subdirectories':
                files = [p for p in all_matches if p.is_dir()]
                item_type = "directories"
            else:
                files = [p for p in all_matches if p.is_file()]
                item_type = "files"

            if not files:
                QtWidgets.QMessageBox.warning(self, tr("Batch Run"), f"No {item_type} found matching '{pattern}' in {folder}")
                return
        else:
            # Generic iterator (e.g. VideoIteratorNode)
            files = iter_node.get_batch_items()
            if not files:
                QtWidgets.QMessageBox.warning(self, tr("Batch Run"), f"No items to iterate from '{iter_node.name()}'.")
                return

        # 3. Perform topological sort with cycle detection
        all_nodes = self.graph.all_nodes()
        visited = set()
        in_stack = set()
        sorted_nodes = []
        _cycle_found = False

        def visit(node):
            nonlocal _cycle_found
            if _cycle_found or node in visited:
                return
            if node in in_stack:
                _cycle_found = True
                return
            in_stack.add(node)
            for port in node.connected_input_nodes().values():
                for upstream_node in port:
                    visit(upstream_node)
            in_stack.discard(node)
            visited.add(node)
            sorted_nodes.append(node)

        for node in all_nodes:
            visit(node)

        if _cycle_found:
            self.statusBar().showMessage(tr("Error: circular connection detected. Remove the cycle and try again."), 8000)
            return

        # 4. Starting execution
        self.btn_run.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._is_batch_running = True
        self._batch_total = len(files)
        self._batch_fail_count = 0
        self._batch_failures = []
        self._batch_start = time.perf_counter()
        self._set_batch_progress_visible(True)
        self._update_batch_progress_ui(0, self._batch_total)
        self.statusBar().showMessage(tr("Starting batch execution..."))

        # Setup Thread and Worker
        self.worker_thread = QtCore.QThread()
        self.worker = BatchGraphWorker(sorted_nodes, iter_node, [str(f) for f in files])
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        
        self.worker.progress.connect(self._on_batch_progress)
        self.worker.file_failed.connect(self._on_batch_file_failed)
        self.worker.finished.connect(self._on_execution_finished)
        self.worker.error.connect(self._on_execution_error)

        self._batch_start = time.perf_counter()
        self.worker_thread.start()

    def _on_batch_progress(self, current, total):
        self._update_batch_progress_ui(current, total)
        self.statusBar().showMessage(f"Batch Execution: Processing file {current} of {total}...")

    def _on_batch_file_failed(self, file_name, message):
        self._batch_fail_count += 1
        self._batch_failures.append((file_name, message))
        self.statusBar().showMessage(
            f"Batch warning: skipped '{file_name}' ({self._batch_fail_count} failed so far).",
            4000,
        )

    def closeEvent(self, event):
        """Handle window close event."""
        if not self._confirm_discard_unsaved(tr("quitting")):
            event.ignore()
            return
        self._autosave_timer.stop()
        self._clear_recovery_file()
        if self.worker:
            # Signal the worker to stop between nodes (cannot interrupt mid-node)
            self.worker.stop()
            # Disconnect all cross-thread signals so no callbacks fire on
            # destroyed widgets while the thread winds down.
            for sig in (getattr(self.worker, s, None)
                        for s in ('finished', 'error', 'progress',
                                  'file_failed')):
                if sig is not None:
                    try:
                        sig.disconnect()
                    except RuntimeError:
                        pass
        try:
            # worker_thread may have been deleted by deleteLater() already
            thread_running = bool(self.worker_thread and self.worker_thread.isRunning())
        except RuntimeError:
            thread_running = False
        if thread_running:
            # Mark the thread as a daemon-like object: when the process
            # exits the OS will clean it up.  Do NOT call terminate() —
            # it kills the thread mid-C-extension (numpy/PIL) and causes
            # a segfault.  A short wait gives cooperative stop a chance.
            if not self.worker_thread.wait(200):
                # Thread still running — detach so it won't block exit.
                # The process exit will clean up the thread.
                self.worker_thread.finished.disconnect()
        event.accept()

    def stop_execution(self):
        """Force stop the current execution."""
        if self.worker:
            self.worker.stop()
            self.statusBar().showMessage(tr("Stopping execution..."))
            self.btn_stop.setEnabled(False)
            # The worker will emit finished (which re-enables buttons) once
            # the current node completes. But if it takes too long, ensure
            # the UI doesn't stay stuck — schedule a safety reset.
            QtCore.QTimer.singleShot(5000, self._safety_reset_ui)

    def _safety_reset_ui(self):
        """Fallback: if the worker thread is still stuck after stop was
        requested, re-enable the UI so the user isn't locked out."""
        try:
            still_running = (self.worker_thread is not None
                             and self.worker_thread.isRunning())
        except RuntimeError:
            still_running = False
        if still_running:
            self.btn_run.setEnabled(True)
            self.btn_batch.setEnabled(True)
            self.btn_stop.setEnabled(True)  # keep available for retry
            if self._is_batch_running:
                self._is_batch_running = False
                self._set_batch_progress_visible(False)
            self.statusBar().showMessage(
                tr("Stop requested — waiting for current node to finish."), 5000)

    def _on_execution_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.btn_stop.setEnabled(False)
        end = time.perf_counter() - self._batch_start
        was_stopped = self.worker and getattr(self.worker, '_stopped', False)
        if self._is_batch_running:
            self._update_batch_progress_ui(self._batch_total, self._batch_total)
            if was_stopped:
                self.statusBar().showMessage(
                    tr("Batch stopped by user."), 5000)
            else:
                self.statusBar().showMessage(
                    f"Batch finished in {end:.2f}s. Failed files: {self._batch_fail_count}.",
                    7000,
                )
            if not was_stopped and self._batch_failures:
                lines = [f"{i+1}. {name}" for i, (name, _msg) in enumerate(self._batch_failures[:10])]
                more = len(self._batch_failures) - len(lines)
                details = "\n".join(lines)
                if more > 0:
                    details += f"\n... and {more} more"
                QtWidgets.QMessageBox.warning(
                    self,
                    tr("Batch Completed with Errors"),
                    f"Processed with {self._batch_fail_count} failed file(s).\n\n{details}",
                )
            self._is_batch_running = False
            self._set_batch_progress_visible(False)
        else:
            if was_stopped:
                self.statusBar().showMessage(tr("Execution stopped by user."), 5000)
            else:
                self.statusBar().showMessage(f"Execution finished in {end:.2f}s.", 5000)
        
        # Collect results for summary (Leaf nodes)
        results = []
        for node in self.graph.all_nodes():
            out_port = node.outputs().get('out')
            if not out_port or not out_port.connected_ports():
                import matplotlib.figure
                val = node.output_values.get('out')
                if val is not None and not isinstance(val, matplotlib.figure.Figure) and not isinstance(node, DisplayNode) and not isinstance(node, DataTableCellNode) and not isinstance(node, DataFigureCellNode):
                    out_str = f"{node.name()}: {str(val)[:100]}..."
                    results.append(out_str)
        
        # if results:
        #     QtWidgets.QMessageBox.information(self, "Execution Complete", f"Graph Executed Successfully.\nFinal Outputs:\n{chr(10).join(results)}")

    def _on_execution_error(self, message):
        self.btn_run.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._is_batch_running:
            self._batch_fail_count += 1
            self._update_batch_progress_ui(self._batch_total, self._batch_total)
            self._is_batch_running = False
            self._set_batch_progress_visible(False)
        try:
            if self.worker_thread and self.worker_thread.isRunning():
                self.worker_thread.quit()
        except RuntimeError:
            pass
        self.statusBar().showMessage(tr("Execution failed."), 5000)
        QtWidgets.QMessageBox.critical(self, tr("Execution Error"), message)

    # ── Execution Order badges ──

    def _toggle_order_badges(self, checked: bool):
        self._show_order_badges = checked
        if checked:
            self._update_order_badges()
        else:
            self._clear_order_badges()

    def _clear_order_badges(self):
        for item in self._order_badge_items:
            scene = item.scene()
            if scene:
                scene.removeItem(item)
        self._order_badge_items.clear()

    def _update_order_badges(self):
        self._clear_order_badges()
        if not self._show_order_badges:
            return
        order = self._compute_topo_order()
        if order is None:
            return
        scene = self.graph.scene()
        if scene is None:
            return
        for i, node in enumerate(order):
            view = node.view
            rect = view.boundingRect()
            # Badge circle + number
            badge_size = 22.0
            x = rect.right() - badge_size - 4
            y = rect.top() + 4
            # Circle background
            circle = QtWidgets.QGraphicsEllipseItem(
                x, y, badge_size, badge_size, view)
            circle.setBrush(QtGui.QColor(30, 130, 230, 200))
            circle.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 1.5))
            circle.setZValue(100)
            # Number text
            text = QtWidgets.QGraphicsSimpleTextItem(str(i + 1), circle)
            font = QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold)
            text.setFont(font)
            text.setBrush(QtGui.QColor(255, 255, 255))
            # Center text in circle
            tr = text.boundingRect()
            text.setPos(
                x + (badge_size - tr.width()) / 2,
                y + (badge_size - tr.height()) / 2
            )
            self._order_badge_items.append(circle)

    # ── Execution Order panel helpers ──

    def _compute_topo_order(self):
        """Compute topological execution order. Returns list of nodes or None on cycle."""
        all_nodes = self.graph.all_nodes()
        if not all_nodes:
            return []
        visited = set()
        in_stack = set()
        sorted_nodes = []
        _cycle = False

        def visit(node):
            nonlocal _cycle
            if _cycle or node in visited:
                return
            if node in in_stack:
                _cycle = True
                return
            in_stack.add(node)
            for port in node.connected_input_nodes().values():
                for upstream in port:
                    visit(upstream)
            in_stack.discard(node)
            visited.add(node)
            sorted_nodes.append(node)

        for n in all_nodes:
            visit(n)
        return sorted_nodes if not _cycle else None

    def _refresh_execution_order(self):
        """Rebuild the execution order list widget."""
        self._exec_order_list.clear()
        order = self._compute_topo_order()
        if order is None:
            item = QtWidgets.QListWidgetItem("Cycle detected!")
            item.setForeground(QtGui.QColor(255, 80, 80))
            self._exec_order_list.addItem(item)
            return

        self._exec_order_nodes = order  # store for click lookup
        for i, node in enumerate(order):
            label = f"{i + 1}. {node.name()}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, node.id)
            self._exec_order_list.addItem(item)

    def _on_exec_order_item_clicked(self, item):
        """Select and center the clicked node on the canvas."""
        node_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not node_id:
            return
        node = self.graph.get_node_by_id(node_id)
        if node:
            self.graph.clear_selection()
            node.set_selected(True)
            self.graph.fit_to_selection()


def main():
    # Handle subcommands before launching the GUI
    if len(sys.argv) >= 2 and sys.argv[1] == 'package':
        from synapse.package_plugin import main as _pack_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip 'package' from argv
        _pack_main()
        return

    load_language()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Synapse")
    app.setApplicationDisplayName("Synapse")
    app.setStyle("Fusion")

    # Auto-regenerate LLM node schema if any node source file changed
    try:
        from synapse.export_node_schema import auto_regenerate_if_stale
        auto_regenerate_if_stale()
    except Exception as e:
        print(f"[schema] auto-regenerate skipped: {e}")

    # Set application icon (taskbar, dock, window icon)
    import pathlib
    _icon_path = pathlib.Path(__file__).parent / 'icons' / 'synapse_icon.png'
    if _icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(_icon_path)))

    # ThemeManager applies the initial dark palette and owns all theme transitions
    theme_manager = ThemeManager(app)

    window = NodeExecutionWindow(theme_manager=theme_manager)
    window.show()
    app.exec()

if __name__ == '__main__':
    main()
