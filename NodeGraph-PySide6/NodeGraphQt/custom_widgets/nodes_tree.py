#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import platform
import sys
from pathlib import Path

from PySide6 import QtWidgets, QtCore, QtGui

from NodeGraphQt.constants import MIME_TYPE, URN_SCHEME

TYPE_NODE     = QtWidgets.QTreeWidgetItem.UserType + 1
TYPE_CATEGORY = QtWidgets.QTreeWidgetItem.UserType + 2

_CAT_FAVORITES = '__favorites__'
_CAT_GROUP_PFX = '__group__'


def _make_port_icon(inputs, outputs, size=18):
    """
    Render a compact QIcon: input dots | divider | output dots, all in one pixmap.
    inputs / outputs are lists of (R, G, B) tuples.
    """
    dot_r = 3
    gap   = 2
    pad   = 1
    div_w = 1
    div_gap = 3

    def grp_w(n):
        return n * (dot_r * 2 + gap) - (gap if n else 0)

    has_div = bool(inputs and outputs)
    total_w = grp_w(len(inputs)) + (div_gap * 2 + div_w if has_div else 0) + grp_w(len(outputs))
    total_w = max(total_w + pad * 2, size)

    pixmap = QtGui.QPixmap(total_w, size)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    cy = size // 2
    cx = pad
    for color in inputs:
        painter.setBrush(QtGui.QColor(*color))
        painter.setPen(QtGui.QPen(QtGui.QColor(*(min(255, c + 60) for c in color)), 1))
        painter.drawEllipse(cx, cy - dot_r, dot_r * 2, dot_r * 2)
        cx += dot_r * 2 + gap

    if has_div:
        cx += div_gap - gap
        painter.setPen(QtGui.QPen(QtGui.QColor(160, 160, 160, 200), div_w))
        painter.drawLine(cx, cy - dot_r - 1, cx, cy + dot_r + 1)
        cx += div_w + div_gap

    for color in outputs:
        painter.setBrush(QtGui.QColor(*color))
        painter.setPen(QtGui.QPen(QtGui.QColor(*(min(255, c + 60) for c in color)), 1))
        painter.drawEllipse(cx, cy - dot_r, dot_r * 2, dot_r * 2)
        cx += dot_r * 2 + gap

    painter.end()
    return QtGui.QIcon(pixmap)


class _BaseNodeTreeItem(QtWidgets.QTreeWidgetItem):

    def __eq__(self, other):
        """
        Workaround fix for QTreeWidgetItem "operator not implemented error".
        see link: https://bugreports.qt.io/browse/PYSIDE-74
        """
        return id(self) == id(other)


class NodesTreeWidget(QtWidgets.QTreeWidget):
    """
    The :class:`NodeGraphQt.NodesTreeWidget` displays all registered nodes and
    lets users personalise the layout:

      - ★ Favorites  — right-click any node → "Add to Favorites"
      - Custom groups — right-click any node → "Add to Group / New Group…"
      - Rename        — right-click any category header → "Rename…"
      - Reorder       — right-click any standard category → "Move Up / Down"

    Layout is persisted to *node_layout.json* next to the app (dev) or in the
    platform user-data directory (frozen build).

    Args:
        parent (QtWidgets.QWidget): parent widget.
        node_graph (NodeGraphQt.NodeGraph): node graph.
        layout_path (str | Path | None): override path for node_layout.json.
    """

    def __init__(self, parent=None, node_graph=None, layout_path=None):
        super().__init__(parent)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)
        self.setHeaderHidden(True)
        self.setWindowTitle('Nodes')

        self._factory = node_graph.node_factory if node_graph else None

        # App-default labels / order (set by main.py via public API)
        self._custom_labels: dict = {}   # cat_id → display label (app default)
        self._category_order: list = []  # default category order

        # User customisations (loaded from / saved to node_layout.json)
        self._user_labels:  dict = {}    # cat_id → user-override display label
        self._user_order:   list = []    # full user-defined top-level cat order ([] = use default)
        self._favorites:    list = []    # list of node type IDs
        self._custom_groups: dict = {}   # {group_name: [node_id, …]}

        self._layout_path: Path | None = Path(layout_path) if layout_path else None
        self._category_items: dict = {}  # cat_id → QTreeWidgetItem
        self._search_data:   dict = {}   # node_id → lowercase searchable string

        # ── Search bar (lives in the top viewport margin) ─────────────────────
        self._search_bar = QtWidgets.QLineEdit(self)
        self._search_bar.setPlaceholderText('Search nodes…')
        self._search_bar.setClearButtonEnabled(True)
        self._search_bar.textChanged.connect(self._apply_filter)
        self._layout_search_bar()

        self._load_layout()
        self._build_tree()

    # ── Search ────────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_search_bar()

    def showEvent(self, event):
        super().showEvent(event)
        self._layout_search_bar()

    def _layout_search_bar(self):
        """Keep search bar and viewport margin in sync for any style/font scale."""
        pad = 4
        search_h = max(self._search_bar.sizeHint().height(), 28)
        self._search_bar.setGeometry(pad, pad, max(0, self.width() - pad * 2), search_h)
        self.setViewportMargins(0, search_h + pad * 2, 0, 0)

    def _apply_filter(self, text: str):
        q = text.strip().lower()

        if not q:
            # Restore all items
            it = QtWidgets.QTreeWidgetItemIterator(self)
            while (item := it.value()):
                item.setHidden(False)
                it += 1
            for cat_item in self._category_items.values():
                cat_item.setExpanded(True)
            return

        # Pass 1: show/hide node items by match
        it = QtWidgets.QTreeWidgetItemIterator(self)
        while (item := it.value()):
            if item.type() == TYPE_NODE:
                node_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                item.setHidden(q not in self._search_data.get(node_id, ''))
            it += 1

        # Pass 2: show/hide categories bottom-up (children before parents).
        # Reversing the pre-order list gives post-order so parent visibility
        # is decided after all its children are already updated.
        # Count both TYPE_NODE and TYPE_CATEGORY children so nested groups
        # (e.g. plugins.Plugins → confocal/filopodia) propagate correctly.
        it = QtWidgets.QTreeWidgetItemIterator(self)
        all_cat_items = []
        while (item := it.value()):
            if item.type() == TYPE_CATEGORY:
                all_cat_items.append(item)
            it += 1
        for item in reversed(all_cat_items):
            has_vis = any(
                not item.child(i).isHidden()
                and item.child(i).type() in (TYPE_NODE, TYPE_CATEGORY)
                for i in range(item.childCount())
            )
            item.setHidden(not has_vis)
            if has_vis:
                item.setExpanded(True)

    # ── Layout persistence ────────────────────────────────────────────────────

    def _get_layout_path(self) -> Path:
        if self._layout_path:
            return self._layout_path
        system = platform.system()
        if system == 'Darwin':
            base = Path.home() / 'Library' / 'Application Support' / 'Synapse'
        elif system == 'Windows':
            base = Path(os.environ.get('APPDATA', str(Path.home()))) / 'Synapse'
        else:
            base = Path.home() / '.synapse'
        return base / 'node_layout.json'

    def _load_layout(self):
        try:
            with open(self._get_layout_path()) as f:
                data = json.load(f)
            self._favorites     = data.get('favorites', [])
            self._custom_groups = data.get('custom_groups', {})
            self._user_labels   = data.get('user_category_labels', {})
            self._user_order    = data.get('user_category_order', [])
        except Exception:
            pass

    def _save_layout(self):
        path = self._get_layout_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump({
                    'favorites':            self._favorites,
                    'custom_groups':        self._custom_groups,
                    'user_category_labels': self._user_labels,
                    'user_category_order':  self._user_order,
                }, f, indent=2)
        except Exception as exc:
            print(f'[NodesTree] Could not save layout: {exc}')

    # ── Label resolution ──────────────────────────────────────────────────────

    def _resolved_label(self, cat_id: str) -> str:
        """User override > app default > last identifier segment."""
        return (self._user_labels.get(cat_id)
                or self._custom_labels.get(cat_id)
                or cat_id.split('.')[-1])

    # ── Tree construction ─────────────────────────────────────────────────────

    def _build_tree(self):
        try:
            from custom_nodes import PORT_COLORS
        except ImportError:
            PORT_COLORS = {}

        self.clear()
        self._category_items = {}
        self._search_data = {}

        # ── Collect all registered node types ────────────────────────────────
        node_types: dict = {}    # node_id → node_name
        all_std_cats: set = set()

        if self._factory:
            for name, node_ids in self._factory.names.items():
                for nid in node_ids:
                    cat = '.'.join(nid.split('.')[:-1])
                    all_std_cats.add(cat)
                    node_types[nid] = name
                    # Ensure parent categories are registered too
                    parts = cat.split('.')
                    while len(parts) > 2:
                        parts.pop()
                        all_std_cats.add('.'.join(parts))

        # ── 1. Favorites ──────────────────────────────────────────────────────
        fav_item = self._make_cat_item('★  Favorites', _CAT_FAVORITES, bold=True)
        self.addTopLevelItem(fav_item)
        self._category_items[_CAT_FAVORITES] = fav_item

        valid_favs = [nid for nid in self._favorites if nid in node_types]
        if valid_favs:
            for nid in valid_favs:
                self._add_node_item(fav_item, nid, node_types[nid], PORT_COLORS)
        else:
            hint = _BaseNodeTreeItem(None, ['Right-click any node → Add to Favorites'])
            hint.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            hint.setForeground(0, QtGui.QColor('#777'))
            fav_item.addChild(hint)
        fav_item.setExpanded(True)

        # ── 2. Custom groups ──────────────────────────────────────────────────
        for group_name, node_ids in self._custom_groups.items():
            gkey = _CAT_GROUP_PFX + group_name
            grp_item = self._make_cat_item(f'  {group_name}', gkey, bold=True)
            self.addTopLevelItem(grp_item)
            self._category_items[gkey] = grp_item
            for nid in node_ids:
                if nid in node_types:
                    self._add_node_item(grp_item, nid, node_types[nid], PORT_COLORS)
            grp_item.setExpanded(True)

        # ── 3. Standard categories (respects user / default order) ────────────
        def _std_cat_sort(cat):
            if self._user_order:
                return (0, self._user_order.index(cat)) if cat in self._user_order else (1, cat)
            if cat in self._category_order:
                return (0, self._category_order.index(cat))
            return (1, cat)

        for category in sorted(all_std_cats, key=_std_cat_sort):
            parts = category.split('.')
            if len(parts) > 2:
                parent_path = '.'.join(parts[:-1])
                parent_item = self._category_items.get(parent_path)
            else:
                parent_item = None

            label    = self._resolved_label(category)
            cat_item = self._make_cat_item(label, category)

            if parent_item is None:
                self.addTopLevelItem(cat_item)
            else:
                parent_item.addChild(cat_item)

            cat_item.setExpanded(True)
            self._category_items[category] = cat_item

        # ── 4. Place node items into standard categories ───────────────────────
        import inspect
        for node_id, node_name in sorted(node_types.items(), key=lambda kv: kv[1]):
            category = '.'.join(node_id.split('.')[:-1])
            cat_item = self._category_items.get(category)
            if cat_item:
                self._add_node_item(cat_item, node_id, node_name, PORT_COLORS)

        # Re-apply active search filter after rebuild
        if hasattr(self, '_search_bar'):
            self._apply_filter(self._search_bar.text())

    @staticmethod
    def _make_cat_item(label: str, cat_id: str, bold: bool = False) -> '_BaseNodeTreeItem':
        item = _BaseNodeTreeItem(None, [label], type=TYPE_CATEGORY)
        item.setFirstColumnSpanned(True)
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        item.setSizeHint(0, QtCore.QSize(100, 26))
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, cat_id)
        if bold:
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
        return item

    def _add_node_item(self, parent_item, node_id: str, node_name: str,
                       PORT_COLORS: dict) -> '_BaseNodeTreeItem':
        import inspect
        item = _BaseNodeTreeItem(None, [node_name], type=TYPE_NODE)
        item.setSizeHint(0, QtCore.QSize(100, 26))
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, node_id)

        if self._factory:
            node_cls = self._factory.nodes.get(node_id)
            doc = inspect.getdoc(node_cls) if node_cls else None
            item.setToolTip(0, doc.split('\n')[0] if doc else node_id)
            self._search_data[node_id] = (node_name + ' ' + (doc or '')).lower()
            if PORT_COLORS:
                port_spec = getattr(node_cls, 'PORT_SPEC', None) if node_cls else None
                if port_spec:
                    in_colors  = [PORT_COLORS.get(k, (95, 106, 106)) for k in port_spec.get('inputs',  [])]
                    out_colors = [PORT_COLORS.get(k, (95, 106, 106)) for k in port_spec.get('outputs', [])]
                    item.setIcon(0, _make_port_icon(in_colors, out_colors))
        else:
            item.setToolTip(0, node_id)
            self._search_data[node_id] = node_name.lower()

        parent_item.addChild(item)
        return item

    # ── Drag-and-drop (node → canvas) ────────────────────────────────────────

    def mimeData(self, items):
        node_ids = ['node:{}'.format(i.data(0, QtCore.Qt.ItemDataRole.UserRole))
                    for i in items if i.data(0, QtCore.Qt.ItemDataRole.UserRole)]
        node_urn  = URN_SCHEME + ';'.join(node_ids)
        mime_data = QtCore.QMimeData()
        mime_data.setData(MIME_TYPE, QtCore.QByteArray(node_urn.encode()))
        return mime_data

    # ── Right-click context menu ──────────────────────────────────────────────

    def contextMenuEvent(self, event):
        item   = self.itemAt(event.pos())
        menu   = QtWidgets.QMenu(self)

        if item is None:
            menu.addAction('New Group…', self._on_new_group)

        elif item.type() == TYPE_NODE:
            node_id   = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            parent_id = (item.parent().data(0, QtCore.Qt.ItemDataRole.UserRole)
                         if item.parent() else None)

            # Favorites toggle
            if node_id in self._favorites:
                menu.addAction('☆  Remove from Favorites',
                               lambda: self._toggle_favorite(node_id))
            else:
                menu.addAction('★  Add to Favorites',
                               lambda: self._toggle_favorite(node_id))

            # Groups submenu
            menu.addSeparator()
            grp_sub = menu.addMenu('Add to Group')
            for g in self._custom_groups:
                if node_id not in self._custom_groups[g]:
                    grp_sub.addAction(g, lambda g=g: self._add_to_group(node_id, g))
            if self._custom_groups:
                grp_sub.addSeparator()
            grp_sub.addAction('New Group…', lambda: self._new_group_with_node(node_id))

            # Remove from current group / favorites
            if parent_id and parent_id.startswith(_CAT_GROUP_PFX):
                group_name = parent_id[len(_CAT_GROUP_PFX):]
                menu.addSeparator()
                menu.addAction(f'Remove from "{group_name}"',
                               lambda: self._remove_from_group(node_id, group_name))
            elif parent_id == _CAT_FAVORITES:
                # Already handled by the toggle at the top, but add explicit remove too
                pass

        elif item.type() == TYPE_CATEGORY:
            cat_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)

            if cat_id == _CAT_FAVORITES:
                pass  # No actions on the Favorites header

            elif cat_id.startswith(_CAT_GROUP_PFX):
                group_name = cat_id[len(_CAT_GROUP_PFX):]
                menu.addAction('Rename…',
                               lambda: self._rename_custom_group(group_name))
                menu.addSeparator()
                menu.addAction('Delete Group',
                               lambda: self._delete_custom_group(group_name))

            else:
                # Standard category
                menu.addAction('Rename…', lambda: self._rename_category(cat_id))
                if cat_id in self._user_labels:
                    menu.addAction('Reset Name', lambda: self._reset_cat_name(cat_id))
                # Only top-level categories are moveable
                if len(cat_id.split('.')) <= 2:
                    menu.addSeparator()
                    menu.addAction('Move Up',   lambda: self._move_category(cat_id, -1))
                    menu.addAction('Move Down', lambda: self._move_category(cat_id, +1))

            menu.addSeparator()
            menu.addAction('New Group…', self._on_new_group)

        if not menu.isEmpty():
            menu.exec(event.globalPos())

    # ── Favorites ─────────────────────────────────────────────────────────────

    def _toggle_favorite(self, node_id: str):
        if node_id in self._favorites:
            self._favorites.remove(node_id)
        else:
            self._favorites.append(node_id)
        self._save_layout()
        self._build_tree()

    # ── Custom groups ──────────────────────────────────────────────────────────

    def _on_new_group(self):
        self._new_group_with_node(None)

    def _new_group_with_node(self, node_id):
        name, ok = QtWidgets.QInputDialog.getText(
            self, 'New Group', 'Group name:')
        if not ok or not name.strip():
            return
        name = name.strip()
        if name not in self._custom_groups:
            self._custom_groups[name] = []
        if node_id and node_id not in self._custom_groups[name]:
            self._custom_groups[name].append(node_id)
        self._save_layout()
        self._build_tree()

    def _add_to_group(self, node_id: str, group_name: str):
        self._custom_groups.setdefault(group_name, [])
        if node_id not in self._custom_groups[group_name]:
            self._custom_groups[group_name].append(node_id)
        self._save_layout()
        self._build_tree()

    def _remove_from_group(self, node_id: str, group_name: str):
        if group_name in self._custom_groups:
            self._custom_groups[group_name] = [
                n for n in self._custom_groups[group_name] if n != node_id
            ]
        self._save_layout()
        self._build_tree()

    def _rename_custom_group(self, old_name: str):
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, 'Rename Group', 'New name:', text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        self._custom_groups[new_name] = self._custom_groups.pop(old_name)
        self._save_layout()
        self._build_tree()

    def _delete_custom_group(self, group_name: str):
        reply = QtWidgets.QMessageBox.question(
            self, 'Delete Group',
            f'Delete group "{group_name}"?\n(Nodes are not deleted.)',
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._custom_groups.pop(group_name, None)
        self._save_layout()
        self._build_tree()

    # ── Category rename / reorder ─────────────────────────────────────────────

    def _rename_category(self, cat_id: str):
        current = self._resolved_label(cat_id)
        new_label, ok = QtWidgets.QInputDialog.getText(
            self, 'Rename Category', 'Display name:', text=current)
        if not ok or not new_label.strip():
            return
        self._user_labels[cat_id] = new_label.strip()
        self._save_layout()
        self._build_tree()

    def _reset_cat_name(self, cat_id: str):
        self._user_labels.pop(cat_id, None)
        self._save_layout()
        self._build_tree()

    def _move_category(self, cat_id: str, direction: int):
        """Move a top-level standard category up (-1) or down (+1)."""
        if not self._user_order:
            # Initialise from the current computed order
            def _sort_key(cat):
                if cat in self._category_order:
                    return (0, self._category_order.index(cat))
                return (1, cat)
            self._user_order = sorted(
                [c for c in self._category_items
                 if c != _CAT_FAVORITES
                 and not c.startswith(_CAT_GROUP_PFX)
                 and len(c.split('.')) <= 2],
                key=_sort_key)

        if cat_id not in self._user_order:
            return
        idx     = self._user_order.index(cat_id)
        new_idx = idx + direction
        if not (0 <= new_idx < len(self._user_order)):
            return
        self._user_order[idx], self._user_order[new_idx] = \
            self._user_order[new_idx], self._user_order[idx]
        self._save_layout()
        self._build_tree()

    # ── Public API (backward-compatible) ─────────────────────────────────────

    def __repr__(self):
        return '<{} object at {}>'.format(self.__class__.__name__, hex(id(self)))

    def _set_node_factory(self, factory):
        self._factory = factory

    def set_category_label(self, category, label):
        """Set the app-default display label (user renames override this)."""
        self._custom_labels[category] = label
        # Update live only if the user hasn't set their own label
        if category not in self._user_labels and category in self._category_items:
            self._category_items[category].setText(0, label)

    def set_category_order(self, order):
        """Set the app-default category order (user reordering overrides this)."""
        self._category_order = order
        if not self._user_order:
            self._build_tree()

    def update(self):
        self._build_tree()
