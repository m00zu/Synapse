"""
plugin_loader.py
================
Dynamic plugin system for Synapse.

Users drop a .py file into the plugin directory and new nodes appear in the
Node Explorer on the next launch — no recompilation required.

Plugin directory locations
--------------------------
  Development (not frozen):   ./plugins/   (next to this file)
  macOS frozen .app:          ~/Library/Application Support/Synapse/plugins/
  Windows frozen:             %APPDATA%\\Synapse\\plugins\\
  Linux frozen:               ~/.synapse/plugins/

Plugin file requirements
------------------------
Each .py file in the plugin directory may define one or more node classes.
A class is auto-registered if it:
  1. Inherits from nodes.base.BaseExecutionNode
  2. Has __identifier__ starting with 'plugins.'
  3. Has a NODE_NAME attribute

Quick example::

    from nodes.base import BaseExecutionNode, PORT_COLORS
    from data_models import ImageData

    class MyNode(BaseExecutionNode):
        \"\"\"One-line description shown in the AI catalog.\"\"\"
        __identifier__ = 'plugins.myplugin'
        NODE_NAME      = 'My Node'
        PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

        def __init__(self):
            super().__init__()
            self.add_input('image',  color=PORT_COLORS['image'])
            self.add_output('image', color=PORT_COLORS['image'])

        def evaluate(self):
            ...
            self.mark_clean()
            return True, None
"""

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME        = "Synapse"
_PLUGIN_CLASSES: list[type] = []   # accumulated across load_plugins() calls



# ---------------------------------------------------------------------------
# Plugin directory
# ---------------------------------------------------------------------------

def get_plugin_dir() -> Path:
    """Return the platform-appropriate plugin directory (created if absent).

    Frozen (Nuitka onefile):
      1. Check for plugins/ next to the real .exe (portable install on D:/ etc.)
      2. Fall back to persistent OS user directory (%APPDATA%, ~/Library/..., etc.)
      Note: sys.executable points to the temp extraction dir in onefile mode,
      so we use sys.argv[0] to find the actual .exe location.

    Dev mode:
      synapse/plugins/ (bundled built-in plugins)
    """
    if getattr(sys, 'frozen', False):
        # 1. Portable: plugins/ next to the real executable
        real_exe = Path(sys.argv[0]).resolve()
        portable = real_exe.parent / 'plugins'
        if portable.exists():
            return portable

        # 2. Persistent OS user directory
        system = platform.system()
        if system == 'Darwin':
            base = Path.home() / 'Library' / 'Application Support' / APP_NAME
        elif system == 'Windows':
            base = Path(os.environ.get('APPDATA', str(Path.home()))) / APP_NAME
        else:
            base = Path.home() / f'.{APP_NAME.lower()}'
    else:
        # Dev mode: bundled plugins are inside synapse/plugins/
        bundled = Path(__file__).parent / 'plugins'
        if bundled.exists():
            return bundled
        # Fallback: project root plugins/
        base = Path(__file__).parent.parent

    plugin_dir = base / 'plugins'
    plugin_dir.mkdir(parents=True, exist_ok=True)
    return plugin_dir


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_plugins(graph) -> list[dict]:
    """Scan the plugin directory, import modules, register node classes.

    Supports two plugin formats:

    * **Flat file** – a single ``.py`` file in the plugin directory.
    * **Package directory** – a sub-folder containing ``__init__.py``.
      If the folder also contains a ``vendor/`` sub-directory, that path is
      prepended to ``sys.path`` before the package is imported, so vendored
      third-party wheels (e.g. ``vendor/rdkit/``) are importable without any
      system-level installation.

    Returns a list of result dicts::

        [{'file': 'my_nodes.py', 'nodes': ['MyNode'], 'error': None}, ...]

    Errors are captured per-file so one bad plugin does not block the others.
    """
    from .nodes.base import BaseExecutionNode  # imported here to avoid circular imports

    plugin_dir = get_plugin_dir()
    results: list[dict] = []

    # Read disabled built-in packages
    disabled_path = plugin_dir / '_disabled.json'
    try:
        disabled: set[str] = set(
            json.loads(disabled_path.read_text()) if disabled_path.exists() else []
        )
    except Exception:
        disabled = set()

    loaded_types = set(graph.node_factory.nodes.keys())
    loaded_plugin_types = {cls.type_ for cls in _PLUGIN_CLASSES}

    def _register_from_mod(mod, registered, error):
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            _LIBRARY_IDENTIFIERS = {'nodeGraphQt.nodes', 'io.jchanvfx.github', 'io.github.jchanvfx'}
            if not (
                isinstance(obj, type)
                and issubclass(obj, BaseExecutionNode)
                and obj is not BaseExecutionNode
                and getattr(obj, '__identifier__', '') not in _LIBRARY_IDENTIFIERS
                and hasattr(obj, 'NODE_NAME')
                and obj.NODE_NAME != 'Node'   # library default — never a real node
            ):
                continue
            if obj.type_ in loaded_types:
                continue
            try:
                graph.register_node(obj)
                loaded_types.add(obj.type_)
                if obj.type_ not in loaded_plugin_types:
                    _PLUGIN_CLASSES.append(obj)
                    loaded_plugin_types.add(obj.type_)
                registered.append(obj.NODE_NAME)
            except Exception as reg_err:
                fragment = f"{attr_name}: {reg_err}"
                error = (error + '; ' + fragment) if error else fragment
        return registered, error

    # ── Flat .py plugins ─────────────────────────────────────────────────────
    for py_file in sorted(plugin_dir.glob('*.py')):
        if py_file.name.startswith('_'):
            continue

        if py_file.stem in disabled:
            results.append({'file': py_file.name, 'nodes': [], 'error': None,
                            'disabled': True})
            continue

        registered: list[str] = []
        error: str | None = None

        module_name = f'_plugin_{py_file.stem}'
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            registered, error = _register_from_mod(mod, registered, error)
        except Exception as exc:
            error = str(exc)

        results.append({'file': py_file.name, 'nodes': registered, 'error': error,
                        'disabled': False})

    # ── Package directory plugins (folder/__init__.py + optional vendor/) ────
    for pkg_dir in sorted(plugin_dir.iterdir()):
        if not pkg_dir.is_dir() or pkg_dir.name.startswith('_'):
            continue
        init_file = pkg_dir / '__init__.py'
        if not init_file.exists():
            continue

        # Skip disabled packages
        if pkg_dir.name in disabled:
            results.append({'file': pkg_dir.name + '/', 'nodes': [], 'error': None,
                            'disabled': True})
            continue

        registered = []
        error = None

        # Prepend vendor/ to sys.path so the package can import bundled wheels
        vendor_dir = pkg_dir / 'vendor'
        if vendor_dir.is_dir() and str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))

        module_name = f'_plugin_pkg_{pkg_dir.name}'
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, init_file,
                submodule_search_locations=[str(pkg_dir)],
            )
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = module_name   # enable relative imports (from .foo)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            registered, error = _register_from_mod(mod, registered, error)
        except Exception as exc:
            error = str(exc)

        results.append({'file': pkg_dir.name + '/', 'nodes': registered, 'error': error,
                        'disabled': False})

    return results


# ---------------------------------------------------------------------------
# LLM catalog integration
# ---------------------------------------------------------------------------

def get_plugin_catalog_entries() -> list[dict]:
    """Return minimal LLM-catalog entries for all loaded plugin classes.

    Each entry is compatible with the format used by
    ``llm_assistant.build_condensed_catalog()``.
    """
    entries = []
    for cls in _PLUGIN_CLASSES:
        port_spec = getattr(cls, 'PORT_SPEC', {'inputs': [], 'outputs': []})
        raw_doc   = (cls.__doc__ or cls.NODE_NAME or '').strip()
        description = raw_doc.split('\n')[0]
        entries.append({
            'class_name':  cls.__name__,
            'description': description,
            'inputs':      [{'name': p} for p in port_spec.get('inputs',  [])],
            'outputs':     [{'name': p} for p in port_spec.get('outputs', [])],
        })
    return entries


# ---------------------------------------------------------------------------
# Install helper (shared by dialog and main-window quick-install action)
# ---------------------------------------------------------------------------

def install_plugin_file(src: Path, plugin_dir: Path,
                        overwrite: bool = False) -> tuple[bool, str]:
    """Copy *src* into *plugin_dir*.

    Returns ``(success, message)``.
    """
    import shutil
    dst = plugin_dir / src.name
    if dst.exists() and not overwrite:
        return False, f"'{src.name}' already exists in the plugin directory."
    try:
        shutil.copy2(str(src), str(dst))
        return True, f"'{src.name}' installed successfully."
    except Exception as exc:
        return False, str(exc)


def install_synpkg(src_pkg: Path, plugin_dir: Path,
                   overwrite: bool = False) -> tuple[bool, str]:
    """Extract a ``.synpkg`` archive (zstd-compressed tar) into *plugin_dir*.

    Returns ``(success, message)``.
    """
    import io
    import shutil
    import tarfile

    try:
        import zstandard as zstd
    except ImportError:
        return False, (
            "zstandard is not installed.\n"
            "pip install zstandard"
        )

    try:
        with open(src_pkg, 'rb') as f:
            raw = zstd.ZstdDecompressor().decompress(
                f.read(), max_output_size=500_000_000
            )

        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            names = tar.getnames()
            if not names:
                return False, "Empty archive."
            top_dirs = {n.split('/')[0] for n in names if '/' in n}
            if not top_dirs:
                return False, "Invalid plugin package: no top-level directory."
            if len(top_dirs) > 1:
                return False, (
                    f"Ambiguous package: multiple top-level directories: "
                    f"{', '.join(sorted(top_dirs))}. Expected exactly one."
                )
            pkg_name = top_dirs.pop()
            dst = plugin_dir / pkg_name
            if dst.exists() and not overwrite:
                return False, f"'{pkg_name}' is already installed. Uninstall it first."
            if dst.exists():
                shutil.rmtree(str(dst))
            tar.extractall(str(plugin_dir))
        return True, f"'{pkg_name}' installed successfully."
    except Exception as exc:
        return False, str(exc)


def install_plugin_package(src_zip: Path, plugin_dir: Path,
                           overwrite: bool = False) -> tuple[bool, str]:
    """Extract a plugin package ``.zip`` into *plugin_dir*.

    The zip must contain a single top-level directory (the package name), e.g.::

        rdkit_nodes/
            __init__.py
            vendor/
                rdkit/   ← extracted wheel contents

    Returns ``(success, message)``.
    """
    import shutil
    import zipfile

    try:
        with zipfile.ZipFile(src_zip) as zf:
            top_dirs = {
                n.split('/')[0] for n in zf.namelist()
                if '/' in n and not n.split('/')[0].startswith('_')
            }
            if not top_dirs:
                return False, "Invalid plugin package: zip contains no top-level directory."
            if len(top_dirs) > 1:
                return False, (
                    f"Ambiguous plugin package: zip has multiple top-level directories: "
                    f"{', '.join(sorted(top_dirs))}. Expected exactly one."
                )
            pkg_name = top_dirs.pop()
            dst = plugin_dir / pkg_name
            if dst.exists() and not overwrite:
                return False, f"'{pkg_name}' is already installed. Uninstall it first."
            if dst.exists():
                shutil.rmtree(str(dst))
            zf.extractall(str(plugin_dir))
        return True, f"'{pkg_name}' installed successfully."
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Plugin Manager dialog
# ---------------------------------------------------------------------------

class PluginManagerDialog(QtWidgets.QDialog):
    """Full plugin management UI: install, status, uninstall, and browse online."""

    # Columns — Installed tab
    _COL_FILE   = 0
    _COL_NODES  = 1
    _COL_STATUS = 2
    _COL_ACTION = 3

    # Columns — Browse tab
    _BR_COL_NAME   = 0
    _BR_COL_SIZE   = 1
    _BR_COL_ACTION = 2

    _REPO = 'm00zu/Synapse-Plugins'
    _METADATA_URL = f'https://raw.githubusercontent.com/{_REPO}/main/plugins.json'

    def __init__(self, plugin_results: list[dict], plugin_dir: Path,
                 parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Plugin Manager")
        self.setMinimumSize(750, 440)
        self.resize(850, 500)

        self._plugin_dir     = plugin_dir
        self._plugin_results = list(plugin_results)   # snapshot from startup load
        self._download_threads: list = []              # keep references to prevent GC

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        # --- Reload banner (hidden until a change is made) ---
        self._restart_banner = QtWidgets.QLabel(
            "  Use View \u2192 Reload Plugins to apply changes, or restart the application."
        )
        self._restart_banner.setStyleSheet(
            "background:#b7950b; color:#fff; padding:5px 8px; border-radius:3px;"
        )
        self._restart_banner.hide()
        layout.addWidget(self._restart_banner)

        # --- Tab widget ---
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs)

        # ═══ Tab 1: Installed ═══
        installed_page = QtWidgets.QWidget()
        installed_layout = QtWidgets.QVBoxLayout(installed_page)
        installed_layout.setContentsMargins(0, 6, 0, 0)

        # Header row
        dir_label = QtWidgets.QLabel(f"<b>Plugin directory:</b>  {plugin_dir}")
        dir_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        open_btn = QtWidgets.QPushButton("Open Folder")
        open_btn.setFixedWidth(100)
        open_btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(plugin_dir))
        ))
        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(dir_label, 1)
        header_row.addWidget(open_btn)
        installed_layout.addLayout(header_row)

        # Table
        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["File", "Nodes Loaded", "Status", ""])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(self._COL_FILE,   QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_NODES,  QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self._COL_STATUS, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_ACTION, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self._COL_ACTION, 180)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.verticalHeader().hide()
        installed_layout.addWidget(self._table)

        # Empty-state label
        self._empty_label = QtWidgets.QLabel(
            "No plugins installed yet.\nGo to the Browse tab to download plugins, "
            "or click 'Install from File' below."
        )
        self._empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color:#888; padding:24px;")
        self._empty_label.hide()
        installed_layout.addWidget(self._empty_label)

        # Bottom row
        btn_row = QtWidgets.QHBoxLayout()
        install_btn = QtWidgets.QPushButton("Install from File...")
        install_btn.clicked.connect(self._on_install)
        btn_row.addWidget(install_btn)
        btn_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        installed_layout.addLayout(btn_row)

        self._tabs.addTab(installed_page, "Installed")

        # ═══ Tab 2: Browse Online ═══
        browse_page = QtWidgets.QWidget()
        browse_layout = QtWidgets.QVBoxLayout(browse_page)
        browse_layout.setContentsMargins(0, 6, 0, 0)

        # Status / info label
        self._browse_status = QtWidgets.QLabel("Click 'Refresh' to fetch available plugins.")
        self._browse_status.setStyleSheet("color:#aaa; padding:4px;")
        browse_layout.addWidget(self._browse_status)

        # Browse table
        self._browse_table = QtWidgets.QTableWidget(0, 3)
        self._browse_table.setHorizontalHeaderLabels(["Plugin", "Size", ""])
        bhdr = self._browse_table.horizontalHeader()
        bhdr.setSectionResizeMode(self._BR_COL_NAME,   QtWidgets.QHeaderView.ResizeMode.Stretch)
        bhdr.setSectionResizeMode(self._BR_COL_SIZE,   QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        bhdr.setSectionResizeMode(self._BR_COL_ACTION, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._browse_table.setColumnWidth(self._BR_COL_ACTION, 120)
        self._browse_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._browse_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._browse_table.setAlternatingRowColors(True)
        self._browse_table.setWordWrap(True)
        self._browse_table.verticalHeader().hide()
        browse_layout.addWidget(self._browse_table)

        # Bottom row
        br_btn_row = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._on_refresh_store)
        br_btn_row.addWidget(refresh_btn)
        br_btn_row.addStretch()
        close_btn2 = QtWidgets.QPushButton("Close")
        close_btn2.setFixedWidth(80)
        close_btn2.clicked.connect(self.accept)
        br_btn_row.addWidget(close_btn2)
        browse_layout.addLayout(br_btn_row)

        self._tabs.addTab(browse_page, "Browse Online")

        self._rebuild_table()
        self._store_assets: list[dict] = []  # cached from last refresh

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _table_rows(self) -> list[dict]:
        """Merge on-disk files with startup load results into unified row data."""
        loaded = {r['file']: r for r in self._plugin_results}

        disk_files: list[str] = sorted(
            f.name for f in self._plugin_dir.glob('*.py')
            if not f.name.startswith('_')
        )
        # Also include package directories (sub-folders with __init__.py)
        for d in sorted(self._plugin_dir.iterdir()):
            if d.is_dir() and not d.name.startswith('_') and (d / '__init__.py').exists():
                disk_files.append(d.name + '/')
        disk_files.sort()

        rows = []
        for fname in disk_files:
            if fname in loaded:
                r = loaded[fname]
                disabled = r.get('disabled', False)
                if disabled:
                    kind = 'disabled'
                elif r['error']:
                    kind = 'error'
                elif r['nodes']:
                    kind = 'ok'
                else:
                    kind = 'no_nodes'
            else:
                kind = 'pending'   # installed after startup, not yet loaded
                disabled = False
            rows.append({
                'file':     fname,
                'nodes':    loaded[fname]['nodes'] if fname in loaded else [],
                'error':    loaded[fname]['error'] if fname in loaded else None,
                'kind':     kind,
                'disabled': disabled,
            })
        return rows

    def _rebuild_table(self):
        rows = self._table_rows()
        self._table.setRowCount(0)

        if not rows:
            self._table.hide()
            self._empty_label.show()
            return

        self._empty_label.hide()
        self._table.show()

        for row in rows:
            i = self._table.rowCount()
            self._table.insertRow(i)

            self._table.setItem(i, self._COL_FILE,
                                QtWidgets.QTableWidgetItem(row['file']))

            nodes_text = ', '.join(row['nodes']) or '—'
            nodes_item = QtWidgets.QTableWidgetItem(nodes_text)
            if row['nodes']:
                nodes_item.setToolTip('\n'.join(row['nodes']))
            self._table.setItem(i, self._COL_NODES, nodes_item)

            kind = row['kind']
            if kind == 'ok':
                status_text  = 'Active'
                status_color = '#2ecc71'
            elif kind == 'disabled':
                status_text  = 'Disabled'
                status_color = '#888888'
            elif kind == 'error':
                status_text  = f"Error: {row['error']}"
                status_color = '#e74c3c'
            elif kind == 'no_nodes':
                status_text  = 'No nodes found'
                status_color = '#f39c12'
            else:  # pending
                status_text  = 'Reload to activate'
                status_color = '#aaaaaa'

            status_item = QtWidgets.QTableWidgetItem(status_text)
            status_item.setForeground(QtGui.QColor(status_color))
            self._table.setItem(i, self._COL_STATUS, status_item)

            # Action cell: [Disable/Enable]  [Uninstall]
            fname = row['file']   # capture for closure
            cell = QtWidgets.QWidget()
            cell_layout = QtWidgets.QHBoxLayout(cell)
            cell_layout.setContentsMargins(2, 1, 2, 1)
            cell_layout.setSpacing(4)

            if row['disabled']:
                toggle_btn = QtWidgets.QPushButton("Enable")
                toggle_btn.setStyleSheet("color:#2ecc71; font-size:11px;")
                toggle_btn.clicked.connect(
                    lambda _c=False, f=fname: self._on_enable_plugin(f))
            else:
                toggle_btn = QtWidgets.QPushButton("Disable")
                toggle_btn.setStyleSheet("color:#e67e22; font-size:11px;")
                toggle_btn.clicked.connect(
                    lambda _c=False, f=fname: self._on_disable_plugin(f))

            uninstall_btn = QtWidgets.QPushButton("Uninstall")
            uninstall_btn.setStyleSheet("color:#e74c3c; font-size:11px;")
            uninstall_btn.clicked.connect(
                lambda _c=False, f=fname: self._on_uninstall(f))

            for btn in (toggle_btn, uninstall_btn):
                btn.setFixedHeight(26)
                cell_layout.addWidget(btn)

            self._table.setCellWidget(i, self._COL_ACTION, cell)

        self._table.resizeRowsToContents()

    def _show_reload_banner(self):
        self._restart_banner.show()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_install(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Plugin File(s)", str(Path.home()),
            "Plugin Files (*.py *.zip *.synpkg)"
        )
        if not paths:
            return

        any_installed = False
        for path_str in paths:
            src = Path(path_str)

            if src.suffix.lower() == '.synpkg':
                # .synpkg — zstd-compressed tar plugin package
                pkg_name = src.stem.split('-')[0]  # e.g. rdkit_nodes-darwin-... → rdkit_nodes
                dst = self._plugin_dir / pkg_name
                if dst.exists():
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Overwrite Plugin?",
                        f"'{pkg_name}' is already installed.\nOverwrite it?",
                        QtWidgets.QMessageBox.StandardButton.Yes |
                        QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                        continue
                ok, msg = install_synpkg(src, self._plugin_dir, overwrite=True)
            elif src.suffix.lower() == '.zip':
                # Plugin package zip — peek to find the package name for overwrite check
                import zipfile
                try:
                    with zipfile.ZipFile(src) as zf:
                        top_dirs = {
                            n.split('/')[0] for n in zf.namelist()
                            if '/' in n and not n.split('/')[0].startswith('_')
                        }
                    pkg_name = next(iter(top_dirs), src.stem)
                except Exception:
                    pkg_name = src.stem
                dst = self._plugin_dir / pkg_name
                if dst.exists():
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Overwrite Plugin?",
                        f"'{pkg_name}' is already installed.\nOverwrite it?",
                        QtWidgets.QMessageBox.StandardButton.Yes |
                        QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                        continue
                ok, msg = install_plugin_package(src, self._plugin_dir, overwrite=True)
            else:
                dst = self._plugin_dir / src.name
                if dst.exists():
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Overwrite Plugin?",
                        f"'{src.name}' is already installed.\nOverwrite it?",
                        QtWidgets.QMessageBox.StandardButton.Yes |
                        QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                        continue
                ok, msg = install_plugin_file(src, self._plugin_dir, overwrite=True)

            if not ok:
                QtWidgets.QMessageBox.warning(self, "Install Failed", msg)
            else:
                any_installed = True

        if any_installed:
            self._rebuild_table()
            self._show_reload_banner()

    def _on_uninstall(self, filename: str):
        reply = QtWidgets.QMessageBox.question(
            self,
            "Uninstall Plugin",
            f"Delete '{filename}'?\n\nThe file will be permanently removed. "
            f"Use Reload Plugins or restart to apply.",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        import shutil
        target = self._plugin_dir / filename.rstrip('/')
        try:
            if target.is_dir():
                shutil.rmtree(str(target))
            else:
                target.unlink()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Uninstall Failed", str(exc))
            return

        # Remove from snapshot so status doesn't linger
        self._plugin_results = [r for r in self._plugin_results if r['file'] != filename]
        self._rebuild_table()
        self._show_reload_banner()

    # ------------------------------------------------------------------
    # Disable / re-enable
    # ------------------------------------------------------------------

    def _read_disabled(self) -> list[str]:
        disabled_path = self._plugin_dir / '_disabled.json'
        try:
            return json.loads(disabled_path.read_text()) if disabled_path.exists() else []
        except Exception:
            return []

    def _write_disabled(self, disabled: list[str]):
        disabled_path = self._plugin_dir / '_disabled.json'
        try:
            disabled_path.write_text(json.dumps(disabled, indent=2))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not write _disabled.json:\n{exc}")

    def _on_disable_plugin(self, filename: str):
        pkg_name = filename.rstrip('/').removesuffix('.py')
        reply = QtWidgets.QMessageBox.question(
            self,
            "Disable Plugin",
            f"Disable '{pkg_name}'?\n\nNodes from this plugin will not load "
            f"until re-enabled. Use Reload Plugins or restart to apply.",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        disabled = self._read_disabled()
        if pkg_name not in disabled:
            disabled.append(pkg_name)
            self._write_disabled(disabled)
        for r in self._plugin_results:
            if r['file'] == filename:
                r['disabled'] = True
                break
        self._rebuild_table()
        self._show_reload_banner()

    def _on_enable_plugin(self, filename: str):
        pkg_name = filename.rstrip('/').removesuffix('.py')
        disabled = self._read_disabled()
        if pkg_name in disabled:
            disabled.remove(pkg_name)
            self._write_disabled(disabled)
        for r in self._plugin_results:
            if r['file'] == filename:
                r['disabled'] = False
                break
        self._rebuild_table()
        self._show_reload_banner()

    # ------------------------------------------------------------------
    # Browse Online (Plugin Store)
    # ------------------------------------------------------------------

    def _compatible_assets(self, assets: list[dict]) -> list[dict]:
        """Filter release assets to those compatible with this platform."""
        import platform as _plat
        os_name  = _plat.system().lower()     # darwin, windows, linux
        arch     = _plat.machine().lower()    # arm64, amd64, x86_64
        pyver    = f'cp{sys.version_info.major}{sys.version_info.minor}'

        # Build set of architectures this machine can run
        arch_set = {arch}
        if arch == 'x86_64':
            arch_set.add('amd64')
        elif arch == 'amd64':
            arch_set.add('x86_64')
        # Windows ARM64 can run AMD64 binaries via emulation
        if os_name == 'windows' and arch == 'arm64':
            arch_set.add('amd64')

        result = []
        for a in assets:
            name = a['name']
            if not name.endswith('.synpkg'):
                continue
            # "slim" packages are pure Python — always compatible
            if '-slim' in name:
                result.append(a)
                continue
            # Platform-specific: check os, arch, python version
            lower = name.lower()
            if os_name in lower and pyver in lower:
                if any(ar in lower for ar in arch_set):
                    result.append(a)

        return result

    def _installed_plugin_names(self) -> set[str]:
        """Return set of installed plugin directory/file names (without extension)."""
        names = set()
        for f in self._plugin_dir.glob('*.py'):
            if not f.name.startswith('_'):
                names.add(f.stem)
        for d in self._plugin_dir.iterdir():
            if d.is_dir() and not d.name.startswith('_') and (d / '__init__.py').exists():
                names.add(d.name)
        return names

    def _on_refresh_store(self):
        """Fetch available plugins from GitHub releases."""
        self._browse_status.setText("Fetching available plugins...")
        self._browse_status.setStyleSheet("color:#f0c040; padding:4px;")
        QtWidgets.QApplication.processEvents()

        import urllib.request

        # Fetch plugin metadata (descriptions, display names)
        self._plugin_metadata = {}
        try:
            req = urllib.request.Request(self._METADATA_URL)
            with urllib.request.urlopen(req, timeout=10) as resp:
                self._plugin_metadata = json.loads(resp.read().decode())
        except Exception:
            pass  # descriptions are optional, continue without them

        # Fetch latest release assets
        url = f'https://api.github.com/repos/{self._REPO}/releases/latest'
        try:
            req = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            self._browse_status.setText(f"Failed to fetch: {exc}")
            self._browse_status.setStyleSheet("color:#e74c3c; padding:4px;")
            return

        assets = data.get('assets', [])
        tag = data.get('tag_name', '')
        compatible = self._compatible_assets(assets)

        if not compatible:
            self._browse_status.setText(
                f"No compatible plugins found in {tag} for this platform."
            )
            self._browse_status.setStyleSheet("color:#e67e22; padding:4px;")
            return

        self._store_assets = compatible
        self._browse_status.setText(
            f"Found {len(compatible)} compatible plugin(s) from release {tag}"
        )
        self._browse_status.setStyleSheet("color:#2ecc71; padding:4px;")
        self._rebuild_browse_table()

    def _rebuild_browse_table(self):
        """Populate the browse table with available plugins."""
        self._browse_table.setRowCount(0)
        installed = self._installed_plugin_names()

        # Group assets by plugin name (strip platform/date suffix)
        for asset in self._store_assets:
            name = asset['name']
            # Extract plugin name: everything before the platform/slim suffix
            base = name.removesuffix('.synpkg')
            pkg_name = base.split('-slim')[0].split('-darwin-')[0].split(
                '-windows-')[0].split('-linux-')[0].split('-novendor')[0]
            size_mb  = asset.get('size', 0) / 1024 / 1024
            dl_url   = asset.get('browser_download_url', '')

            i = self._browse_table.rowCount()
            self._browse_table.insertRow(i)

            # Name + description from remote metadata
            meta = getattr(self, '_plugin_metadata', {}).get(pkg_name, {})
            display_name = meta.get('name', pkg_name)
            desc = meta.get('description', '')
            display = f"<b>{display_name}</b>"
            if desc:
                display += f"<br><span style='color:#aaa; font-size:11px;'>{desc}</span>"
            name_label = QtWidgets.QLabel(display)
            name_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            name_label.setContentsMargins(6, 4, 6, 4)
            self._browse_table.setCellWidget(i, self._BR_COL_NAME, name_label)

            # Size
            size_item = QtWidgets.QTableWidgetItem(f"{size_mb:.1f} MB")
            size_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self._browse_table.setItem(i, self._BR_COL_SIZE, size_item)

            # Action button
            is_installed = pkg_name in installed
            if is_installed:
                btn = QtWidgets.QPushButton("Update")
                btn.setStyleSheet("color:#f0c040; font-size:11px;")
            else:
                btn = QtWidgets.QPushButton("Install")
                btn.setStyleSheet("color:#2ecc71; font-size:11px;")
            btn.setFixedHeight(28)
            btn.clicked.connect(
                lambda _c=False, url=dl_url, fn=name: self._on_download_plugin(url, fn))
            btn_container = QtWidgets.QWidget()
            btn_lay = QtWidgets.QVBoxLayout(btn_container)
            btn_lay.setContentsMargins(4, 0, 4, 0)
            btn_lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
            btn_lay.addWidget(btn)
            self._browse_table.setCellWidget(i, self._BR_COL_ACTION, btn_container)

        for row in range(self._browse_table.rowCount()):
            self._browse_table.setRowHeight(row, 48)

    def _on_download_plugin(self, url: str, filename: str):
        """Download a .synpkg from the given URL and install it."""
        # Find the button that triggered this and update it
        sender = self.sender()
        if sender:
            sender.setEnabled(False)
            sender.setText("Downloading...")

        import urllib.request
        import tempfile
        try:
            tmp_dir = tempfile.mkdtemp()
            tmp_path = Path(tmp_dir) / filename
            self._browse_status.setText(f"Downloading {filename}...")
            self._browse_status.setStyleSheet("color:#f0c040; padding:4px;")
            QtWidgets.QApplication.processEvents()

            urllib.request.urlretrieve(url, str(tmp_path))

            ok, msg = install_synpkg(tmp_path, self._plugin_dir, overwrite=True)
            if not ok:
                QtWidgets.QMessageBox.warning(self, "Install Failed", msg)
                if sender:
                    sender.setEnabled(True)
                    sender.setText("Retry")
                return

            self._browse_status.setText(f"Installed {filename}")
            self._browse_status.setStyleSheet("color:#2ecc71; padding:4px;")
            if sender:
                sender.setText("Installed")
                sender.setStyleSheet("color:#888; font-size:11px;")

            self._rebuild_table()
            self._rebuild_browse_table()
            self._show_reload_banner()

        except Exception as exc:
            self._browse_status.setText(f"Download failed: {exc}")
            self._browse_status.setStyleSheet("color:#e74c3c; padding:4px;")
            if sender:
                sender.setEnabled(True)
                sender.setText("Retry")
        finally:
            # Clean up temp file
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                Path(tmp_dir).rmdir()
            except Exception:
                pass
