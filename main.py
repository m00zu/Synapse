"""Splash-aware entry point for Synapse.

Shows a loading splash before triggering the slow ``synapse.app`` import
chain so users on slower machines know the app is starting up.  Updates
the splash with per-phase status (core import / UI init / plugin loading).
"""
import sys
import pathlib
from PySide6 import QtCore, QtGui, QtWidgets


def _find_icon():
    base = pathlib.Path(__file__).parent
    # Source layout first, then Nuitka-bundle root, then a couple of
    # other reasonable spots.  ``is_file()`` is wrapped in try/except
    # because in a bundle a parent component may itself be a file
    # (Nuitka has been seen to do this), which raises NotADirectoryError.
    candidates = [
        base / 'synapse' / 'icons' / 'synapse_icon.png',
        base / 'synapse_icon.png',
        base / 'icons' / 'synapse_icon.png',
    ]
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def _show_splash(app):
    icon = _find_icon()
    pix = None
    if icon is not None:
        loaded = QtGui.QPixmap(str(icon))
        if not loaded.isNull():
            pix = loaded.scaledToWidth(400, QtCore.Qt.SmoothTransformation)
    if pix is None:
        # Solid-colour fallback — never show uninitialised pixel memory.
        pix = QtGui.QPixmap(400, 200)
        pix.fill(QtGui.QColor('#1e1e1e'))

    splash = QtWidgets.QSplashScreen(pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    splash.show()
    app.processEvents()
    return splash


def main():
    # Subcommands skip the GUI / splash entirely
    if len(sys.argv) >= 2 and sys.argv[1] == 'package':
        from synapse.app import main as _gui_main
        _gui_main()
        return

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Synapse")
    app.setStyle("Fusion")

    splash = _show_splash(app)

    def status(text: str) -> None:
        splash.showMessage(
            text,
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter,
            QtGui.QColor('white'),
        )
        app.processEvents()

    status("Loading core modules…")

    # ── Heavy imports happen here; splash stays visible ──
    from synapse.app import main as _gui_main

    status("Starting…")

    # Hand control to the GUI; splash auto-closes when the main window paints.
    _gui_main(splash=splash, on_status=status)


if __name__ == '__main__':
    main()
