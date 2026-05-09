"""Splash-aware entry point for Synapse.

Shows a loading splash before triggering the slow ``synapse.app`` import
chain so users on slower machines know the app is starting up.  Updates
the splash with per-phase status (core import / UI init / plugin loading).
"""
import sys
import pathlib
from PySide6 import QtCore, QtGui, QtWidgets


def _show_splash(app):
    icon = pathlib.Path(__file__).parent / 'synapse' / 'icons' / 'synapse_icon.png'
    pix = QtGui.QPixmap(str(icon)) if icon.exists() else QtGui.QPixmap(400, 200)
    pix = pix.scaledToWidth(400, QtCore.Qt.SmoothTransformation)
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
