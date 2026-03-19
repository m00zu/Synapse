"""
minimap.py
==========
A floating minimap overlay that sits in the corner of the NodeGraph viewer.
Shows a bird's-eye view of the entire scene with a viewport indicator rectangle.
"""

from PySide6 import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt


class MinimapWidget(QtWidgets.QFrame):
    """
    Semi-transparent minimap rendered as a small QGraphicsView
    overlaid on top of the main NodeGraph viewer.
    """

    _SIZE = 200          # default side length (px)
    _MARGIN = 8          # distance from the corner of the parent
    _BG = QtGui.QColor(30, 30, 30, 200)
    _BORDER = QtGui.QColor(80, 80, 80, 220)
    _VIEWPORT_PEN = QtGui.QColor(100, 180, 255, 200)
    _VIEWPORT_FILL = QtGui.QColor(100, 180, 255, 30)

    def __init__(self, viewer: QtWidgets.QGraphicsView, parent=None):
        super().__init__(parent or viewer)
        self._viewer = viewer
        self._scene = viewer.scene()

        # Frame styling
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self._SIZE, self._SIZE)

        # Internal mini-view that shares the same scene
        self._mini = QtWidgets.QGraphicsView(self._scene, self)
        self._mini.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        self._mini.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._mini.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._mini.setInteractive(False)
        self._mini.setStyleSheet("background: transparent; border: none;")
        self._mini.setFixedSize(self._SIZE, self._SIZE)

        # Viewport rect overlay (drawn manually)
        self._vp_rect = QtCore.QRectF()

        # Refresh timer — update the minimap at a modest interval
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(250)  # 4 fps is enough
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        # Also listen for scene changes
        self._scene.changed.connect(self._schedule_refresh)
        self._pending_refresh = False

        self._reposition()
        self._refresh()

    # ── public ────────────────────────────────────────────────────────────

    def set_visible(self, visible: bool):
        self.setVisible(visible)

    # ── positioning ───────────────────────────────────────────────────────

    def _reposition(self):
        """Pin to the bottom-right corner of the parent widget."""
        p = self.parentWidget()
        if p:
            x = p.width() - self._SIZE - self._MARGIN
            y = p.height() - self._SIZE - self._MARGIN
            self.move(max(0, x), max(0, y))

    # ── refresh logic ─────────────────────────────────────────────────────

    def _schedule_refresh(self, _regions=None):
        self._pending_refresh = True

    def _refresh(self):
        if not self.isVisible():
            return

        # Fit the entire scene into the mini-view
        items_rect = self._scene.itemsBoundingRect()
        if items_rect.isNull() or items_rect.isEmpty():
            items_rect = QtCore.QRectF(-500, -500, 1000, 1000)

        # Add padding
        pad = max(items_rect.width(), items_rect.height()) * 0.1
        padded = items_rect.adjusted(-pad, -pad, pad, pad)
        self._mini.fitInView(padded, Qt.AspectRatioMode.KeepAspectRatio)

        # Compute the viewport rectangle in scene coordinates
        vp = self._viewer.viewport().rect()
        top_left = self._viewer.mapToScene(vp.topLeft())
        bottom_right = self._viewer.mapToScene(vp.bottomRight())
        self._vp_rect = QtCore.QRectF(top_left, bottom_right)

        self._pending_refresh = False
        self.update()

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Background
        painter.setBrush(self._BG)
        painter.setPen(QtGui.QPen(self._BORDER, 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        # Draw viewport rectangle
        if not self._vp_rect.isNull():
            # Map scene rect → mini-view widget coords → our local coords
            tl = self._mini.mapFromScene(self._vp_rect.topLeft())
            br = self._mini.mapFromScene(self._vp_rect.bottomRight())
            local_rect = QtCore.QRectF(QtCore.QPointF(tl), QtCore.QPointF(br))

            pen = QtGui.QPen(self._VIEWPORT_PEN, 2)
            painter.setPen(pen)
            painter.setBrush(self._VIEWPORT_FILL)
            painter.drawRect(local_rect)

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._mini.setFixedSize(self.size())

    # ── click-to-navigate ─────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._navigate_to(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._navigate_to(event.position())

    def _navigate_to(self, local_pos):
        """Center the main viewer on the scene position corresponding to local_pos."""
        scene_pos = self._mini.mapToScene(QtCore.QPoint(int(local_pos.x()), int(local_pos.y())))
        self._viewer.centerOn(scene_pos)
        self._refresh()

    # ── parent resize tracking ────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj == self.parentWidget() and event.type() == QtCore.QEvent.Type.Resize:
            self._reposition()
        return super().eventFilter(obj, event)
