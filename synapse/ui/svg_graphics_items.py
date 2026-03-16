"""
ui/svg_graphics_items.py
========================
Custom QGraphicsItems parsing individual SVG nodes into editable PySide objects.
This allows moving, resizing, and altering attributes of parsed SVG.
"""
from PySide6 import QtWidgets, QtCore, QtGui
from xml.etree.ElementTree import Element
import re

def parse_transform(transform_str):
    """Parses SVG transform strings into a QTransform."""
    transform = QtGui.QTransform()
    if not transform_str: return transform
        
    for match in re.finditer(r'(\w+)\s*\(([^)]+)\)', transform_str):
        cmd, args = match.groups()
        args = [float(x) for x in re.split(r'[,\s]+', args.strip()) if x]
        
        if cmd == 'translate':
            dx = args[0]
            dy = args[1] if len(args) > 1 else 0.0
            transform.translate(dx, dy)
        elif cmd == 'scale':
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            transform.scale(sx, sy)
        elif cmd == 'rotate':
            a = args[0]
            transform.rotate(a)
        elif cmd == 'matrix' and len(args) == 6:
            # SVG matrix: a, b, c, d, e, f
            # QTransform: m11, m12, m13, m21, m22, m23, m31, m32, m33
            mat = QtGui.QTransform(args[0], args[1], args[2], args[3], args[4], args[5])
            transform = mat * transform
            
    return transform

# Helper to map SVG stroke/fill properties to QPen/QBrush
def apply_svg_style(item, xml_elem: Element):
    style_str = xml_elem.get('style', '')
    style_dict = {}
    for part in style_str.split(';'):
        if ':' in part:
            k, v = part.split(':', 1)
            style_dict[k.strip()] = v.strip()
    
    # 1. Fill
    fill_color = xml_elem.get('fill', style_dict.get('fill', 'none'))
    if fill_color != 'none' and fill_color:
        brush = QtGui.QBrush(QtGui.QColor(fill_color))
        if hasattr(item, 'setBrush'):
            item.setBrush(brush)
    else:
        if hasattr(item, 'setBrush'):
            item.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
            
    # 2. Stroke
    stroke_color = xml_elem.get('stroke', style_dict.get('stroke', 'none'))
    stroke_width = xml_elem.get('stroke-width', style_dict.get('stroke-width', '1'))
    
    if stroke_color != 'none' and stroke_color:
        pen = QtGui.QPen(QtGui.QColor(stroke_color))
        try:
            pen.setWidthF(float(stroke_width.replace('px', '')))
        except ValueError:
            pass
        if hasattr(item, 'setPen'):
            item.setPen(pen)
    else:
        if hasattr(item, 'setPen'):
            item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
            
    # 3. Interactivity
    # Exclude structural invisible items (like mpl white backgrounds or hitboxes) from selection
    is_invisible = False
    
    # Check if fill is effectively none (white background is also tricky but let's stick to true none/0 opacity for now)
    has_fill = fill_color and fill_color != 'none' and style_dict.get('fill-opacity', '1') != '0'
    has_stroke = stroke_color and stroke_color != 'none' and style_dict.get('stroke-opacity', '1') != '0'
    
    if not has_fill and not has_stroke:
        is_invisible = True
        
    # Some matplotlib backgrounds are pure white rectangles covering the whole thing.
    # While debatable, usually users don't want to drag the background.
    # We will just rely on explicit "none" for now.
    
    if is_invisible and hasattr(item, 'setFlags'):
        flags = item.flags()
        flags &= ~QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        flags &= ~QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
        item.setFlags(flags)


class EditableSvgItemMixin:
    """
    Mixin added to standard QGraphicsItems to link them back to their XML elements.
    It catches drag/move events and writes changes back to the ElementTree node.
    """
    def set_xml_source(self, xml_elem: Element, update_callback=None, selection_callback=None):
        self.xml_elem = xml_elem
        self.update_callback = update_callback
        self.selection_callback = selection_callback
        
    def write_back_position(self):
        """Called subclasses when an item is moved visually."""
        if hasattr(self, 'xml_elem') and self.xml_elem is not None:
            # We will use transform matrix strictly to handle movement rather than mutating the original
            if self.pos() != QtCore.QPointF(0, 0):
                dx = self.x()
                dy = self.y()
                
                # Check for existing transform
                t = self.xml_elem.get('transform', '')
                if t:
                    self.xml_elem.set('transform', f"translate({dx}, {dy}) {t}")
                else:
                    self.xml_elem.set('transform', f"translate({dx}, {dy})")
                
                # Reset local pos to 0 since we baked it into the XML
                # In a real sync we'd redraw the scene, but applying it locally works if we don't redraw immediately
                self.setPos(0, 0)
                
                if self.update_callback:
                    self.update_callback(self.xml_elem)

    def mousePressEvent(self, event):
        """When an item is clicked in the graphics view, notify the main editor."""
        super().mousePressEvent(event)
        if hasattr(self, 'selection_callback') and self.selection_callback:
            self.selection_callback(self.xml_elem, self)
        else:
            # Bubble up to the closest parent with a callback (e.g., cloned paths inside a <use> group)
            p = self.parentItem()
            while p:
                if hasattr(p, 'selection_callback') and getattr(p, 'selection_callback'):
                    p.selection_callback(p.xml_elem, p)
                    break
                p = p.parentItem()


class SvgRectItem(QtWidgets.QGraphicsRectItem, EditableSvgItemMixin):
    def __init__(self, xml_elem: Element):
        super().__init__()
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.set_xml_source(xml_elem)
        
        try:
            x = float(xml_elem.get('x', 0))
            y = float(xml_elem.get('y', 0))
            w = float(xml_elem.get('width', 0))
            h = float(xml_elem.get('height', 0))
            self.setRect(x, y, w, h)
        except ValueError:
            pass
            
        apply_svg_style(self, xml_elem)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.write_back_position()
        return super().itemChange(change, value)


class SvgEllipseItem(QtWidgets.QGraphicsEllipseItem, EditableSvgItemMixin):
    def __init__(self, xml_elem: Element):
        super().__init__()
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.set_xml_source(xml_elem)
        
        try:
            cx = float(xml_elem.get('cx', 0))
            cy = float(xml_elem.get('cy', 0))
            r = float(xml_elem.get('r', 0))
            # QGraphicsEllipseItem uses bounding rect, not center+radius
            self.setRect(cx - r, cy - r, r * 2, r * 2)
        except ValueError:
            pass
            
        apply_svg_style(self, xml_elem)
        
    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.write_back_position()
        return super().itemChange(change, value)


class SvgTextItem(QtWidgets.QGraphicsTextItem, EditableSvgItemMixin):
    def __init__(self, xml_elem: Element):
        super().__init__()
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        # Enable text editing on canvas
        self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
        
        self.set_xml_source(xml_elem)
        
        text = xml_elem.text or ''
        self.setPlainText(text)
        
        try:
            x = float(xml_elem.get('x', 0))
            y = float(xml_elem.get('y', 0))
            self.setPos(x, y)
        except ValueError:
            pass

        # Parse basic font specs if present
        font = self.font()
        font_size = xml_elem.get('font-size', None)
        if font_size:
            try:
                font.setPointSizeF(float(font_size.replace('px', '').replace('pt','')))
            except ValueError:
                pass
        self.setFont(font)
        
        fill = xml_elem.get('fill', 'black')
        self.setDefaultTextColor(QtGui.QColor(fill))

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        # Write back text changes
        new_text = self.toPlainText()
        if self.xml_elem is not None and self.xml_elem.text != new_text:
            self.xml_elem.text = new_text
            if self.update_callback:
                self.update_callback(self.xml_elem)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.write_back_position()
        return super().itemChange(change, value)


class SvgPathItem(QtWidgets.QGraphicsPathItem, EditableSvgItemMixin):
    """
    Parses complex d="..." strings into QPainterPath.
    """
    def __init__(self, xml_elem: Element):
        super().__init__()
        # Paths from matplotlib might be groups. We make them movable.
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.set_xml_source(xml_elem)
        
        d_str = xml_elem.get('d', '')
        if d_str:
            qpath = self._parse_d_to_qpath(d_str)
            self.setPath(qpath)
            
        apply_svg_style(self, xml_elem)

    def _parse_d_to_qpath(self, d_str):
        """
        Parses SVG `d` parameter to a QPainterPath.
        Matplotlib outputs mostly 'M' (move), 'L' (line), 'C' (bezier curve), 'Z' (close).
        """
        path = QtGui.QPainterPath()
        
        import re
        # Tokenize splitting letters and numbers
        tokens = re.findall(r'[a-zA-Z]+|-?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?', d_str)
        
        idx = 0
        cmd = ''
        try:
            while idx < len(tokens):
                t = tokens[idx]
                if t.isalpha():
                    cmd = t
                    idx += 1
                
                if cmd == 'M':   # Move to (absolute)
                    x, y = float(tokens[idx]), float(tokens[idx+1])
                    path.moveTo(x, y)
                    idx += 2
                elif cmd == 'm': # Move to (relative)
                    x, y = float(tokens[idx]), float(tokens[idx+1])
                    if path.elementCount() == 0: path.moveTo(x, y)
                    else:
                        cur = path.currentPosition()
                        path.moveTo(cur.x() + x, cur.y() + y)
                    idx += 2
                    cmd = 'l' # subsequent coordinates after m are treated as relative l
                elif cmd == 'L': # Line to (absolute)
                    x, y = float(tokens[idx]), float(tokens[idx+1])
                    path.lineTo(x, y)
                    idx += 2
                elif cmd == 'l': # Line to (relative)
                    x, y = float(tokens[idx]), float(tokens[idx+1])
                    cur = path.currentPosition()
                    path.lineTo(cur.x() + x, cur.y() + y)
                    idx += 2
                elif cmd == 'Z' or cmd == 'z': # Close path
                    path.closeSubpath()
                # Matplotlib creates complex C (cubic bezier) curves natively
                elif cmd == 'C': # Cubic Bezier (absolute)
                    if idx + 5 < len(tokens):
                        cx1, cy1 = float(tokens[idx]), float(tokens[idx+1])
                        cx2, cy2 = float(tokens[idx+2]), float(tokens[idx+3])
                        x, y     = float(tokens[idx+4]), float(tokens[idx+5])
                        path.cubicTo(cx1, cy1, cx2, cy2, x, y)
                        idx += 6
                    else:
                        break
                else:
                    # Unimplemented commands just advance token buffer (simplistic fallback)
                    idx += 1
                    
        except (ValueError, IndexError):
            pass # Malformed path string

        return path

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.write_back_position()
        return super().itemChange(change, value)


class SvgGroupItem(QtWidgets.QGraphicsItemGroup, EditableSvgItemMixin):
    """
    Renders a <g> element as a group.
    """
    def __init__(self, xml_elem: Element):
        super().__init__()
        self.xml_elem = xml_elem
        
        t = xml_elem.get('transform', '')
        if t: self.setTransform(parse_transform(t))
        
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        
    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        # Optional: draw bounding box when selected
        if self.isSelected():
            pen = QtGui.QPen(QtGui.QColor("#2196F3"))
            pen.setWidth(1)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(self.boundingRect())

