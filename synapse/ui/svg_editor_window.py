"""
ui/svg_editor_window.py
=======================
A standalone Qt application window for editing SVG files, designed to be launched 
from the SvgEditorNode within the node graph.
"""
import xml.etree.ElementTree as ET
from PySide6 import QtWidgets, QtCore, QtGui
import traceback

class SVGParsingException(Exception):
    pass

class SvgGraphicsScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

class PropertiesPanel(QtWidgets.QWidget):
    """
    A smart form that parses SVG attributes and generates intuitive GUI widgets
    (like Color Pickers for 'fill', SpinBoxes for 'width', etc.)
    """
    properties_changed = QtCore.Signal() # Fired when user tweaks a GUI control

    def __init__(self, parent=None):
        super().__init__(parent)
        self.xml_elem = None
        
        # We'll use a polished group box to group settings visually
        self.group_box = QtWidgets.QGroupBox("Element Properties")
        self.group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #bbb;
            }
        """)
        
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self.group_box)
        main_layout.addStretch()
        
        self.layout = QtWidgets.QFormLayout(self.group_box)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
    def set_element(self, xml_elem):
        self.xml_elem = xml_elem
        
        # Clear existing layout
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if self.xml_elem is None:
            return
            
        # Common aesthetic attributes we want to build GUI for
        KNOWN_PROPS = {
            'fill': 'color',
            'stroke': 'color',
            'stroke-width': 'spinbox_float',
            'opacity': 'slider',
            'font-size': 'spinbox_int',
        }
        
        # Also handle "style" string
        style_str = self.xml_elem.get('style', '')
        self.style_dict = {}
        for part in style_str.split(';'):
            if ':' in part:
                k, v = part.split(':', 1)
                self.style_dict[k.strip()] = v.strip()
                
        # We will iterate over a specific prioritized list to make it look organized like Prism
        PRIORITY_KEYS = ['fill', 'stroke', 'stroke-width', 'opacity']
        
        # If it's a text element, we explicitly add a text editor box
        if self.xml_elem.tag.endswith('text'):
            PRIORITY_KEYS = ['text', 'font-family', 'font-size'] + PRIORITY_KEYS
            
        # Add basic positional / geometric dims if applicable
        if 'cx' in self.xml_elem.attrib: PRIORITY_KEYS = ['cx', 'cy', 'r'] + PRIORITY_KEYS
        if 'x' in self.xml_elem.attrib and 'width' in self.xml_elem.attrib: PRIORITY_KEYS = ['x', 'y', 'width', 'height'] + PRIORITY_KEYS
        
        # Deduplicate while preserving order
        seen = set()
        ordered_keys = [x for x in PRIORITY_KEYS if not (x in seen or seen.add(x))]
        
        for key in ordered_keys:
            val = self._get_attr_or_style(key)
            if val is not None:
                self._add_row_for_key(key, val)
                
        # Add remaining unmatched standard attributes to the bottom as raw text
        for attr, val in self.xml_elem.attrib.items():
            if attr not in ordered_keys and attr not in ['style', 'd', 'transform', 'id', 'class']:
                self._add_row_for_key(attr, val, force_text=True)

    def _get_attr_or_style(self, key):
        if key == 'text':
            return self.xml_elem.text if self.xml_elem.text else ''
        if key in self.xml_elem.attrib:
            return self.xml_elem.attrib[key]
        if key in self.style_dict:
            return self.style_dict[key]
        return None
        
    def _set_attr_or_style(self, key, value):
        if key == 'text':
            self.xml_elem.text = str(value)
        elif key in self.style_dict or (key not in self.xml_elem.attrib and self.style_dict):
            self.style_dict[key] = str(value)
            # Reconstruct style string
            new_style = ";".join([f"{k}:{v}" for k, v in self.style_dict.items()])
            self.xml_elem.set('style', new_style)
        else:
            self.xml_elem.set(key, str(value))
            
        self.properties_changed.emit()

    def _add_row_for_key(self, key, current_val, force_text=False):
        # Color Picker
        if not force_text and (key in ['fill', 'stroke'] or str(current_val).startswith('#')):
            btn = QtWidgets.QPushButton()
            btn.setFixedSize(50, 24)
            # Handle "none" color explicitly
            init_color = current_val if current_val and current_val != 'none' else '#ffffff'
            
            # Draw initial color on button
            def update_btn_style(c):
                txt = "None" if (current_val == 'none' and c == '#ffffff') else ""
                btn.setText(txt)
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {c};
                        border: 1px solid #777;
                        border-radius: 3px;
                        color: {'#000' if txt else 'transparent'};
                    }}
                    QPushButton:hover {{ border: 1px solid #aaa; }}
                """)
            update_btn_style(init_color)
            
            def on_color_clicked():
                qc = QtGui.QColor(init_color)
                col = QtWidgets.QColorDialog.getColor(qc, self, f"Select {key} Color", QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
                if col.isValid():
                    # Preserve alpha if needed, using hex8 if alpha < 255
                    if col.alpha() < 255:
                        hex_col = col.name(QtGui.QColor.NameFormat.HexArgb)
                    else:
                        hex_col = col.name()
                    update_btn_style(hex_col)
                    self._set_attr_or_style(key, hex_col)
            btn.clicked.connect(on_color_clicked)
            self.layout.addRow(f"<b>{key.capitalize()}</b>", btn)
        
        # Spinbox
        elif not force_text and key in ['stroke-width', 'r', 'width', 'height', 'font-size', 'x', 'y', 'cx', 'cy', 'opacity']:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-99999, 99999)
            
            if key == 'opacity':
                spin.setRange(0, 1)
                spin.setSingleStep(0.1)
                spin.setDecimals(2)
            else:
                spin.setSingleStep(0.5)
                spin.setDecimals(2)
                
            spin.setStyleSheet("""
                QDoubleSpinBox {
                    background: #222; border: 1px solid #444; color: #eee; 
                    padding: 3px; border-radius: 3px;
                }
                QDoubleSpinBox:focus { border: 1px solid #2e7d32; }
            """)
            
            # Some SVGs use 'px' or 'pt' suffixes
            try:
                num_val = float(str(current_val).replace('px', '').replace('pt', '').strip())
                spin.setValue(num_val)
            except ValueError:
                spin.setValue(0.0)
                
            def on_spin_changed(v):
                suffix = 'px' if 'px' in str(current_val) else ''
                suffix = 'pt' if 'pt' in str(current_val) else suffix
                self._set_attr_or_style(key, f"{v}{suffix}")
            spin.valueChanged.connect(on_spin_changed)
            self.layout.addRow(f"<b>{key.capitalize()}</b>", spin)
            
        # Default Line Edit
        else:
            ledit = QtWidgets.QLineEdit(str(current_val))
            ledit.setStyleSheet("""
                QLineEdit {
                    background: #222; border: 1px solid #444; color: #eee; 
                    padding: 3px; border-radius: 3px;
                }
                QLineEdit:focus { border: 1px solid #2e7d32; }
            """)
            ledit.editingFinished.connect(lambda k=key, w=ledit: self._set_attr_or_style(k, w.text()))
            self.layout.addRow(f"<b>{key.capitalize()}</b>", ledit)

    
class SvgEditorWindow(QtWidgets.QMainWindow):
    """
    Main editor window for tweaking SVG documents graphically or via XML/DOM.
    """
    svg_updated = QtCore.Signal(str) # Emitted with new SVG string on save

    def __init__(self, initial_svg_string=None, parent=None):
        super(SvgEditorWindow, self).__init__(parent)
        self.setWindowTitle("SVG Content Editor")
        self.resize(1200, 800)
        
        self.svg_string = initial_svg_string
        self.xml_tree = None
        self.root_node = None
        
        self.init_ui()
        if self.svg_string:
            self.load_svg_string(self.svg_string)

    def init_ui(self):
        # 1. Main Canvas Area (Center)
        self.scene = SvgGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.setCentralWidget(self.view)
        
        # 2. XML Source Editor (Bottom Dock)
        self.xml_dock = QtWidgets.QDockWidget("XML Source", self)
        self.xml_editor = QtWidgets.QPlainTextEdit()
        self.xml_editor.setFont(QtGui.QFont("Courier New", 11))
        self.xml_dock.setWidget(self.xml_editor)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.xml_dock)
        
        # 3. DOM Tree & Properties (Right Dock)
        self.dom_dock = QtWidgets.QDockWidget("DOM Explorer", self)
        
        dock_widget = QtWidgets.QWidget()
        dock_layout = QtWidgets.QVBoxLayout(dock_widget)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.properties_panel = PropertiesPanel()
        scroll_area.setWidget(self.properties_panel)
        self.properties_panel.properties_changed.connect(self._on_apply_properties)
        
        self.apply_button = QtWidgets.QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self._on_apply_properties)
        
        # Add a title label for the properties section to make it look nicer
        props_label = QtWidgets.QLabel("<b>Selected Element Properties</b>")
        props_label.setContentsMargins(5, 5, 0, 5)
        dock_layout.addWidget(props_label)
        
        dock_layout.addWidget(scroll_area, stretch=1)
        
        self.dom_dock.setWidget(dock_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dom_dock)
        
        # Toolbar
        toolbar = self.addToolBar("Main")
        save_action = toolbar.addAction("Save to NodeGraph")
        save_action.triggered.connect(self.save)
        
        sync_down_action = toolbar.addAction("Sync XML -> DOM")
        sync_down_action.triggered.connect(self._sync_xml_to_dom)

    def _sync_xml_to_dom(self):
        """Parse text from XML editor into ElementTree and rebuild UI"""
        try:
            self.load_svg_string(self.xml_editor.toPlainText())
        except ET.ParseError as e:
            QtWidgets.QMessageBox.critical(self, "Parse Error", f"Invalid XML: {e}")

    def load_svg_string(self, svg_str):
        self.svg_string = svg_str
        self.xml_editor.setPlainText(svg_str)
        try:
            # Strip namespace mappings temporarily for easier parsing, or handle them natively
            # Matplotlib outputs heavy namespaces and defs
            self.xml_tree = ET.ElementTree(ET.fromstring(svg_str))
            self.root_node = self.xml_tree.getroot()
            self._rebuild_dom_tree()
            self._rebuild_scene()
        except Exception as e:
            traceback.print_exc()

    def _rebuild_dom_tree(self):
        # We no longer display the raw XML DOM Tree, as user found it confusing.
        pass

    def _rebuild_scene(self):
        self.scene.clear()
        if self.root_node is None: return

        from ui.svg_graphics_items import SvgRectItem, SvgEllipseItem, SvgTextItem, SvgPathItem, SvgGroupItem
        
        # Build ID map for resolving <use> tags
        id_map = {}
        for el in self.root_node.iter():
            ident = el.get('id', '')
            if ident: 
                id_map[ident] = el
        
        def render_element(elem, parent_item=None, is_clone=False):
            tag = elem.tag.split('}')[-1]
            
            # Skip structural and hidden tags (unless we are cloning them for a <use>)
            if not is_clone and tag in ['defs', 'clipPath', 'style', 'title', 'desc']:
                return
                
            qitem = None
            
            if tag == 'rect':
                qitem = SvgRectItem(elem)
            elif tag == 'circle' or tag == 'ellipse':
                qitem = SvgEllipseItem(elem)
            elif tag == 'text':
                qitem = SvgTextItem(elem)
            elif tag == 'path':
                qitem = SvgPathItem(elem)
            elif tag in ['g', 'svg']:
                qitem = SvgGroupItem(elem)
            elif tag == 'use':
                # <use> is essentially a group with a transform and an x/y offset that contains a clone
                qitem = SvgGroupItem(elem)
                
                # Apply x/y offsets which <use> tags have natively
                dx = float(elem.get('x', '0'))
                dy = float(elem.get('y', '0'))
                if dx != 0 or dy != 0:
                    # PySide translates locally before the item's transform
                    import PySide6.QtGui as QtGui
                    cur_t = qitem.transform()
                    t = QtGui.QTransform()
                    t.translate(dx, dy)
                    qitem.setTransform(t * cur_t)
                
                href = elem.get('{http://www.w3.org/1999/xlink}href', elem.get('href', ''))
                if href.startswith('#'):
                    target_elem = id_map.get(href[1:])
                    if target_elem is not None:
                        # Render the clone into this group
                        render_element(target_elem, parent_item=qitem, is_clone=True)
                

            if qitem:
                if hasattr(qitem, 'set_xml_source') and not is_clone:
                    # Only map original elements back to XML selection, not internal cloned paths
                    qitem.set_xml_source(elem, update_callback=self._on_item_updated, selection_callback=self._on_item_selected)
                
                if parent_item and isinstance(parent_item, QtWidgets.QGraphicsItemGroup):
                    parent_item.addToGroup(qitem)
                else:
                    self.scene.addItem(qitem)
                
                # Normally recurse into children
                if tag != 'use': # <use> children are practically empty, we handled its clone above
                    for child in elem:
                        render_element(child, qitem, is_clone)
                    
        render_element(self.root_node)
        
    def _on_item_updated(self, xml_elem):
        """Called when an item is dragged or text changed visually."""
        # Simple string rewrite and sync
        self.xml_editor.setPlainText(ET.tostring(self.root_node, encoding='unicode'))

    def _on_item_selected(self, xml_elem, qitem, update_tree=True):
        """Called when a graphics item is clicked on the canvas."""
        self._current_edit_elem = xml_elem
        self.properties_panel.set_element(xml_elem)

    def _on_apply_properties(self):
        """Called when any GUI widget in PropertiesPanel is tweaked."""
        # The PropertiesPanel already wrote the change back to the XML Element
        # We just need to sync the text view and rebuild the visual scene.
        self.xml_editor.setPlainText(ET.tostring(self.root_node, encoding='unicode'))
        
        # Remember the selected item to re-select it after scene rebuild
        old_elem = self._current_edit_elem
        
        self._rebuild_scene()
        
        # Restore selection silently
        if old_elem is not None:
            self.scene.clearSelection()
            for item in self.scene.items():
                if hasattr(item, 'xml_elem') and item.xml_elem is old_elem:
                    item.setSelected(True)
                    break
        
    def save(self):
        """Emits the current SVG string back to the NodeGraph."""
        self.svg_updated.emit(self.xml_editor.toPlainText())
        self.close()

if __name__ == "__main__":
    # Test stub
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    app = QtWidgets.QApplication(sys.argv)
    try:
        with open("matplotlib_test.svg", "r") as f:
            test_svg = f.read()
    except Exception as e:
        print(e)
        test_svg = '<svg height="100" width="100"><circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" /></svg>'
        
    w = SvgEditorWindow()
    w.load_svg_string(test_svg)
    w.show()
    sys.exit(app.exec())
