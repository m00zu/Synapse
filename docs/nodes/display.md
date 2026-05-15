# Display

### Data Figure Node

Displays incoming Image or Figure data directly on the node surface.

??? note "Details"
    Accepts `FigureData` (with optional SVG override) or raw matplotlib figures.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | figure |

---

### Data Table Node

Displays incoming DataFrame data directly on the node surface.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |

---

### Image Viewer

Displays a PIL Image directly on the node surface for quick inline inspection.

??? note "Details"
    Accepted input types:

    - *ImageData* -- unwraps the payload
    - *LabelData* -- uses the pre-generated colored visualization
    - *Raw PIL Image* -- displayed as-is

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | image |

---

### Pop-up Display

Takes any input and pops up a preview window to inspect it.

??? note "Details"
    Supported data types:

    - *DataFrame* -- shown as an editable table dialog
    - *Figure* -- rendered to PNG and shown in a scrollable image dialog
    - *Image* -- displayed as a scrollable PIL image dialog
    - *Other* -- shown as a plain text message box

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |

---

### SVG Editor

Converts an upstream matplotlib Figure to SVG for interactive element editing.

??? note "Details"
    Usage:

    - Click any highlighted element to select it.
    - Double-click to open the properties panel (fill, stroke, opacity, etc.).
    - Drag text labels (orange cursor) to reposition them.
    - Click "Apply" in the properties panel to commit changes.
    - Click "Reset SVG" to discard edits and reload from the figure.
    
    Edits are stored in the `_svg_data` node property and survive
    re-evaluation as long as the upstream figure is unchanged. Reset SVG
    clears them.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | figure |
| **Output** | `out` | figure |

---
