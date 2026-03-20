# ROI & Drawing

### ROI Mask

Draws an ROI (ellipse, rectangle, or polygon) directly on the node surface and outputs a binary mask plus a cropped image.

??? note "Details"
    Inputs:

    - **image** — the image to draw on (sets the background)
    
    Outputs:

    - **mask** — binary L-mode PIL image (0 / 255)
    - **cropped_image** — input image with non-ROI pixels set to black

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `cropped_image` | image |

---

### Crop

Crops an image or mask to a rectangle drawn directly on the node.

??? note "Details"
    Click and drag on the node surface to draw the crop rectangle. Drag the edges to resize it, or drag the body to move it. Press Delete to clear the selection (outputs the full image when no rectangle is drawn). Supports both ImageData and MaskData inputs.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

---

### Draw Shape

Draw shapes, text, and annotations on an image.

??? note "Details"
    Shapes: rectangle, ellipse, polygon, arrow, bezier curve, and free text.
    Each shape has its own color, line width, line style (solid/dashed/dotted),
    and optional fill with adjustable opacity. Shapes can be moved, resized,
    and edited interactively on the canvas.
    
    Inputs:

    - **image** — background image (optional)
    - **mask** (multi-input) — binary masks shown as colored contours
    - **label_image** — segmentation labels shown as colored overlay
    
    Controls:

    - Line width, style, and color per shape
    - Fill toggle + opacity for closed shapes (rectangle, ellipse, polygon)
    - Auto Fill — fill all mask contours at once
    - Fill All — apply fill to every mask input
    - Label overlay opacity — transparency of segmentation label colors
    - Geometry spinboxes (X, Y, W, H) for precise positioning
    - Font size for texts
    - No Preview — skip interactive canvas rendering for speed
    
    Hold Shift while drawing to constrain to square/circle.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Input** | `mask` | mask |
| **Input** | `label_image` | label_image |
| **Output** | `image` | image |

---

### Mask Editor

Interactively edits a mask by drawing shapes and applying boolean operations.

??? note "Details"
    Supports add, subtract, and intersect modes with rect, ellipse, polygon, and lasso tools. Accepts an optional background image for visual reference and an optional mask input as the starting state.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Input** | `mask` | mask |
| **Output** | `mask` | mask |

---

### Scale Bar

Draws a calibrated scale bar on a microscopy image.

??? note "Details"
    Reads the `scale_um` metadata from the upstream ImageData to calculate
    the correct pixel length for the bar. If no scale info is available,
    the node reports an error.
    
    Options:

    - **bar_length_um** — desired bar length in micrometers
    - **position** — corner placement
    - **bar_color** — color of the bar and label
    - **bar_height** — thickness in pixels
    - **show_label** — display "100 µm" text
    - **font_size** — label size
    - **padding_x / padding_y** — margin from image edge

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Bar Length (µm)`, `Position`, `Show Label`, `Text-Bar Gap (px)`

---

### Mask Overlay

Draw a mask contour (or fill) on an image.

??? note "Details"
    A lightweight alternative to Draw Shape for simple mask visualization.
    Connect an image and a mask, and the mask boundary is drawn as a colored
    contour on the output image. Optionally fill the masked region with a
    semi-transparent color.
    
    Controls:

    - Line width, style (solid/dashed/dotted), and color
    - Fill toggle with adjustable opacity

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Input** | `mask` | mask |
| **Output** | `image` | image |

---
