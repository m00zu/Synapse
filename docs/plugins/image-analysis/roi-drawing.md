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

Draws multiple editable shapes over an input image and outputs the annotated result.

??? note "Details"
    Each shape has independent colour, width, and style. Optionally accepts a mask input and draws the mask outline as well.

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
