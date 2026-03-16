# Transform

### Zoom

Resizes an image by a zoom factor using high-quality Lanczos resampling.

??? note "Details"
    A factor of 2.0 doubles the size; 0.5 halves it. The preview overlay shows the actual output dimensions. Works with both ImageData and MaskData inputs.
    
    - **zoom** — scale factor (default: 1.0).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Zoom Factor`

---

### Resize

Resizes an image or mask to an exact pixel size (width x height).

??? note "Details"
    - **resize_width** — target width in pixels (default: 300).
    
    - **resize_height** — target height in pixels (default: 300).
    
    - **resample** — resampling method: *lanczos*, *bilinear*, *nearest*, or *bicubic*.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Width (px)`, `Height (px)`, `Resampling`

---

### Rotate

Rotates an image counter-clockwise by a given angle in degrees.

??? note "Details"
    The canvas is expanded to fit the full rotated image; surrounding areas are filled with black. Works with both ImageData and MaskData inputs.
    
    - **angle** — rotation angle in degrees (default: 0.0).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Angle (°)`

---

### Mirror / Flip

Flips or mirrors an image or mask along one or both axes.

??? note "Details"
    Axis options:

    - *horizontal* — flip left-right (mirror across the vertical centre line)
    - *vertical* — flip top-bottom (mirror across the horizontal centre line)
    - *both* — flip both axes (equivalent to 180-degree rotation)
    
    Works with both ImageData and MaskData inputs.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Flip Axis`

---
