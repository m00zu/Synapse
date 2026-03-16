# Exposure & Contrast

### Equalize Adapthist

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast.

??? note "Details"
    - **clip_limit** — controls contrast amplification; lower values produce subtler enhancement (default: 0.01).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Clip Limit`

---

### Image Math

Performs pixel-wise arithmetic and logical operations on one or two images or masks.

??? note "Details"
    Two-input operations (connect both A and B):

    - *A + B* — saturating add: `clip(A + B, 0, 255)`
    - *A - B* — saturating subtract: `clip(A - B, 0, 255)`
    - *A x B (image)* — normalised multiply: `clip(A * B / 255, 0, 255)`
    - *A x B (apply mask)* — mask A by B: `A * (B > 0)`
    - *A AND B* — mask intersection: `(A>0) & (B>0)` to 0/255
    - *A OR B* — mask union: `(A>0) | (B>0)` to 0/255
    - *Max(A, B)* — element-wise maximum
    - *Min(A, B)* — element-wise minimum
    - *Blend* — weighted blend: `A*v + B*(1-v)` where **v** is the scalar value (0-1)
    
    Single-input operations (only A is required):

    - *Invert A* — `255 - A`
    - *Normalize A* — stretch to 0-255
    - *Gamma A^v* — `(A/255)^v * 255` where **v** is the scalar value
    - *Threshold A > v* — `(A > v)` to 0/255 where **v** is the scalar value (0-255)
    
    Both outputs always carry the same pixel data. Binary operations (AND, OR, Apply Mask, Threshold) wrap the result as MaskData; all other operations wrap it as ImageData.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `A (image/mask)` | image |
| **Input** | `B (mask)` | mask |
| **Output** | `image` | image |
| **Output** | `mask` | mask |

**Properties:** `Operation`

---

### Brightness / Contrast

Adjusts brightness and contrast interactively using a histogram with draggable Min/Max lines.

??? note "Details"
    Drag the red **Min** line and cyan **Max** line on the histogram to set the display window. The output image is always 8-bit with a linear stretch: `output = clip((pixel - Min) / (Max - Min) * 255, 0, 255)`.
    
    Works with 8-bit and 16-bit input images. For 16-bit input the histogram spans 0-65535 and the handles can be placed anywhere in that range.
    
    Convenience controls:

    - **Brightness** (-100 to +100) — shifts the window centre up or down
    - **Contrast** (-100 to +100) — narrows or widens the window symmetrically

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

---

### Gamma Contrast

Applies gamma correction to adjust image tonality non-linearly.

??? note "Details"
    Transforms each pixel via `O = I^gamma * gain`.
    
    - **gamma** — exponent controlling the curve shape (default: 1.0). Values below 1 brighten; values above 1 darken.
    
    - **gain** — multiplicative scaling factor applied after gamma (default: 1.0).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Gamma`, `Gain`

---
