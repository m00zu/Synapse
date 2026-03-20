# Color

### Split RGB

Splits an RGB image into its individual Red, Green, and Blue channels.

??? note "Details"
    Each output is a single-channel grayscale image corresponding to one color plane.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `red` | image |
| **Output** | `green` | image |
| **Output** | `blue` | image |

---

### Merge RGB

Merges three single-channel grayscale images into a single RGB image.

??? note "Details"
    Any unconnected channel defaults to zero (black). All connected channels must have the same dimensions.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `red` | image |
| **Input** | `green` | image |
| **Input** | `blue` | image |
| **Output** | `image` | image |

---

### RGB to Gray

Converts an RGB or RGBA image to a single-channel grayscale image.

??? note "Details"
    Method options:

    - *Luminosity (Rec.601)* — standard broadcast weights (PIL default): `L = 0.299R + 0.587G + 0.114B`
    - *Luminosity (Rec.709)* — HDTV/sRGB weights used by skimage: `L = 0.2125R + 0.7154G + 0.0721B`
    - *Average* — simple mean of R, G, B
    - *Red* / *Green* / *Blue* — extracts a single colour channel as grayscale
    
    Output is single-channel (L-mode) ImageData.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Method`

---

### Color Deconvolution

Separates staining colours in histology images using colour deconvolution.

??? note "Details"
    Stain matrices prefixed with SK are from scikit-image, CD2 matrices are
    from ImageJ's Colour Deconvolution 2 plugin.
    
    Output mode:
    
    - *Colored* — each channel retains the stain's original colour on a white background
    - *Grayscale* — intensity map where brighter = more stain (for quantification)
    
    Third stain completion:
    
    - *Ruifrok* — Ruifrok/Johnston fallback
    - *Cross Product* — stain-3 = stain-1 x stain-2
    - *Auto* — keep matrix as provided

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `ch1` | image |
| **Output** | `ch2` | image |
| **Output** | `ch3` | image |

**Properties:** `Stain Matrix`, `Third Stain`, ``

---

### Channel Colorize

Remaps RGB channels to custom colors and composites them.

??? note "Details"
    Each channel can be assigned any color. The node multiplies each
    channel's grayscale intensity by its chosen color, then additively
    blends all channels into one RGB output.
    
    Use cases:
    
    - Change DAPI from blue to cyan
    - Show two channels in magenta + green for better contrast

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

---

### Merge Image

Additively blend multiple images into one output.

??? note "Details"
    Connect any number of images to the input port. The node sums all
    input pixel values and clips to [0, 1]. Useful for combining
    individually color-adjusted channels into a single composite
    (e.g. merge ch1 + ch2 + ch4, skipping DAPI).
    
    Works with both grayscale and RGB inputs. Grayscale inputs are
    broadcast across all 3 RGB channels.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

---
