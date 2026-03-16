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
    Uses `skimage.color.separate_stains` / `combine_stains` with both skimage presets and ImageJ Colour Deconvolution 2 presets.
    
    Supported staining matrices:

    - *H&E (HED)* — Haematoxylin + Eosin + DAB (most common H&E stain)
    - *H-DAB* — Haematoxylin + DAB (immunohistochemistry)
    - *Feulgen + LG* — Feulgen + Light Green (DNA staining)
    - *Giemsa* — Methyl Blue + Eosin (blood/bone marrow)
    - *FastRed+Blue+DAB*, *Methyl Green+DAB*, *H+AEC*, *Blue+Rose+Orange*, *Methyl Blue+Ponceau*, *Alcian Blue+H*, *H+PAS*, *Masson Trichrome*
    
    Output mode:

    - *Colored* (default) — each channel is reconstructed as a full RGB image retaining the stain's original colour on a white background
    - *Grayscale* — each channel is an 8-bit intensity map where brighter pixels indicate higher stain concentration (useful for downstream quantification)
    
    Third stain completion mode:

    - *Auto (matrix)* — keep matrix as provided (if 3rd vector missing, uses Ruifrok fallback)
    - *Cross Product* — force stain-3 = stain-1 x stain-2
    - *Ruifrok* — force stain-3 via Ruifrok/Johnston fallback per channel
    
    Outputs ch1, ch2, ch3 correspond to the first, second, and third stain vectors after completion and normalisation.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `ch1` | image |
| **Output** | `ch2` | image |
| **Output** | `ch3` | image |

**Properties:** `Stain Matrix`, `Third Stain`, ``

---
