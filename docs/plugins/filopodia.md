# Filopodia Analysis

### Cell Edge Mask

Generates a binary cell-body mask from a fluorescence image.

??? note "Details"
    Step 1 of the FiloQuant pipeline. Converts the input to grayscale,
    applies a lower-bound intensity threshold, optionally fills interior
    holes, then smooths the mask with morphological opening
    (**n_open** erosions followed by **n_open** dilations). Extra
    erode+dilate cycles can be added to further refine the boundary.
    
    Parameters:

    - **threshold** — intensity cutoff (0--255)
    - **n_open** — number of opening iterations for smoothing
    - **n_erode_dilate** — additional erode+dilate cycles
    - **fill_holes** — fill interior holes before opening
    
    Output port `mask` is a MaskData (white = cell body).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `mask` | mask |

**Properties:** `Open Iterations`, `Extra Erode+Dilate`, ``

---

### Filopodia Detect

Detects filopodia candidates as a binary mask.

??? note "Details"
    Step 2 of the FiloQuant pipeline. Optionally applies CLAHE for local
    contrast enhancement and a 5x5 centre-surround sharpening convolution
    to accentuate thin bright structures, followed by a 3x3 median
    despeckle (x2) and intensity thresholding. Small isolated blobs
    (<8 px) are discarded. If a `cell_mask` is connected, an exclusion
    zone is dilated around the cell body so candidates too close to the
    cell edge are removed.
    
    Parameters:

    - **threshold** — intensity cutoff (0--255)
    - **n_distance_from_edge** — exclusion zone width in pixels around the cell body
    - **use_convolve** — enable 5x5 sharpening kernel
    - **use_clahe** — enable CLAHE local contrast pre-enhancement
    
    Output port `mask` is a MaskData of filopodia candidate regions.
    Connect to FilopodiaAnalyzeNode together with the `cell_mask`.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `cell_mask` | mask |
| **Output** | `mask` | mask |

**Properties:** `Distance from Edge (px)`, ``, ``

---

### Filopodia Analyze

Skeletonizes the filopodia mask and measures each branch.

??? note "Details"
    Step 3 (final step) of the FiloQuant pipeline.
    
    Processing steps:

    - Subtract the cell body (`cell_mask`) from the filopodia candidate mask to isolate protrusions only
    - Remove objects smaller than **min_size_px**
    - Optionally close small gaps with **repair_cycles** morphological close iterations (FiloQuant's "Filopodia repair")
    - Skeletonize via `skimage.morphology.skeletonize`
    - Measure each connected skeleton branch with diagonal-aware edge counting
    - Measure total cell edge length via `skimage.measure.perimeter`
    
    Outputs:

    - `table` — TableData with columns: `x`, `y`, `filopodia_length_px`, `edge_length_px` (one row per detected filopodium skeleton branch)
    - `visualization` — colour composite (dark background, green = cell body, cyan = isolated filopodia mask, magenta = skeleton)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `filopodia_mask` | mask |
| **Input** | `cell_mask` | mask |
| **Output** | `visualization` | image |

**Properties:** `Min Size (px)`, `Repair Cycles (close)`

---
