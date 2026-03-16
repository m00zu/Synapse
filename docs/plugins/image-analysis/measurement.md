# Measurement & Analysis

### Blob Detect

Detects bright blobs using Laplacian-of-Gaussian (LoG) filtering.

??? note "Details"
    Finds roughly circular bright spots such as cells, nuclei, vesicles, or puncta
    using `skimage.feature.blob_log`.
    
    Outputs:

    - *table* — one row per blob with columns `y`, `x`, `radius_px`
    - *overlay* — original image with detected blobs circled in red
    
    - **Min Radius** — smallest blob radius to detect (pixels).
    - **Max Radius** — largest blob radius to detect (pixels).
    - **Threshold** — detection sensitivity; lower values find more blobs.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `overlay` | image |

**Properties:** `Min Radius (px)`, `Max Radius (px)`, `Threshold`

---

### Colocalization

Computes fluorescence colocalization metrics between two channels, mirroring ImageJ Coloc 2.

??? note "Details"
    Metrics computed:

    - `PCC_full` — Pearson's r over the whole image (or ROI if mask connected)
    - `PCC_masked` — Pearson's r restricted to pixels in the mask (same as PCC_full when no mask)
    - `MOC` — Manders' Overlap Coefficient `sum(I1*I2) / sqrt(sum(I1^2) * sum(I2^2))`
    - `M1` — Manders' M1: fraction of ch1 intensity colocalising with ch2
    - `M2` — Manders' M2: fraction of ch2 intensity colocalising with ch1 (M1/M2 thresholds auto-determined by Otsu per channel)
    - `ICQ` — Li's Intensity Correlation Quotient, range (-0.5, 0.5)
    
    Outputs a 1-row table plus a scatter plot of ch1 vs ch2 pixel intensities
    (subsampled to 20,000 pixels for speed).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `ch1` | image |
| **Input** | `ch2` | image |
| **Output** | `table` | table |
| **Output** | `figure` | figure |

---

### GLCM Texture

Computes Haralick texture features from a Grey-Level Co-occurrence Matrix (GLCM).

??? note "Details"
    Averages texture features over four orientations (0, 45, 90, 135 degrees) at the
    given pixel distance. Outputs a single-row table with columns: `contrast`,
    `dissimilarity`, `homogeneity`, `energy`, `correlation`, `ASM`.
    
    - **Distance** — pixel offset for co-occurrence pairs.
    - **Grey Levels** — number of quantisation levels (fewer = faster, coarser).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `table` | table |

**Properties:** `Distance (px)`, `Grey Levels`

---

### Image Stats

Measures whole-mask region properties and pixel intensity statistics in a single table row.

??? note "Details"
    Connect an image, a mask, or both -- at least one must be connected.
    
    Mask columns (present when mask is connected):

    - `image_size_px` — total pixels in the image (H x W); denominator for area_fraction
    - `area_px` — number of foreground pixels
    - `area_fraction` — area_px / image_size_px (0-1); multiply by 100 for %
    - `perimeter_px` — boundary length in pixels
    - `solidity` — area / convex_hull_area (1 = convex)
    - `eccentricity` — shape elongation (0 = circle, 1 = line)
    - `major_axis_px` — major axis of the fitted ellipse
    - `minor_axis_px` — minor axis of the fitted ellipse
    - `extent` — area / bounding_box_area
    - `euler_number` — 1 = no holes; decreases by 1 per enclosed hole
    - `centroid_y`, `centroid_x` — pixel coordinates of the mask centroid
    
    Intensity columns (present when image is connected, pixel values 0-255):

    - `mean`, `std`, `min`, `max`, `median` — overall grayscale or luminance; restricted to masked region when mask is also connected
    
    Per-channel columns (RGB image with *Per Channel* checked):

    - `mean_r/g/b`, `std_r/g/b`, `min_r/g/b`, `max_r/g/b`
    
    - **Column Prefix** — optional string prepended to all column names.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Input** | `mask` | mask |
| **Output** | `table` | table |

**Properties:** ``

---

### Intensity Profile

Plots pixel intensity along an interactively drawn line segment.

??? note "Details"
    Draw the line directly on the image preview. The plot shows intensity (or per-channel
    R/G/B) vs distance in pixels. Useful for measuring gradients, checking membrane
    sharpness, or verifying stain distribution across tissue layers.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `figure` | figure |

**Properties:** ``

---

### Find Contours

Finds all contours in a binary mask or edge image at a given intensity level.

??? note "Details"
    Outputs:

    - *mask* — binary image with all selected contours drawn as lines
    - *table* — coordinate table with columns `contour_id`, `x`, `y`
    
    Modes:

    - *All contours* — return every contour found, sorted largest first
    - *Largest only* — return only the contour with the greatest enclosed area
    - *Filter by min area* — discard contours with enclosed area below **Min Area**
    
    - **Contour Level** — intensity threshold for contour detection (normalised 0-1).
    - **Line Width** — stroke width in pixels for the output mask drawing.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image/mask` | image |
| **Output** | `mask` | mask |
| **Output** | `table` | table |

**Properties:** `Contour Level (0–1)`, `Line Width (px)`, `Mode`, `Min Area (px²)`

---

### Hough Circles

Detects circles in a Canny edge image using the Hough circle transform.

??? note "Details"
    Sweeps a range of radii and votes for circle centres; peaks in the accumulator
    become detections. Connect a CannyEdgeNode output to the input.
    
    Outputs:

    - *overlay* — RGB image with detected circles drawn in green
    - *table* — columns `cx`, `cy`, `radius` for every detected circle
    
    - **Min Radius** — smallest circle radius to search for (pixels).
    - **Max Radius** — largest circle radius to search for (pixels).
    - **Threshold** — fraction of the peak accumulator value required for a detection.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `overlay` | image |

**Properties:** `Min Radius (px)`, `Max Radius (px)`, `Max Circles`, `Threshold (0–1)`

---

### Hough Lines

Detects straight lines in a Canny edge image using the Hough line transform.

??? note "Details"
    Each detected line is described by (`theta`, `rho`): the perpendicular angle and
    distance from the image origin. Lines are extended to the full image boundary for
    the overlay. Connect a CannyEdgeNode output to the input.
    
    Outputs:

    - *overlay* — RGB image with detected lines drawn in red
    - *table* — columns `theta` (rad), `rho` (px), and endpoint coordinates `x0`, `y0`, `x1`, `y1`
    
    - **Threshold** — fraction of the peak accumulator value required for a detection.
    - **Min Distance** — minimum pixel separation between detected lines.
    - **Min Angle** — minimum angular separation in degrees between detected lines.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `overlay` | image |

**Properties:** `Max Lines`, `Threshold (0–1)`, `Min Distance (px)`, `Min Angle (deg)`

---

### Hough Ellipse

Detects ellipses in a Canny edge image using the Hough ellipse transform.

??? note "Details"
    Slow on large images -- resize input to under 300x300 px for best speed. Uses
    `skimage.transform.hough_ellipse`.
    
    Outputs:

    - *overlay* — RGB image with detected ellipses drawn in cyan
    - *table* — columns `cx`, `cy`, `a` (semi-major), `b` (semi-minor), `orientation` (rad)
    
    - **Min Semi-Major** — smallest semi-major axis to search for (pixels).
    - **Max Semi-Major** — largest semi-major axis to search for (pixels).
    - **Accuracy** — step size in pixels for the accumulator; larger = faster but coarser.
    - **Threshold** — fraction of the peak accumulator value required for a detection.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `overlay` | image |

**Properties:** `Min Semi-Major (px)`, `Max Semi-Major (px)`, `Accuracy (px step)`, `Max Ellipses`, `Threshold (0–1)`

---

### Image Histogram

Plots the pixel intensity histogram of an image.

??? note "Details"
    Outputs a figure showing intensity distribution per channel (R/G/B for colour images,
    a single Intensity curve for grayscale). Optionally accepts a mask to restrict the
    histogram to the masked region only. Also outputs a table with columns `Pixel_Value`
    and one column per channel.
    
    - **Bins** — number of histogram bins (default 256; auto-capped at max pixel value + 1).
    - **Log Y-axis** — show frequency on a log scale.
    - **Fill Alpha** — line / fill opacity (0.0-1.0).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Input** | `mask` | mask |
| **Output** | `figure` | figure |
| **Output** | `table` | table |

**Properties:** `Bins`, ``, `Fill Alpha`

---
