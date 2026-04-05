# Morphology

### Remove Small Obj

Removes small connected components from a binary mask.

??? note "Details"
    Objects with area at or below the threshold are discarded. Useful for cleaning noise after thresholding.
    
    - **max_size** — maximum object area in pixels to remove (default: 500).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `mask` | mask |

**Properties:** `Maximum Size (px²)`

---

### Remove Small Holes

Fills small enclosed holes in a binary mask up to a given area threshold.

??? note "Details"
    Only holes (background regions fully enclosed by foreground) smaller than the threshold are filled; larger holes remain. Contrast with FillHolesNode which fills all holes regardless of size.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `mask` | mask |

**Properties:** `Max Hole Size (px²)`

---

### Keep Max Intensity

Keeps the top N connected components ranked by total intensity in the reference image.

??? note "Details"
    Finds all connected components in the mask, sums each region's pixel intensities from the intensity image input, and retains only the brightest regions as a binary mask.
    
    - **top_n** — number of highest-intensity regions to keep (default: 5).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `intensity_image` | image |
| **Output** | `mask` | mask |

**Properties:** `Top N Regions (by area)`

---

### Distance Ring Mask

Creates an annular (ring) mask by expanding foreground objects outward by a specified distance.

??? note "Details"
    Computes the Euclidean distance transform of the background and selects pixels within the given range, producing a ring-shaped region around the original mask.
    
    - **local_distance** — maximum expansion distance in pixels (default: 200).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `ring_mask` | mask |

**Properties:** `Local Distance (px)`, ``

---

### Erode Mask

Applies binary erosion to shrink foreground regions of a mask.

??? note "Details"
    Each iteration removes one layer of boundary pixels. Useful for separating touching objects or removing thin protrusions.
    
    - **iterations** — number of erosion passes (default: 1).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `mask` | mask |

**Properties:** `Iterations`

---

### Dilate Mask

Applies binary dilation to expand foreground regions of a mask.

??? note "Details"
    Each iteration adds one layer of boundary pixels. Useful for bridging small gaps or growing regions uniformly.
    
    - **iterations** — number of dilation passes (default: 1).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `mask` | mask |

**Properties:** `Iterations`

---

### Morph Open

Applies binary opening (erosion then dilation) to a mask.

??? note "Details"
    Removes small foreground objects and thin protrusions while preserving the overall shape of larger regions.
    
    - **iterations** — number of opening passes (default: 1).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `mask` | mask |

**Properties:** `Iterations`

---

### Morph Close

Applies binary closing (dilation then erosion) to a mask.

??? note "Details"
    Fills small holes and connects nearby fragments without significantly changing the overall region size.
    
    - **iterations** — number of closing passes (default: 1).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `mask` | mask |

**Properties:** `Iterations`

---

### Fill Holes

Fills all enclosed background holes in a binary mask regardless of size.

??? note "Details"
    Uses `scipy.ndimage.binary_fill_holes`. Contrast with RemoveSmallHolesNode, which only fills holes smaller than a user-defined area threshold and leaves larger holes intact.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `mask` | mask |

---

### Skeletonize

Reduces foreground regions to 1-pixel-wide centrelines (skeleton).

??? note "Details"
    Uses `skimage.morphology.skeletonize`. Outputs SkeletonData, which can feed SkeletonAnalysisNode or any node that accepts a mask.
    
    - **Method** — *zhang* (default) or *lee*. Lee tends to produce cleaner skeletons on thick blob-like shapes.
    
    - **Prune Spurs** — iteratively removes endpoint pixels (dead-end tips). Setting N removes all branches shorter than N pixels (0 = off).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `skeleton` | skeleton |

**Properties:** `Method`, `Prune Spurs (iterations)`

---

### Skeleton Analysis

Analyses a skeleton produced by SkeletonizeNode.

??? note "Details"
    Outputs:

    - **skeleton** — filtered SkeletonData (short isolated segments removed)
    - **stats** — total `skeleton_length_px` and `junction_count` (after filtering)
    - **junction_image** — RGB visualisation: skeleton in grey, junction pixels in red on black
    - **junction_mask** — binary MaskData of junction pixels only (pass to DrawShapeNode)
    
    - **Min Segment Length** — remove isolated skeleton segments shorter than this many pixels (0 = keep all).
    
    - **Junction Radius** — dilate each junction point into a filled circle of this radius in pixels for the junction_mask output (default: 4).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `skeleton` | skeleton |
| **Output** | `stats` | table |
| **Output** | `junction_image` | image |
| **Output** | `junction_mask` | mask |

**Properties:** `Min Segment Length (px)`, `Skeleton Width (px)`, `Junction Radius (px)`

---

### Medial Axis

Computes the medial axis (distance-based skeleton) of a binary mask.

??? note "Details"
    Uses `skimage.morphology.medial_axis`, which finds the set of pixels equidistant from the nearest background pixel. Unlike skeletonize, it also produces a distance transform that encodes the local radius at each skeleton point.
    
    Outputs:

    - **skeleton** — the 1-pixel-wide medial-axis skeleton (SkeletonData)
    - **distance** — skeleton coloured by local radius using the plasma colourmap (purple = thin, yellow = thick); background is black

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `distance` | image |

---

### Particle Props

Labels connected components in a mask and measures each region.

??? note "Details"
    Shape columns (always present):

    - `label` — integer region ID (matches label_image pixel values)
    - `area` — number of pixels in the region
    - `equivalent_diameter` — diameter of a circle with the same area
    - `centroid_y` / `centroid_x` — pixel coordinates of the region centre
    - `bbox_top` / `bbox_left` / `bbox_bottom` / `bbox_right` — tight bounding box corners
    - `perimeter` — outer boundary length in pixels
    - `circularity` — `4*pi*area/perimeter^2`; 1.0 = perfect circle, lower = more irregular
    - `eccentricity` — 0 = circle, 1 = line; measures elongation
    - `orientation` — angle of major axis in degrees
    - `major_axis` / `minor_axis` — lengths of the fitted ellipse axes
    - `solidity` — area / convex_hull_area; 1 = perfectly convex
    - `extent` — area / bounding_box_area; fraction of bbox filled
    - `euler_number` — 1 = no holes; decreases by 1 for each enclosed hole
    
    Intensity columns (present only when intensity_image is connected):

    - `mean_intensity` — average pixel value inside the region
    - `sum_intensity` — total pixel intensity (mean x area)
    - `max_intensity` / `min_intensity` — brightest and darkest pixels
    - `weighted_centroid_y` / `weighted_centroid_x` — intensity-weighted centre
    
    Outputs:

    - **table** — TableData with all columns above
    - **label_image** — LabelData with integer label array and coloured RGB visualisation

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `intensity_image` | image |
| **Output** | `table` | table |
| **Output** | `label_image` | label_image |

**Properties:** `Min Region Area (px²)`

---

### Particle Select

Filters particles from a LabelData input using dynamic filter rules.

??? note "Details"
    Add filter rules (e.g. area >= 100, circularity >= 0.5) to keep only
    particles matching all conditions. Available properties are auto-detected
    from the upstream data.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `label_image` | label_image |
| **Input** | `image` | image |
| **Output** | `mask` | mask |
| **Output** | `label_image` | label_image |

---

### Visual Particle Select

Selects particles visually by clicking them directly in the label image.

??? note "Details"
    Displays the label image with coloured regions. Click a particle to deselect it (dims); click again to re-select. All / None buttons for bulk operations. Min/Max area filters to exclude tiny debris.
    
    Same inputs and outputs as Particle Select but with visual, spatial interaction instead of a checkbox list.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `label_image` | label_image |
| **Input** | `image` | image |
| **Output** | `mask` | mask |
| **Output** | `label_image` | label_image |

---

### Particle Classify

Assigns particles to named groups by drawing shapes around them.

??? note "Details"
    ### Usage
    
    - Connect a **label image** (from Particle Props or Watershed).
    - Click **+ New Group** to create groups (Group 1, Group 2, …).
    - Select a group, then draw a **rect** or **lasso** around particles.
    - All labels fully enclosed by the shape are assigned to that group.
    - Unassigned labels remain as "Ungrouped" (label 0 in output).
    
    ### Outputs
    
    - **label_image** — relabeled: Group 1 → 1, Group 2 → 2, ungrouped → 0.
    - **table** — original particle table with an added `group` column.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `label_image` | label_image |
| **Input** | `image` | image |
| **Output** | `label_image` | label_image |
| **Output** | `table` | table |

---

### White Top-Hat

Applies a morphological white top-hat filter to extract small bright features.

??? note "Details"
    Subtracts the morphological opening (background estimate) from the original image,
    leaving only bright structures that fit inside the disk of the given radius. Useful
    for equalising uneven illumination or removing broad bright background before
    thresholding. Works on grayscale and each channel of RGB independently.
    
    - **Disk Radius** — radius of the structuring element in pixels.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Disk Radius (px)`

---

### Black Top-Hat

Applies a morphological black top-hat filter to extract small dark features.

??? note "Details"
    Subtracts the original image from its morphological closing, revealing small dark
    structures (holes, valleys, cracks) on a bright background. The complement of
    White Top-Hat -- use White Top-Hat for bright features on dark backgrounds. Works
    on grayscale and each channel of RGB independently.
    
    - **Disk Radius** — radius of the structuring element in pixels.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `image` | image |

**Properties:** `Disk Radius (px)`

---

### Watershed

Separates touching or overlapping objects using marker-controlled watershed segmentation.

??? note "Details"
    Pipeline:

    - Compute Euclidean distance transform of the binary mask.
    - Find local maxima (object centres) in the distance map; **Min Object Sep.** controls the minimum allowed gap between two peaks.
    - Run watershed on the inverted distance map to delineate each object.
    
    Table columns:

    - `label` — integer region ID (matches label_image pixel values)
    - `area` — number of pixels in the region
    - `equivalent_diameter` — diameter of a circle with the same area (`sqrt(4*area/pi)`)
    - `centroid_y`, `centroid_x` — pixel coordinates of the region centre
    - `perimeter` — outer boundary length in pixels
    - `circularity` — `4*pi*area/perimeter^2`; 1.0 = perfect circle, lower = more irregular
    - `eccentricity` — 0 = circle, 1 = line; measures elongation
    - `orientation` — angle of major axis in degrees; 0 = right, +90 = up, -90 = down
    - `major_axis` — length of longest axis of the fitted ellipse
    - `minor_axis` — length of shortest axis of the fitted ellipse
    - `solidity` — area / convex_hull_area; 1 = perfectly convex, <1 = concave
    - `extent` — area / bounding_box_area; fraction of bounding box filled
    - `euler_number` — 1 = no holes; decreases by 1 for each enclosed hole

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mask` | mask |
| **Output** | `label_image` | label_image |
| **Output** | `table` | table |

**Properties:** `Min Object Sep. (px)`

---
