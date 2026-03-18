# 3D Volume Processing

### 3D Load Z-Stack

Load a multi-page TIFF file as a 3D volume.

??? note "Details"
    Each page in the TIFF becomes one Z-slice.
    Mode "Grayscale" outputs VolumeData (Z, H, W).
    Mode "Color (RGB)" outputs VolumeColorData (Z, H, W, 3).

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `volume` | volume |
| **Output** | `volume_color` | volume_color |

**Properties:** `Z Spacing`, `XY Spacing`

---

### 3D Slice Viewer

Interactive Z-slice browser for 3D volumes.

??? note "Details"
    Accepts volume, volume_mask, or volume_label input.  Use the slider
    to scrub through slices and the axis selector to view XY/XZ/YZ planes.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Input** | `volume_mask` | volume_mask |
| **Input** | `volume_label` | volume_label |

---

### 3D Volume Viewer

Interactive 3D isosurface viewer for volume masks and label volumes.

??? note "Details"
    Extracts meshes via marching cubes and renders them with Three.js.
    For label volumes, each label gets a distinct colour.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Input** | `volume_label` | volume_label |

**Properties:** `Opacity`, ``

---

### 3D Split RGB

Split a 3D color volume (Z, H, W, 3) into R, G, B channel volumes.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_color` | volume_color |
| **Output** | `volume` | volume |
| **Output** | `volume` | volume |
| **Output** | `volume` | volume |
| **Output** | `red` | red |
| **Output** | `green` | green |
| **Output** | `blue` | blue |

---

### 3D Merge RGB

Merge R, G, B grayscale volumes into a single 3D color volume.

??? note "Details"
    Unconnected channels default to zero.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Input** | `volume` | volume |
| **Input** | `volume` | volume |
| **Input** | `red` | red |
| **Input** | `green` | green |
| **Input** | `blue` | blue |
| **Output** | `volume_color` | volume_color |

---

### 3D RGB to Gray

Convert a 3D color volume to grayscale.

??? note "Details"
    Methods: Luminosity (Rec.709), Average, or extract a single channel.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_color` | volume_color |
| **Output** | `volume` | volume |

**Properties:** `Method`

---

### 3D Gaussian Blur

Apply 3D Gaussian blur to a volume.

??? note "Details"
    Sigma can be set independently for Z and XY axes to account for
    anisotropic voxel spacing.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Output** | `volume` | volume |

**Properties:** `Sigma XY`, `Sigma Z`

---

### 3D Invert

Invert a 3D volume (for uint8: 255 − value; for bool: logical NOT).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Output** | `volume` | volume |

---

### 3D Invert Mask

Invert a 3D binary mask (logical NOT).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

---

### 3D Max Projection

Maximum Intensity Projection (MIP) along an axis.

??? note "Details"
    Collapses a 3D volume to a 2D image by taking the max value per pixel.
    Commonly used in fluorescence microscopy to visualize Z-stacks.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Output** | `image` | image |

**Properties:** `Axis`

---

### 3D Min Projection

Minimum Intensity Projection along an axis.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Output** | `image` | image |

**Properties:** `Axis`

---

### 3D Mean Projection

Mean Intensity Projection along an axis.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Output** | `image` | image |

**Properties:** `Axis`

---

### 3D Apply Mask

Apply a 3D mask to a volume — zero out voxels outside the mask.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume` | volume |

---

### 3D Threshold

Threshold a 3D volume to produce a binary volume mask.

??? note "Details"
    Methods: manual value, Otsu auto-threshold, Li auto-threshold.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Method`, `Threshold`

---

### 3D Distance Ring Mask

Expand a 3D mask outward by a given distance (ring / shell mask).

??? note "Details"
    Uses the Euclidean distance transform.  The *spacing-aware* option
    accounts for anisotropic voxel dimensions (e.g. Z ≠ XY).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Distance (px)`, ``, ``

---

### 3D Remove Small Obj

Remove small 3D connected components from a volume mask.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Min Size (voxels)`, `Connectivity`

---

### 3D Fill Holes

Fill small holes / voids inside a 3D volume mask.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Max Hole Size (voxels)`

---

### 3D Erode

3D morphological erosion with ball / cube / octahedron kernel.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Radius (voxels)`, `Kernel`

---

### 3D Dilate

3D morphological dilation with ball / cube / octahedron kernel.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Radius (voxels)`, `Kernel`

---

### 3D Open

3D morphological opening (erosion → dilation).  Removes small protrusions.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Radius (voxels)`, `Kernel`

---

### 3D Close

3D morphological closing (dilation → erosion).  Fills small gaps.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Radius (voxels)`, `Kernel`

---

### 3D Label

Label connected components in a 3D binary volume.

??? note "Details"
    Outputs a label volume (integer per region) and a properties table
    with volume, centroid, bounding box, and equivalent diameter.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_label` | volume_label |
| **Output** | `table` | table |

**Properties:** `Connectivity`

---

### 3D Watershed

3D marker-based watershed to separate touching objects.

??? note "Details"
    Pipeline: distance transform → peak detection → watershed.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_label` | volume_label |
| **Output** | `table` | table |

**Properties:** `Min Object Sep. (px)`

---
