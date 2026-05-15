# Morphology

### 3D Close

3D morphological closing (dilation → erosion).  Fills small gaps.

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

### 3D Erode

3D morphological erosion with ball / cube / octahedron kernel.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Radius (voxels)`, `Kernel`

---

### 3D Fill Holes

Fill small holes / voids inside a 3D volume mask.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Max Hole Size (voxels)`

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

### 3D Open

3D morphological opening (erosion → dilation).  Removes small protrusions.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Radius (voxels)`, `Kernel`

---

### 3D Remove Small Obj

Remove small 3D connected components from a volume mask.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume_mask` | volume_mask |

**Properties:** `Min Size (voxels)`, `Connectivity`

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
