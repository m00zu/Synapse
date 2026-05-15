# Exposure

### 3D Apply Mask

Apply a 3D mask to a volume -- zero out voxels outside the mask.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `volume` | volume |
| **Input** | `volume_mask` | volume_mask |
| **Output** | `volume` | volume |

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

### 3D Mean Projection

Mean Intensity Projection along an axis.

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
