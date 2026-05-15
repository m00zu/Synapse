# Filters

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
