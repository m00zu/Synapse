# Display

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
