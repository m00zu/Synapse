# IO

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
