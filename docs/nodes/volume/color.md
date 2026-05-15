# Color

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
