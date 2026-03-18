# SAM2 Segmentation

### SAM2 Segment

Interactive SAM2 segmentation — click to include/exclude regions.

??? note "Details"
    Connect an image, then click on objects to segment them.
    Use "+ Object" to add multiple objects, each with a distinct label.
    Toggle between Include (+foreground) and Exclude (−background) modes
    with the toolbar button.
    
    Outputs: binary mask (union of all objects), integer label image
    (each object = unique label), and a colored overlay.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `mask` | mask |
| **Output** | `label_image` | label_image |
| **Output** | `overlay` | overlay |

**Properties:** `Model`

---

### Grounding DINO

Detect objects by text description using GroundingDINO.

??? note "Details"
    Type a text query (e.g. "nucleus" or "cell, membrane") and get
    bounding-box detections as rectangular masks.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `mask` | mask |
| **Output** | `label_image` | label_image |
| **Output** | `overlay` | overlay |

**Properties:** `Min Score`

---

### SAM2 Text Segment

Text-prompted segmentation using GroundingDINO + SAM2.

??? note "Details"
    Type a text description to detect and segment objects.
    Chains GroundingDINO (text → boxes) with SAM2 (boxes → precise masks).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Output** | `mask` | mask |
| **Output** | `label_image` | label_image |
| **Output** | `overlay` | overlay |

**Properties:** `Min Score`, `SAM2 Model`

---
