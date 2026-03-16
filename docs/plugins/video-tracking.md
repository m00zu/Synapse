# Video & Tracking

### SAM2 Track

Track objects across timelapse frames using SAM2.

??? note "Details"
    Connect reference masks from SAM2 Segment (frame 1) and an image
    stream (e.g. VideoIterator or FolderIterator → ImageReader).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Input** | `label_image` | label_image |
| **Output** | `mask` | mask |
| **Output** | `label_image` | label_image |
| **Output** | `overlay` | overlay |

**Properties:** `Model`, `Mode`, `Min Score`, `Min IoU`, `Dormant Frames`, `Appearance Wt`

---

### Video to Frames

Extract frames from a video file and save as numbered images.

??? note "Details"
    Select a video file and an output folder.  Run the node once to
    export all (or a range of) frames as individual image files.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `folder_path` | path |

**Properties:** `Format`

---

### Particle Linker

Link particle detections across video frames into tracks.

??? note "Details"
    Takes a regionprops table (frame, centroid_y, centroid_x columns) and
    assigns a track_id to each detection so the same physical particle
    shares one ID across all frames.
    
    Uses nearest-neighbor linking (Hungarian algorithm) with gap-closing.
    Unlinked detections get track_id = -1.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `table` | table |

**Properties:** `Max Displacement (px)`, `Max Gap (frames)`, `Min Track Length`, `Area Weight`

---

### Track Properties

Compute per-track statistics from a linked particle table.

??? note "Details"
    Input must have track_id, frame, centroid_y, centroid_x columns
    (output of Particle Linker).
    
    Output columns: track_id, n_frames, duration, total_path,
    net_displacement, confinement_ratio, mean_speed, max_speed,
    diffusion_coeff, alpha (anomalous exponent).
    
    Motion type interpretation of alpha:
      alpha ≈ 1  → Brownian diffusion (random walk)
      alpha < 1  → Confined / sub-diffusion (tethered, corralled)
      alpha > 1  → Directed / super-diffusion (motor-driven transport)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `table` | table |

---

### Trajectory Plot

Draw particle trajectories as colored paths on a background image.

??? note "Details"
    Connect a reference image (e.g. first frame or overlay) and the
    linked table from Particle Linker. Each track gets a distinct color.
    
    Parameters:
      Tail Frames — 0 = show full history; N = show last N frames only
      Color By    — track_id (fixed color) or speed (red=fast, green=slow)
      Show IDs    — label each track with its ID number

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `image` | image |
| **Input** | `table` | table |
| **Output** | `image` | image |

**Properties:** `Tail Frames (0=all)`, `Line Width`, `Color By`, `Show IDs`

---

### Track Filter

Filter a linked particle table by per-track statistics.

??? note "Details"
    Removes rows belonging to tracks that fall outside the specified
    min/max bounds. Unlinked rows (track_id = -1) are always kept unless
    'Drop Unlinked' is checked.
    
    Set any bound to 0 to disable that limit.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `table` | table |

**Properties:** `Min Length (frames)`, `Max Length (frames)`, `Min Net Disp (px)`, `Max Net Disp (px)`, `Min Mean Speed`, `Max Mean Speed`, `Drop Unlinked (track_id=-1)`

---

### MSD Analysis

Compute Mean Squared Displacement (MSD) vs lag time for all tracks.

??? note "Details"
    MSD(τ) = <|r(t+τ) − r(t)|²> averaged over all track-time-origin pairs.
    
    Output table columns: lag, msd_ensemble, n_samples, plus per-track
    columns if 'Per Track' is checked.
    
    Interpretation:
      log-log slope (alpha) ≈ 1 → Brownian diffusion
      alpha < 1              → Confined motion (sub-diffusion)
      alpha > 1              → Directed transport (super-diffusion)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `table` | table |
| **Output** | `plot` | plot |

**Properties:** `Max Lag (frames, 0=auto)`, `Per-Track MSD columns`

---
