# IO

### Batch Accumulator

Collects the output of each batch iteration and merges them after the batch finishes.

??? note "Details"
    Connect upstream data to the `in` port; the `out` port emits the merged
    result only after the entire batch is complete.
    
    Batch context stamping:

    - Automatically adds `frame` and `file` metadata to each collected value.
    - For `TableData`, this lets downstream nodes identify which frame each row came from.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |
| **Output** | `out` | any |

---

### Batch Gate

Pass-through gate that pauses the batch pipeline for user review.

??? note "Details"
    Wire between any two nodes using the single `any`-typed input/output.
    Blocking happens inside `evaluate()`, so multiple gates pause
    independently at their own step in the topological evaluation order.
    
    Controls:

    - *Next* -- let this iteration continue past the gate
    - *Refresh* -- re-evaluate upstream nodes and update previews
    - *Pass All* -- stop pausing for the rest of this batch run

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |
| **Output** | `out` | any |

---

### Folder Iterator

Selects a folder and file pattern for batch processing.

??? note "Details"
    The actual looping is managed by the Batch Runner in `main.py`.
    
    - **folder_path** -- directory to iterate over.
    - **pattern** -- glob pattern for matching files (default: `*.csv`).
    - **iterate_mode** -- iterate over *Files* or *Subdirectories*.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `file_path` | path |

**Properties:** `Iterate`

---

### Image Reader

Reads an image file and outputs it as a float32 [0,1] numpy array.

??? note "Details"
    Supported formats:
    
    - *Standard* -- JPEG, PNG, BMP, and other PIL-supported formats (8-bit)
    - *TIFF* -- 8/12/14/16-bit microscopy TIFFs (bit depth preserved). Multi-page TIFFs output a CollectionData with one ImageData per page.
    - *OIR* -- Olympus .oir files (Rust accelerated, with Python fallback)
    
    The original bit depth is stored as metadata for downstream nodes
    (threshold sliders, histogram, save). All processing uses float32 [0,1]
    internally.
    
    Options:
    
    - **channels** -- comma-separated channel numbers (0-4, where 0 = black/pad).
      `2` for single grayscale channel,
      `1,2,3` for RGB,
      `2,3,4` to map channels 2/3/4 as R/G/B,
      `1,0,3` to map ch1 as red, black as green, ch3 as blue.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `file_path` | path |
| **Output** | `out` | image |

---

### Table Reader

Reads a tabular file (CSV, TSV) using pandas and outputs a DataFrame.

??? note "Details"
    - **file_path** -- path to the input file (widget or upstream port).
    - **separator** -- column delimiter (default: `,`).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `file_path` | path |
| **Output** | `out` | table |

---

### Video Iterator

Browses and iterates over frames of a video file.

??? note "Details"
    Preview any frame with the browse slider. Select a start/end range
    with the dual-handle range slider, then use Batch Run to process
    each frame through the downstream graph.
    
    - **video_path** -- path to the video file.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `file_path` | path |

---
