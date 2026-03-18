# IO & Display

### Table Reader

Reads a tabular file (CSV, TSV) using pandas and outputs a DataFrame.

??? note "Details"
    - **file_path** — path to the input file (widget or upstream port).
    - **separator** — column delimiter (default: `,`).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `path` | path |
| **Input** | `file_path` | file_path |
| **Output** | `table` | table |
| **Output** | `out` | out |

---

### Folder Iterator

Selects a folder and file pattern for batch processing.

??? note "Details"
    The actual looping is managed by the Batch Runner in `main.py`.
    
    - **folder_path** — directory to iterate over.
    - **pattern** — glob pattern for matching files (default: `*.csv`).
    - **iterate_mode** — iterate over *Files* or *Subdirectories*.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `file_path` | path |

**Properties:** `Iterate`

---

### Video Iterator

Browses and iterates over frames of a video file.

??? note "Details"
    Preview any frame with the browse slider. Select a start/end range
    with the dual-handle range slider, then use Batch Run to process
    each frame through the downstream graph.
    
    - **video_path** — path to the video file.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `file_path` | path |

---

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

### Image Reader

Reads an image file and outputs it as a PIL Image.

??? note "Details"
    Supported formats:

    - *Standard* — JPEG, PNG, BMP, and other PIL-supported formats
    - *TIFF* — 12/16-bit microscopy TIFFs (auto-normalized to 8-bit)
    - *OIR* — Olympus .oir files (via `oir_reader_rs` extension)
    
    - **file_path** — path to the image file (widget or upstream port).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `path` | path |
| **Input** | `file_path` | file_path |
| **Output** | `image` | image |
| **Output** | `out` | out |

---

### Data Saver

Saves incoming data to a file on disk.

??? note "Details"
    Supported output types:

    - *DataFrame* — saved as CSV, TSV, or `.pzfx` (GraphPad Prism)
    - *Figure* — saved as an image at the figure's native DPI
    - *Image* — saved via PIL in any format matching the file extension
    
    - **file_path** — destination path (widget or upstream port).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `any` | any |
| **Input** | `path` | path |
| **Input** | `in` | in |
| **Input** | `file_path_in` | file_path_in |

---

### Batch Gate

Pass-through gate that pauses the batch pipeline for user review.

??? note "Details"
    Wire between any two nodes using the single `any`-typed input/output.
    Blocking happens inside `evaluate()`, so multiple gates pause
    independently at their own step in the topological evaluation order.
    
    Controls:

    - *Next* — let this iteration continue past the gate
    - *Refresh* — re-evaluate upstream nodes and update previews
    - *Pass All* — stop pausing for the rest of this batch run

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |
| **Output** | `out` | any |

---

### Pop-up Display

Takes any input and pops up a preview window to inspect it.

??? note "Details"
    Supported data types:

    - *DataFrame* — shown as an editable table dialog
    - *Figure* — rendered to PNG and shown in a scrollable image dialog
    - *Image* — displayed as a scrollable PIL image dialog
    - *Other* — shown as a plain text message box

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |

---

### Data Table Node

Displays incoming DataFrame data directly on the node surface.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |

---

### Data Figure Node

Displays incoming Image or Figure data directly on the node surface.

??? note "Details"
    Accepts `FigureData` (with optional SVG override) or raw matplotlib figures.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | figure |

---

### Image Viewer

Displays a PIL Image directly on the node surface for quick inline inspection.

??? note "Details"
    Accepted input types:

    - *ImageData* — unwraps the payload
    - *LabelData* — uses the pre-generated colored visualization
    - *Raw PIL Image* — displayed as-is

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | image |

---

### SVG Editor

Converts an upstream matplotlib Figure to SVG for interactive element editing.

??? note "Details"
    Usage:

    - Click any highlighted element to select it.
    - Double-click to open the properties panel (fill, stroke, opacity, etc.).
    - Drag text labels (orange cursor) to reposition them.
    - Click "Apply" in the properties panel to commit changes.
    - Click "Reset SVG" to discard edits and reload from the figure.
    
    Edits are stored in the `_svg_data` node property and survive
    re-evaluation as long as the upstream figure is unchanged. Reset SVG
    clears them.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | figure |
| **Output** | `out` | figure |

---

### Universal Node

Executes arbitrary Python code to process multiple inputs and push results to outputs.

??? note "Details"
    Available variables in user code:

    - `inputs` — list of upstream data values
    - `output` — assign the result here (auto-wrapped into `TableData`, `ImageData`, or `FigureData`)
    - `pd`, `np`, `plt`, `sns` — pre-imported libraries

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |
| **Output** | `out` | any |

---

### Path Modifier

Takes a file path and modifies it by adding a suffix, changing the extension, or overriding the folder.

??? note "Details"
    - **suffix** — string appended to the file stem (default: `_analyzed`).
    - **ext** — replacement file extension (leave empty to keep original).
    - **folder** — optional folder override for the output path.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `path` | path |
| **Output** | `path` | path |

---
