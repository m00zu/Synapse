# IO & Display

### Table Reader

Reads a tabular file (CSV, TSV) using pandas and outputs a DataFrame.

??? note "Details"
    - **file_path** — path to the input file (widget or upstream port).
    - **separator** — column delimiter (default: `,`).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `file_path` | path |
| **Output** | `out` | table |

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

Reads an image file and outputs it as a float32 [0,1] numpy array.

??? note "Details"
    Supported formats:
    
    - *Standard* — JPEG, PNG, BMP, and other PIL-supported formats (8-bit)
    - *TIFF* — 8/12/14/16-bit microscopy TIFFs (bit depth preserved)
    - *OIR* — Olympus .oir files (Rust accelerated, with Python fallback)
    
    The original bit depth is stored as metadata for downstream nodes
    (threshold sliders, histogram, save). All processing uses float32 [0,1]
    internally.
    
    Options:
    
    - **channels** — comma-separated channel numbers (0-4, where 0 = black/pad).
      `2` for single grayscale channel,
      `1,2,3` for RGB,
      `2,3,4` to map channels 2/3/4 as R/G/B,
      `1,0,3` to map ch1 as red, black as green, ch3 as blue.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `file_path` | path |
| **Output** | `out` | image |

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
| **Input** | `in` | any |
| **Input** | `file_path_in` | path |

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

### Collect

Pack multiple data items into a named collection.

??? note "Details"
    Connect any number of items to the multi-input port. Each connection
    gets a name (auto-populated from the upstream port name, editable).
    The output is a single CollectionData that flows as one wire.
    
    Downstream nodes that expect a single item will automatically loop
    over all items in the collection and repack the results.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |
| **Output** | `collection` | collection |

---

### Select Collection

Extract a single item from a collection by name.

??? note "Details"
    Type a name or pick from the dropdown. The dropdown auto-populates
    with available item names when the collection is connected.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `out` | any |

---

### Pop Collection

Extract one item from a collection and output the rest separately.

??? note "Details"
    Two outputs: the extracted item on **item**, and a new collection
    without that item on **rest**.  Type a name or pick from the dropdown.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `item` | any |
| **Output** | `rest` | collection |

---

### Split Collection

Split a collection into two groups by selecting which items go to each output.

??? note "Details"
    Type item names separated by ' | ' or pick from the dropdown to add.
    Selected items go to **selected**, the rest go to **rest**.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `selected` | collection |
| **Output** | `rest` | collection |

---

### Save Collection

Saves all items in a collection to disk.

??? note "Details"
    Each item is saved as a separate file using the item name as a suffix.
    Supports images (TIFF, PNG), tables (CSV, TSV), and figures.
    
    If a path is connected, it is used as the base — the item name is inserted
    before the extension.  Otherwise the folder + extension fields are used.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `file_path` | path |
| **Output** | `status` | table |

---

### Rename Collection

Rename items in a collection using a visual mapping table.

??? note "Details"
    When a collection is connected, the table auto-populates with original
    names. Edit the 'New Name' column to rename items. Leave blank to keep
    the original name.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `collection` | collection |

---

### Collection Info

Outputs a table listing item names, types, shapes, and metadata.

??? note "Details"
    All number and string valued metadata fields are included as extra columns.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `info` | table |

---

### Filter Collection

Keep or remove items by pattern matching on names.

??? note "Details"
    Supports simple wildcards (* and ?) or exact names.
    Multiple patterns separated by | (pipe).
    
    Mode:

    - *Keep* — only matching items pass through
    - *Remove* — matching items are excluded

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `matched` | collection |
| **Output** | `rest` | collection |

**Properties:** `Mode`

---

### Map Names

Batch rename collection items using find/replace, prefix, or suffix.

??? note "Details"
    Operations (applied in order):
    1. Find/Replace — replace substring in all names
    2. Prefix — add text before each name
    3. Suffix — add text after each name

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `collection` | collection |
| **Output** | `collection` | collection |

---
