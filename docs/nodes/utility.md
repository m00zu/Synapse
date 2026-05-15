# Utility

### Cast Type

Relabel data as a different ``NodeData`` subclass without re-parsing.

??? note "Details"
    Useful when a polymorphic pass-through node erases your subtype
    info -- e.g., filtering a ``MolTableData`` through ``Filter Table``
    gives you back ``TableData`` even though the DataFrame still has
    the rdkit Mol column.  A ``Table -> MolTable`` node would re-parse
    every SMILES (slow).  This node just wraps the existing payload in
    the target class -- microseconds, no re-parse.
    
    Targets are populated from the live port-type registry, so any
    plugin-registered type whose data class accepts only a ``payload``
    argument appears in the dropdown automatically.
    
    Light validation: for the common ndarray-typed and DataFrame-typed
    targets, the node checks at evaluate time that the payload looks
    right.  For plugin types, the cast is permissive (the user takes
    responsibility -- this node is a *type assertion*, not a
    conversion).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `data` | any |
| **Output** | `data` | image |

**Properties:** `Cast to`

---

### Data Saver

Saves incoming data to a file on disk.

??? note "Details"
    Supported output types:

    - *DataFrame* -- saved as CSV, TSV, or `.pzfx` (GraphPad Prism)
    - *Figure* -- saved as an image at the figure's native DPI
    - *Image* -- saved via PIL in any format matching the file extension
    
    - **file_path** -- destination path (widget or upstream port).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | any |
| **Input** | `file_path_in` | path |

---

### Path Modifier

Takes a file path and modifies it by adding a suffix, changing the extension, or overriding the folder.

??? note "Details"
    - **suffix** -- string appended to the file stem (default: `_analyzed`).
    - **ext** -- replacement file extension (leave empty to keep original).
    - **folder** -- optional folder override for the output path.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `path` | path |
| **Output** | `path` | path |

---

### Python Script

Run custom Python code with dynamic input and output ports.

??? note "Details"
    Use this node for operations that no dedicated node covers -- custom
    formulas, advanced scipy/skimage functions, string parsing, conditional
    logic, or any one-off data transformation.
    
    ### Setup
    
    - **Inputs / Outputs** spinboxes control how many ports the node has.
    - Click **Edit Script...** to open the full code editor (dark theme).
    - The inline preview on the node card shows the current script.
    - `print()` output is shown as a popup after execution.
    
    ### Variables
    
    | Variable | Description |
    |----------|-------------|
    | `in_1`, `in_2`, ... | Data from each input port (DataFrame, ndarray, or raw value). Unconnected = `None`. |
    | `out_1`, `out_2`, ... | Assign results here to send downstream. |
    | `pd` | pandas |
    | `np` | numpy |
    | `scipy` | scipy (use `scipy.stats`, `scipy.ndimage`, etc.) |
    | `skimage` | scikit-image (use `skimage.filters`, etc.) |
    | `cv2` | OpenCV |
    | `PIL` | Pillow |
    | `plt` | matplotlib.pyplot |
    | `set_progress(0-100)` | Update the node's progress bar during long operations |
    
    You can `import` any additional module installed in your environment.
    
    ### Output types
    
    Results are auto-wrapped: DataFrame → TableData, 2D ndarray → ImageData,
    Figure → FigureData, scalar → single-cell TableData.
    To force a type, use: `out_1 = MaskData(payload=arr)` or `ImageData(payload=arr, bit_depth=16)`.
    
    ### Examples
    
    - **Fold-change** (qPCR) -- `df['fold_change'] = 2 ** (-df['ddCt'])`:
    
    - `df = in_1.copy()`
    - `df['fold_change'] = 2 ** (-df['ddCt'])`
    - `out_1 = df`
    
    - **Column ratio** -- `df['ratio'] = df['intensity'] / df['area']`:
    
    - `df = in_1.copy()`
    - `df['ratio'] = df['intensity'] / df['area']`
    - `out_1 = df`
    
    **Split by median** (set Outputs to 2):
    
    - `med = in_1['value'].median()`
    - `out_1 = in_1[in_1['value'] > med]`
    - `out_2 = in_1[in_1['value'] <= med]`
    
    **Custom scipy test**:
    
    - `from scipy.stats import mannwhitneyu`
    - `g1 = in_1[in_1['group']=='A']['value']`
    - `u, p = mannwhitneyu(g1, g2)`
    - `out_1 = pd.DataFrame({'U': [u], 'p': [p]})`
    
    **Image filter**:
    
    - `from scipy.ndimage import gaussian_filter`
    - `out_1 = gaussian_filter(in_1, sigma=3)`

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in_1` | any |
| **Output** | `out_1` | any |

**Properties:** `Inputs`, `Outputs`

---
