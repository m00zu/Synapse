# Utility

### Editable Table

Displays an input table in an editable spreadsheet widget and outputs the modified result.

??? note "Details"
    The node accepts a TableData input and presents it in an Excel-like spreadsheet
    where you can type anywhere to add data, double-click column headers to rename
    them, single-click headers to sort, and right-click for insert/delete operations.
    Copy/paste and Delete key are supported. Changes are pushed downstream automatically.
    
    Parameters:

    - **Reset Edits on Next Run** — when checked, discards local edits and reloads from upstream on the next evaluation

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** `Reset Edits on Next Run`

---

### Extract Object

Extracts a single object (image, figure, label, etc.) from a table's object column.

??? note "Details"
    After batch-accumulating images or figures, this node lets you pick one
    item by row index and outputs it as its original data type.
    
    Parameters:

    - **Row Index** — 1-based row number to extract from
    - **Object Column** — name of the column containing the objects (default: `object`)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | any |

**Properties:** `Row Index (1-based)`

---

### Blank Subtract

Subtract a reference row's value from all rows in a column.

??? note "Details"
    Common use: subtract background (BG) measurement from all cells.
    
    - **Reference Column** — the column containing group/cell labels (e.g. 'cell')
    - **Reference Value** — the label of the reference row (e.g. 'BG')
    - **Target Columns** — columns to subtract from (comma-separated, or leave empty for all numeric)
    
    The reference row's value is subtracted from every row in each target column.
    The reference row itself is kept (will become 0).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `table` | table |

---

### Row Normalize

Normalize all rows by a reference row's value (divide instead of subtract).

??? note "Details"
    Useful for fold-change calculations: value / reference.
    
    - **Reference Column** — column containing group labels
    - **Reference Value** — label of the reference row (e.g. 'Control', 'BG')
    - **Target Columns** — columns to normalize (comma-separated, or empty for all numeric)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `table` | table |

---

### Drop Rows

Drop rows where a column matches any of the specified values.

??? note "Details"
    - **Column** — which column to check
    - **Values to Drop** — comma-separated list of values to remove (e.g. 'BG, Artifact (BG)')

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `table` | table |

---

### Python Script

Run custom Python code with dynamic input and output ports.

??? note "Details"
    Use this node for operations that no dedicated node covers — custom
    formulas, advanced scipy/skimage functions, string parsing, conditional
    logic, or any one-off data transformation.
    
    ### Setup
    
    - **Inputs / Outputs** spinboxes control how many ports the node has.
    - Click **Edit Script…** to open the full code editor (dark theme).
    - The inline preview on the node card shows the current script.
    - `print()` output is shown as a popup after execution.
    
    ### Variables
    
    | Variable | Description |
    |----------|-------------|
    | `in_1`, `in_2`, … | Data from each input port (DataFrame, ndarray, or raw value). Unconnected = `None`. |
    | `out_1`, `out_2`, … | Assign results here to send downstream. |
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
    
    - **Fold-change** (qPCR) — `df['fold_change'] = 2 ** (-df['ddCt'])`:
    
    - `df = in_1.copy()`
    - `df['fold_change'] = 2 ** (-df['ddCt'])`
    - `out_1 = df`
    
    - **Column ratio** — `df['ratio'] = df['intensity'] / df['area']`:
    
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
