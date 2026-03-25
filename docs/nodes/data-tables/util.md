# Utility

### Editable Table

Displays an input table in an editable spreadsheet widget and outputs the modified result.

??? note "Details"
    The node accepts a TableData input and presents it in an interactive QTableWidget
    where you can edit cell values, add/remove rows and columns, rename headers, and
    sort by clicking column headers. Changes are pushed downstream automatically.
    
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
