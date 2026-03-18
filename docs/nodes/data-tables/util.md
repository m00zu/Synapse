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
