# Transform

### Rename Group

Renames values in a target column based on a mapping string or the built-in mapping table.

??? note "Details"
    Mapping syntax:

    - `OldName : NewName` — rename a single value
    - `OldA | OldB : Combined` — merge multiple values into one
    - Comma-separated for multiple rules: `A : Control, B : Treated`
    
    Example: Target Column = `Group`, Mapping = `ctrl : Control, exp1 | exp2 : Experimental`
    
    | Group (before) | Group (after)  |
    |----------------|----------------|
    | ctrl           | Control        |
    | exp1           | Experimental   |
    | exp2           | Experimental   |
    
    You can also use the mapping table widget below the text input for a visual editor.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

---

### Reshape Table

Converts a table between wide and long format.

??? note "Details"
    Modes:

    - *Wide to Long* (melt) — unpivots multiple value columns into rows
    - *Long to Wide* (pivot) — spreads row values back into columns
    - *Collect by Group* — gathers values by group into side-by-side columns
    
    **Wide to Long example:**
    
    | Sample | Ch1 | Ch2 | Ch3 |
    |--------|-----|-----|-----|
    | A      | 10  | 20  | 30  |
    | B      | 15  | 25  | 35  |
    
    Settings: ID Columns = `Sample`, Group Column = `Channel`, Value Column = `Intensity`
    
    | Sample | Channel | Intensity |
    |--------|---------|-----------|
    | A      | Ch1     | 10        |
    | A      | Ch2     | 20        |
    | A      | Ch3     | 30        |
    | B      | Ch1     | 15        |
    | B      | Ch2     | 25        |
    | B      | Ch3     | 35        |
    
    **Long to Wide** reverses the above. Settings: Index Columns = `Sample`, Pivot Column = `Channel`, Value Column = `Intensity`
    
    Parameters:

    - **ID Columns** — columns to keep as-is (comma-separated). Leave empty to melt all non-numeric columns.
    - **Value Columns** — which columns to unpivot (leave empty = all remaining).
    - **Group Column Name** — name for the new column holding the original column names (default: `Group`).
    - **Value Column Name** — name for the new column holding the values (default: `Value`).
    - **Index Columns (pivot)** — columns that identify each row in the wide output.
    - **Pivot Column (pivot)** — column whose unique values become new column headers.
    - **Value Column (pivot)** — column whose values fill the new columns.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** `Mode`

---

### Sort Table

Sorts a table by one or more columns in ascending or descending order.

??? note "Details"
    Parameters:

    - **Sort By** — comma-separated column names to sort by (applied in order)
    - **Order** — ascending or descending

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** `Order`

---

### Select Columns

Keeps only the columns listed in 'Columns' and drops everything else.

??? note "Details"
    - **Columns** — comma-separated list of column names to keep. `*` anywhere in a name triggers glob matching:
    - `*Intensity` — ends with "Intensity"
    - `Intensity*` — starts with "Intensity"
    - `*Intensity*` — contains "Intensity"
    
    - **Drop mode** — when checked, the listed columns are DROPPED instead of kept.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** ``

---

### Drop / Fill NaN

Removes or fills NaN values in a table.

??? note "Details"
    Modes:

    - *Drop rows* — remove any row containing at least one NaN in the specified columns
    - *Fill constant* — replace NaN with a fixed value (e.g. `0` or `"unknown"`)
    - *Fill mean / median / mode* — replace with column statistics
    - *Forward fill / Back fill* — propagate the last or next valid value
    
    - **Columns** — comma-separated column names to act on. Leave empty to apply to all columns.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** `Mode`

---

### Type Cast Column

Converts the data type of one or more columns.

??? note "Details"
    Target types:

    - *float* — convert to floating-point number
    - *int* — convert to integer (rounds, then casts)
    - *str* — convert to string
    - *bool* — convert to boolean (`0`/`False`/`false`/`no` become False, else True)
    - *category* — pandas Categorical (saves memory for low-cardinality columns)
    - *datetime* — parse as datetime using pandas `to_datetime`
    
    Parameters:

    - **Columns** — comma-separated column names to cast
    - **Coerce errors to NaN** — when checked, unparseable values become NaN instead of raising an error

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** `Target Type`, ``

---

### String Column Ops

Applies a string operation to a text column.

??? note "Details"
    Operations:

    - *Strip whitespace* — remove leading/trailing spaces
    - *To upper / To lower / Title case* — change case
    - *Replace* — replace a substring or regex pattern with another string
    - *Extract regex group* — extract first capture group; non-matching rows become NaN
    - *Split to two columns* — split on a delimiter and put left/right parts into two new columns
    - *Pad / Zfill* — left-pad with zeros to a fixed width
    
    Parameters:

    - **Column** — source text column to operate on
    - **Pattern / Delimiter / Width** — context-dependent input for the selected operation
    - **Replace With** — replacement string (for Replace operation)
    - **Result Column** — name for the output column. Leave empty to overwrite the source column.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** `Operation`, ``

---
