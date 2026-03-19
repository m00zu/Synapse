# Filter

### Filter Table

Filters rows in a TableData object using a pandas query string.

??? note "Details"
    Examples:

    - `Area > 100` — keep rows where Area is greater than 100
    - `Area > 50 and Circularity > 0.8` — multiple conditions
    - `Group == "Control"` — match a specific text value
    - `Group != "Background"` — exclude rows
    - `Area > Area.mean()` — compare to column statistics
    - `label in [1, 2, 5]` — match specific values from a list
    
    Uses pandas `DataFrame.query()` syntax. Column names with spaces need backticks: `` `Column Name` > 10 ``

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

---

### Top N

Extracts the top (or bottom) N rows ranked by a numeric column.

??? note "Details"
    Outputs:

    - **top_n** — the selected N rows
    - **rest** — all remaining rows not in top_n
    
    Parameters:

    - **Rank By Column** — numeric column to rank by
    - **N** — number of rows to select
    - **Select** — *Top (largest)* or *Bottom (smallest)*

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `top_n` | table |
| **Output** | `rest` | table |

**Properties:** `N (number of rows)`, `Select`

---

### Column Value Split

Splits a table into two outputs based on whether a column's value matches a list of specified values.

??? note "Details"
    - **Values** — comma-separated. `*` anywhere triggers glob matching:
    - `Control*` — starts with "Control"
    - `*treated` — ends with "treated"
    - `*GFP*` — contains "GFP"
    - Entries without `*` are exact matches
    
    Outputs:

    - **matched** — rows where the column value matches any entry
    - **rest** — all other rows

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `matched` | table |
| **Output** | `rest` | table |

**Properties:** ``

---

### Random Sample

Randomly samples N rows from the input table.

??? note "Details"
    If N exceeds the table size, the full table is returned (no error).
    
    Parameters:

    - **N** — number of rows to draw
    - **Seed** — random seed for reproducibility; leave at `-1` for a different sample each run
    
    Outputs:

    - **sampled** — the N randomly selected rows
    - **rest** — all remaining rows not in the sample

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `sampled` | table |
| **Output** | `rest` | table |

**Properties:** `N (rows to sample)`, `Random Seed (-1=random)`

---

### Drop Duplicates

Removes duplicate rows from a table.

??? note "Details"
    Parameters:

    - **Subset Columns** — comma-separated columns to consider when checking for duplicates. Leave empty to compare all columns.
    - **Keep** — which duplicate to keep: *first* occurrence, *last*, or *none*
    
    Outputs:

    - **unique** — rows after removing duplicates
    - **dropped** — the removed duplicate rows

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `unique` | table |
| **Output** | `dropped` | table |

**Properties:** `Keep`

---
