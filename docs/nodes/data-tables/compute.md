# Compute

### Single Table Math

Creates or modifies a column using a pandas eval expression.

??? note "Details"
    Examples:

    - `Ratio = Intensity_Ch1 / Intensity_Ch2` — create a new column
    - `Area_um2 = Area * 0.065 * 0.065` — convert pixels to physical units
    - `Normalized = Intensity / Intensity.mean()` — normalize to mean
    - `Log_Area = @np.log10(Area)` — use numpy functions with `@` prefix
    
    Uses pandas `DataFrame.eval()` syntax. The left side of `=` is the new column name.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

---

### Aggregate Table

Reduces a table to aggregate statistics across rows, optionally grouped by column.

??? note "Details"
    Without grouping, all numeric columns are reduced to a single row.
    With grouping, each unique group gets its own summary row.
    
    | Group   | Area |
    |---------|------|
    | Control | 110  |
    | Treated | 190  |
    
    Parameters:

    - **Operation** — sum, mean, median, min, max, count, std, var
    - **Group By** — column name(s) to group by (comma-separated, leave empty for no grouping)
    - **Columns** — restrict to specific columns (comma-separated, leave empty = all numeric)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** `Operation`

---

### Two Table Math

Computes a scalar arithmetic operation between one value from each of two input tables.

??? note "Details"
    For each input table the node picks the first numeric column (or the column
    named in the matching property) and uses row 0 as the scalar value.
    Designed for comparing scalar outputs such as stained-area measurements.
    
    Outputs a single-row result table: `left_value | right_value | operation | result`
    
    Parameters:

    - **Operation** — `left / right`, `left * right`, `left + right`, or `left - right`
    - **Left Column** — column name in the left table (blank = first numeric)
    - **Right Column** — column name in the right table (blank = first numeric)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `left` | table |
| **Input** | `right` | table |
| **Output** | `result` | table |

**Properties:** `Operation`

---

### Normalize Column

Normalizes one or more numeric columns.

??? note "Details"
    Methods:

    - *Min-Max (0-1)* — scales each column to [0, 1]
    - *Z-score* — subtracts mean and divides by std (standard score)
    - *Log10 / Log2 / Ln* — log transform (adds 1 before log to handle zeros)
    - *Robust (IQR)* — subtracts median, divides by IQR; robust to outliers
    
    Parameters:

    - **Columns** — comma-separated names. Leave empty to normalize all numeric columns.
    - **Suffix** — text appended to new column names (e.g. `_norm`). Leave empty to overwrite in-place.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** `Method`

---

### Value Counts

Counts occurrences of each unique value in a column.

??? note "Details"
    Outputs a two-column table with the original column name and a `count` column,
    sorted by count descending by default.
    
    Parameters:

    - **Column** — the column to count unique values in
    - **Sort by count (descending)** — sort results by frequency
    - **Add percentage column** — include a `pct` column with relative frequencies

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

**Properties:** ``, ``

---

### Group Normalization

Normalizes numerical columns relative to a specified control group mean.

??? note "Details"
    Each numeric column is divided by the mean of its corresponding control group,
    producing fold-change values. A mapping table widget lets you assign a different
    control group for each unique group in the data.
    
    Parameters:

    - **Global Control Group** — default control group name used for normalization
    - **Target Column** — column containing group labels (e.g. `Group`)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `in` | table |
| **Output** | `out` | table |

---
