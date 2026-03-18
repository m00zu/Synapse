# Combine

### Concat Tables

Concatenates two tables by stacking rows (vertical) or columns (horizontal).

??? note "Details"
    Modes:

    - *Vertical (stack rows)* — both tables should have the same columns; mismatched columns are filled with NaN when 'Fill missing columns' is checked
    - *Horizontal (side by side)* — both tables are placed side by side; shorter side is padded with NaN
    
    Parameters:

    - **Direction** — vertical or horizontal
    - **Fill missing columns with NaN** — when unchecked, only common columns are kept in vertical mode

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `top` | table |
| **Input** | `bottom` | table |
| **Output** | `out` | table |

**Properties:** `Direction`, ``

---

### Join Tables

Merges two tables on a shared key column (like SQL JOIN).

??? note "Details"
    Example: Left key = `particle_id`, Right key = `id` to match particles to metadata.
    
    Parameters:

    - **Key Column (left)** — column name to join on in the left table
    - **Key Column (right)** — column name in the right table (leave blank to use the same name as left)
    - **Join Type** — *inner*, *left*, *right*, or *outer*

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `left` | table |
| **Input** | `right` | table |
| **Output** | `out` | table |

**Properties:** `Join Type`

---
