# Convert

### MolTable to Molecule

Pick a single molecule from a MolTable by row index.

??? note "Details"
    Bridges batch (mol_table) to single-molecule nodes (molecule port).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `molecule` | molecule |

**Properties:** `Row Index`

---

### MolTable to Table

Convert a MolTable to a plain Table by dropping the Mol column.

??? note "Details"
    Useful for connecting to existing table nodes (Sort, Filter, Plot, etc.).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `table` | table |

---

### Substructure Filter

Filter rows of a SMILES table by a SMARTS substructure pattern.

??? note "Details"
    Outputs two tables: matches (has the substructure) and rejects (does not).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `matches` | table |
| **Output** | `rejects` | table |

---

### Table to MolTable

Convert a TableData (DataFrame) to a MolTable by parsing a SMILES column.

??? note "Details"
    Mirrors the behaviour of *MolTable Reader*, but reads from an upstream
    table port rather than a file.
    
    If **ID Column** is left blank (or the named column is missing),
    identifiers are auto-generated as ``Mol_1``, ``Mol_2``, ....
    
    **Property Columns** is a comma-separated list of extra columns to
    carry through (e.g. ``activity, pIC50``).  Leave blank to skip.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `mol_table` | mol_table |

**Properties:** `Workers`

---
