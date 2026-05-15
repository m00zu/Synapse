# IO

### IUPAC to SMILES

Resolve a column of chemical names / IUPAC strings to SMILES via PubChem.

??? note "Details"
    Sends each name to PubChem PUG REST
    (``https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/<name>/property/CanonicalSMILES/TXT``)
    in a thread pool and parses the returned SMILES into RDKit Mols.  Outputs a
    standard MolTable (with ``name`` / ``smiles`` / ``ROMol`` columns and any
    user-selected property columns carried through).
    
    PubChem allows up to 5 requests/second per IP -- keep ``Workers`` ≤ 5.
    
    **Network access required.**  Names that fail to resolve (HTTP 404,
    timeout, or invalid SMILES) are dropped.
    
    Property Columns: comma-separated list of extra columns from the input
    table to carry forward.  Leave blank to keep only name / smiles / ROMol.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `mol_table` | mol_table |

**Properties:** `Workers`, `Timeout (s)`

---

### Mol Reader

Read molecules from a file or all files in a directory.

??? note "Details"
    Supported formats: SDF, SMI, CSV/TSV, MOL, MOL2, PDB, XYZ.
    Uses RDKit's threaded suppliers (MultithreadedSDMolSupplier,
    MultithreadedSmilesMolSupplier) for SDF and SMILES files.
    For directories, reads all matching files with ThreadPoolExecutor.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `mol_table` | mol_table |

**Properties:** `Source`

---

### MolTable Reader

Read a tabular file and parse a SMILES column to a MolTable.

??? note "Details"
    Supported formats: CSV, TSV, TXT, XLSX.  The user names which column
    holds the identifier and which holds the SMILES.  SMILES are parsed
    to RDKit Mols in parallel via a process pool.  Rows whose SMILES
    fail to parse are dropped.
    
    If **ID Column** is left blank (or the named column is missing),
    identifiers are auto-generated as ``Mol_1``, ``Mol_2``, ....
    
    **Property Columns** is a comma-separated list of extra columns to
    carry through onto the MolTable (e.g. ``activity, pIC50``).  Leave
    blank to skip.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `mol_table` | mol_table |

**Properties:** `Workers`

---

### SMILES to IUPAC

Resolve a SMILES column to IUPAC names via PubChem.

??? note "Details"
    For each row, queries PubChem PUG REST
    (``/compound/smiles/<smiles>/property/IUPACName/TXT``) to retrieve the
    preferred IUPAC name and stores it in a new column on the output table.
    
    Accepts either a ``mol_table`` (uses its ``smiles`` column) or any
    ``table`` with a SMILES column you select.
    
    PubChem allows up to 5 requests/second per IP -- keep ``Workers`` ≤ 5.
    
    **Network access required.**  Rows whose lookup fails get ``None`` in
    the IUPAC column; the original row is preserved.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `table` | table |
| **Output** | `table` | table |

**Properties:** `Workers`, `Timeout (s)`

---
