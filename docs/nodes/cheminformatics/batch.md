# Batch

### Add Hydrogens

Add explicit hydrogen atoms to every molecule in a MolTable.

??? note "Details"
    Uses ``Chem.AddHs(mol, addCoords=True)`` -- when 3D conformers are present,
    the new H atoms get reasonable positions; for 2D / no-conformer mols the
    flag is a no-op and Hs are added without coordinates.
    
    Useful before docking (PDBQT requires explicit Hs) or before any node
    that needs a hydrogen-complete representation (Mol 3D Embed, GNINA, etc.).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

---

### Batch 3D Embed

Embed all molecules in a MolTable in 3D using ETKDGv3.

??? note "Details"
    Failed embeddings are dropped from the output table.
    Uses ThreadPoolExecutor for parallelism (RDKit releases the GIL
    during embedding/optimisation).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

**Properties:** `Keep Hydrogens`, `Optimize`, `Force Field`, `Timeout sec (0=none)`, `Random Coords Fallback`

---

### Batch Catalog Filter

Filter a MolTable using RDKit's built-in structural-alert catalogs.

??? note "Details"
    Enable one or more catalogs (PAINS, BRENK, NIH, ZINC, CHEMBL variants).
    A molecule is flagged if it matches *any* enabled catalog.
    
    Include mode keeps clean molecules (no alerts); Exclude mode keeps
    only flagged molecules.
    
    Outputs two MolTables: *matches* (kept) and *rejects* (removed).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `matches` | mol_table |
| **Output** | `rejects` | mol_table |

**Properties:** `Mode`, ``

---

### Batch Descriptors

Compute physicochemical descriptors for every molecule in a MolTable.

??? note "Details"
    Toggle common descriptors via checkboxes.  For any RDKit descriptor not
    listed, type comma-separated names in the *Custom* field (e.g.
    ``BalabanJ, FractionCSP3, ExactMolWt``).
    
    Uses ThreadPoolExecutor for parallelism (RDKit releases the GIL).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

**Properties:** ``

---

### Batch File Writer

Write all molecules in a MolTable to disk.

??? note "Details"
    Can write a single multi-record file (SDF, SMI) or individual files
    per molecule (MOL2, SDF, PDB, XYZ) into a chosen directory.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |

**Properties:** `Mode`, `Format`

---

### Batch Mol Drawer

Render every molecule in a MolTable to a PNG or SVG file on disk.

??? note "Details"
    Writes one image per row to **Output Folder**, named from the chosen
    **Filename Column** (sanitised).  Adds an ``image_path`` column to the
    output MolTable so downstream nodes can find the files.
    
    Optional features:

    - **Legend Column**: caption rendered under each structure.
    - **Highlight SMARTS**: matched atoms drawn in red.
    - **Dark Mode**: dark background (matches the single-mol viewer node).
    
    Filename collisions get a ``_1``, ``_2`` suffix.  Empty / missing
    filenames fall back to ``mol_<row_index>``.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

**Properties:** `Format`, `Size (px)`, ``

---

### Batch Substructure Filter

Filter a MolTable by SMARTS substructure pattern.

??? note "Details"
    Splits into two outputs: matches (has substructure) and rejects.
    Operates directly on the Mol objects -- no SMILES re-parsing.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `matches` | mol_table |
| **Output** | `rejects` | mol_table |

---

### Butina Cluster

Cluster molecules using Taylor-Butina algorithm.

??? note "Details"
    Computes fingerprints, pairwise similarity (Rust), then clusters.
    Adds ``cluster_id`` and ``is_centroid`` columns to the output.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

**Properties:** `Metric`, `Cluster Method`, `Similarity Threshold`

---

### Drug-likeness Filter

Apply a classic drug-likeness rule set to a MolTable.

??? note "Details"
    All rules within the chosen preset are AND'd.
    Outputs *matches* (pass all rules) and *rejects* (fail at least one).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `matches` | mol_table |
| **Output** | `rejects` | mol_table |

**Properties:** `Preset`

---

### Fingerprint

Add a fingerprint column to a MolTable, one numpy array per row.

??? note "Details"
    The new column's dtype follows the chosen FP method (``bool`` for
    bit-vector FPs, ``float64`` for ErG, ``uint32`` for MHFP).  Downstream
    ML nodes can stack the column into an ``(N, D)`` feature matrix with
    ``np.stack(df['fp'].tolist())``.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

---

### Largest Fragment

Replace each molecule in a MolTable with its largest connected fragment.

??? note "Details"
    Useful upstream of docking: salts (``CC(=O)O.[Na]``), counter-ions,
    and solvents in disconnected SMILES break PDBQT round-trips and
    embedding.  This node uses ``rdMolStandardize.LargestFragmentChooser``,
    which picks the largest fragment by heavy-atom count and breaks ties by
    molecular weight.
    
    The ``smiles`` column is also rewritten to the canonical SMILES of the
    chosen fragment so downstream nodes see a consistent identifier.
    
    Single-fragment molecules pass through untouched.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

---

### MolTable Merge

Combine two MolTables with AND or OR logic.

??? note "Details"
    - **AND** -- keep only molecules whose *name* appears in **both** inputs
    (intersection).  Rows are taken from input A.
    
    - **OR** -- keep molecules from **either** input (union, duplicates by
    name removed, first occurrence kept).
    
    Pair with PropertyFilterNode / DrugLikenessFilterNode to build
    complex filter chains.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table_a` | mol_table |
| **Input** | `mol_table_b` | mol_table |
| **Output** | `mol_table` | mol_table |

**Properties:** `Logic`

---

### Murcko Scaffold

Add a Murcko scaffold SMILES column to every row of a MolTable.

??? note "Details"
    The Murcko scaffold of a molecule is its ring system together with the
    linker atoms that connect those rings -- substituents are stripped.
    Useful for scaffold-based diversity analysis or for non-leaky
    scaffold-aware train/test splits.
    
    The optional **Generic** flag additionally strips atom identities
    (every atom becomes carbon, every bond becomes single) -- coarser
    grouping that captures topology only.
    
    Output: input MolTable + a new ``scaffold`` column (SMILES string).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

**Properties:** ``

---

### Pairwise Similarity

Compute an NxN pairwise similarity matrix for all molecules in a MolTable.

??? note "Details"
    Fingerprints are computed with RDKit; the NxN pairwise calculation runs in
    Rust (sdfrust) with rayon parallelism and hardware popcount.
    
    Output is a Table whose first column is the molecule name and remaining
    columns are named after each molecule (suitable for Heatmap).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `table` | table |

**Properties:** `Metric`

---

### Property Filter

Filter a MolTable by a single molecular property.

??? note "Details"
    Pick a common property from the dropdown **or** select *Custom* and type
    any RDKit descriptor name (e.g. ``BalabanJ``, ``ExactMolWt``).
    
    Choose a comparison operator (<, >, ≤, ≥, =) and a threshold value.
    
    Combine multiple PropertyFilterNodes with a MolTable Merge node to build
    complex AND / OR filter chains.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `matches` | mol_table |
| **Output** | `rejects` | mol_table |

**Properties:** `Property`, `Operator`, `Value`

---

### Remove Hydrogens

Remove explicit hydrogen atoms from every molecule in a MolTable.

??? note "Details"
    Calls ``Chem.RemoveHs(mol)``.  Useful for slimming representations before
    fingerprinting / SMILES export, or for cleanup after docking.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

---

### Sanitize Stereo

Re-assign stereochemistry on every molecule in a MolTable.

??? note "Details"
    Calls ``Chem.AssignStereochemistry(mol, cleanIt=True, force=True)`` on
    each Mol, which strips inconsistent stereo flags (E/Z markers without
    valid geometry, etc.).  Useful before SECFP / MHFP fingerprinting on
    modern RDKit, which asserts on bad bond stereo in ``Canon.cpp``.
    
    Input mols are cloned so upstream nodes are not mutated.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

**Properties:** ``

---

### Similarity Search

Rank all molecules in a MolTable by similarity to a query molecule.

??? note "Details"
    Adds a ``similarity`` column and sorts descending.  Optionally filters
    by a minimum similarity threshold.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `molecule` | molecule |
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

**Properties:** `Metric`, `Min Similarity`

---
