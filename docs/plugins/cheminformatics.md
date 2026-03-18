# Cheminformatics

### SMILES Input

Parse a SMILES string and output a Molecule object.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `molecule` | molecule |

---

### SMILES Viewer

Display a 2-D molecule structure diagram.

??? note "Details"
    Type a SMILES string (e.g. ``c1ccc2ccccc2c1`` for naphthalene) directly
    into the node. The structure renders in the image viewer panel and is also
    available on the ``image`` output port for downstream processing.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `image` | image |

**Properties:** `Dark Mode`

---

### Molecule to Image

Render a molecule structure as a 2-D diagram image.

??? note "Details"
    Accepts a Molecule object from SMILES Input and outputs an ImageData.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `molecule` | molecule |
| **Output** | `image` | image |

**Properties:** `Dark Mode`, `Size (px)`

---

### Molecular Descriptors

Compute a table of physicochemical descriptors for a molecule.

??? note "Details"
    Outputs a DataFrame with one row containing: smiles, mol_weight, logp,
    hbd, hba, tpsa, rotatable_bonds, rings, aromatic_rings.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `molecule` | molecule |
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

### Mol 3D Embed

Embed a molecule in 3D using ETKDGv3 and optionally optimize.

??? note "Details"
    Generates one or more 3D conformers, optionally runs force-field
    minimisation (MMFF or UFF).  All conformers are kept on the output
    molecule.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `molecule` | molecule |
| **Output** | `molecule` | molecule |

**Properties:** `Keep Hydrogens`, `Optimize`, `Force Field`, `Num Conformers`, `Max Iterations (0=default)`, `Prune RMSD (-1=off)`, `Random Seed (-1=random)`, `Timeout sec (0=none)`, `Random Coords Fallback`

---

### Mol File Writer

Write a molecule to a 3D file format.

??? note "Details"
    Accepts a Molecule with a 3D conformer (e.g. from Mol 3D Embed) and
    writes it to disk. MOL2 uses sdfrust; SDF/PDB/XYZ use RDKit;
    PDBQT uses Meeko (optional).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `molecule` | molecule |

**Properties:** `Format`

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

### Batch Substructure Filter

Filter a MolTable by SMARTS substructure pattern.

??? note "Details"
    Splits into two outputs: matches (has substructure) and rejects.
    Operates directly on the Mol objects — no SMILES re-parsing.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `matches` | mol_table |
| **Output** | `rejects` | mol_table |

---

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

### MolTable Merge

Combine two MolTables with AND or OR logic.

??? note "Details"
    - **AND** — keep only molecules whose *name* appears in **both** inputs
    (intersection).  Rows are taken from input A.
    
    - **OR** — keep molecules from **either** input (union, duplicates by
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

**Properties:** `Fingerprint`, `Metric`, `Bits`, `Radius`

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

**Properties:** `Fingerprint`, `Metric`, `Bits`, `Radius`, `Min Similarity`

---

### Butina Cluster

Cluster molecules using Taylor–Butina algorithm.

??? note "Details"
    Computes fingerprints, pairwise similarity (Rust), then clusters.
    Adds ``cluster_id`` and ``is_centroid`` columns to the output.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |

**Properties:** `Fingerprint`, `Metric`, `Cluster Method`, `Bits`, `Radius`, `Similarity Threshold`

---

### PDB Loader

Load a PDB or CIF file and output cleaned protein data.

??? note "Details"
    Removes non-protein atoms (water, ligands), handles multi-model files.
    Optionally returns HETATM ligand bounding-box info for auto-boxing.

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `protein` | protein |

**Properties:** `Clean`

---

### PDB Downloader

Download a protein structure from RCSB PDB or AlphaFold Database.

??? note "Details"
    Enter a PDB ID (e.g. ``1AKE``) or UniProt ID (for AlphaFold) and the
    structure is fetched, cleaned, and output as ProteinData.  Automatically
    falls back to CIF format when the PDB file is not available.
    
    HETATM ligands are extracted with bounding-box info (useful for
    auto-centering a docking box on a co-crystallised ligand).

| Direction | Port | Type |
|-----------|------|------|
| **Output** | `ligands` | table |

**Properties:** `Database`

---

### Protein Editor

Filter a protein structure by chain and residue range.

??? note "Details"
    Useful for trimming multi-chain complexes, keeping only the chain(s)
    of interest, or restricting to a residue range for focused docking.
    
    Leave a field empty to keep everything (no filter on that axis).

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `protein` | protein |
| **Output** | `protein` | protein |

**Properties:** `Remove Water`, `Remove HETATM`

---

### Protein Prep

Prepare a protein for docking: fix structure, add H, generate PDBQT.

??? note "Details"
    Pipeline: PDBFixer (fix + add H) → protonation checks → PDBQT typing.
    Requires OpenMM for hydrogen addition.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `protein` | protein |
| **Output** | `receptor` | receptor |

**Properties:** `pH`, `Fix Missing Atoms`, `Fill Gaps`

---

### Docking Box

Define the docking search box with an integrated 3D viewer.

??? note "Details"
    Click on the protein structure to set the docking center.  The docking
    box is drawn in the viewer in real-time as you adjust center/size values.
    Flexible residues can be selected by clicking in "Add Flexible" mode.
    
    Accepts either raw ProteinData or prepared ReceptorData (PDBQT) for
    display.  The receptor is passed through for downstream docking nodes.
    
    Modes:

    - Manual        — enter center/size directly in spinboxes
    - Auto from Ligand — compute box from a connected molecule's coordinates

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `protein` | protein |
| **Input** | `receptor` | receptor |
| **Input** | `molecule` | molecule |
| **Output** | `receptor` | receptor |
| **Output** | `box_config` | box_config |

**Properties:** `Mode`, `Click Mode`, `Padding`

---

### Vina Dock

Dock a single ligand against a prepared receptor.

??? note "Details"
    Supports Vina CLI and QVina2 (Rust) backends.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `receptor` | receptor |
| **Input** | `molecule` | molecule |
| **Input** | `box_config` | box_config |
| **Output** | `energies` | table |

**Properties:** `Backend`, `Scoring`

---

### Batch Dock

Dock every molecule in a MolTable against a prepared receptor.

??? note "Details"
    Docked poses are converted back to RDKit Mol objects and stored in
    the output mol_table (as conformers on the original molecule).
    A real-time progress table shows each molecule's docking status.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `receptor` | receptor |
| **Input** | `mol_table` | mol_table |
| **Input** | `box_config` | box_config |
| **Output** | `results` | table |

**Properties:** `Backend`, `Scoring`

---

### GNINA Rescore

Rescore docking poses with GNINA CNN models.

??? note "Details"
    Accepts:

    - MoleculeData (single docked molecule from VinaDockNode, conformers = poses)
    - MolTableData (batch results from BatchDockNode, uses dock_mol conformers)
    - DockingResultData (legacy PDBQT poses)
    
    Outputs a scores table and (for batch mode) an updated mol_table with
    CNN scores added as columns.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `result` | molecule |
| **Output** | `scores` | table |

**Properties:** `CNN Ensemble`, `Score Mode`, `Scoring Workers`

---

### Structure Writer

Write a protein (PDB) or prepared receptor (PDBQT) to a file.

??? note "Details"
    Accepts ProteinData or ReceptorData on the *structure* input.
    When connected to a ReceptorData with flexible residues, the flex PDBQT
    is written to a separate ``*_flex.pdbqt`` file alongside the rigid one.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `structure` | receptor |

**Properties:** `Auto Extension`

---

### DrugCLIP Screen

Screen molecules against a protein pocket using DrugCLIP embeddings.

??? note "Details"
    Computes contrastive similarity between a protein binding pocket and
    molecules via the DrugCLIP dual-encoder model (ONNX Runtime).
    
    Inputs:

    - receptor  (ProteinData or ReceptorData)
    - box_config (TableData from DockingBoxNode — defines pocket center)
    - mol_table  (MolTableData — batch of molecules)
    
    Outputs:

    - mol_table  (MolTableData with ``drugclip_score`` column, sorted desc)
    - table      (TableData — summary scores)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `receptor` | receptor |
| **Input** | `box_config` | box_config |
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |
| **Output** | `scores` | scores |

**Properties:** `Box Padding (A)`, `Max Pocket Atoms`, `Conformers (if no 3D)`, `Workers`

---

### 3D Structure Viewer

Interactive 3D molecular structure viewer.

??? note "Details"
    Displays proteins as cartoon and ligands as ball-and-stick using 3Dmol.js.
    Connect any combination of protein, receptor, molecule, or docking result
    inputs to visualize the structure.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `protein` | protein |
| **Input** | `receptor` | receptor |
| **Input** | `molecule` | molecule |
| **Input** | `docking_result` | docking_result |

---
