# Protein

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
