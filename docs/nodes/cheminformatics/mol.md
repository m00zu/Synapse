# Mol

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
