# Docking

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

### Docking Box

Define the docking search box with an integrated 3D viewer.

??? note "Details"
    Click on the protein structure to set the docking center.  The docking
    box is drawn in the viewer in real-time as you adjust center/size values.
    Flexible residues can be selected by clicking in "Add Flexible" mode.
    
    Accepts either raw ProteinData or prepared ReceptorData (PDBQT) for
    display.  The receptor is passed through for downstream docking nodes.
    
    Modes:

    - Manual        -- enter center/size directly in spinboxes
    - Auto from Ligand -- compute box from a connected molecule's coordinates

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `protein` | protein |
| **Input** | `receptor` | receptor |
| **Input** | `molecule` | molecule |
| **Output** | `receptor` | receptor |
| **Output** | `box_config` | box_config |

**Properties:** `Mode`, `Click Mode`, `Padding`

---

### DrugCLIP Screen

Screen molecules against a protein pocket using DrugCLIP embeddings.

??? note "Details"
    Computes contrastive similarity between a protein binding pocket and
    molecules via the DrugCLIP dual-encoder model (ONNX Runtime).
    
    Inputs:

    - receptor  (ProteinData or ReceptorData)
    - box_config (TableData from DockingBoxNode -- defines pocket center)
    - mol_table  (MolTableData -- batch of molecules)
    
    Outputs:

    - mol_table  (MolTableData with ``drugclip_score`` column, sorted desc)
    - table      (TableData -- summary scores)

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `receptor` | receptor |
| **Input** | `box_config` | box_config |
| **Input** | `mol_table` | mol_table |
| **Output** | `mol_table` | mol_table |
| **Output** | `scores` | scores |

**Properties:** `Box Padding (A)`, `Max Pocket Atoms`, `Conformers (if no 3D)`, `Workers`

---

### GNINA Rescore

Rescore docking poses with GNINA CNN models.

??? note "Details"
    Accepts:

    - MoleculeData (single docked molecule from VinaDockNode, conformers = poses)
    - MolTableData (batch results from BatchDockNode; uses mol_col conformers)
    - DockingResultData (legacy PDBQT poses)
    
    Outputs a scores table and (for batch mode) an updated mol_table with
    CNN scores added as columns.

| Direction | Port | Type |
|-----------|------|------|
| **Input** | `result` | molecule |
| **Output** | `scores` | table |

**Properties:** `CNN Ensemble`, `Score Mode`, `Scoring Workers`

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
