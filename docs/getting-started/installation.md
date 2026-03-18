# Installation

## Requirements

- **Python 3.14+**
- **Operating System:** macOS, Windows, or Linux

## Install from Source

```bash
git clone https://github.com/m00zu/Synapse
cd Synapse
pip install .
```

Launch:

```bash
synapse
```

Or with a virtual environment (recommended):

```bash
conda create -n synapse python=3.14
conda activate synapse
pip install .
synapse
```

## Optional: Rust-accelerated Modules

Synapse works fully with pure Python, but optional Rust extensions provide significant speedups for OIR microscopy file reading and image processing:

```bash
# Install Rust toolchain (one time)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build OIR reader
cd oir_reader_rs && maturin develop --release && cd ..

# Build image processing
cd image_process_rs && maturin develop --release && cd ..
```

## Installing Plugins

Synapse ships with core data processing nodes. Image analysis, statistics, plotting, and other domain-specific nodes are available as **plugins**.

### Method 1: In-app (recommended)

1. Go to **Plugins > Install Plugin** in the menu bar
2. Select a `.py`, `.zip`, or `.synpkg` file
3. Click **Plugins > Reload Plugins** — new nodes appear in the Node Explorer

### Method 2: Manual

Drop plugin files directly into the `plugins/` directory:

| Platform | Plugin directory |
|----------|-----------------|
| macOS | `~/Library/Application Support/Synapse/plugins/` |
| Windows | `%APPDATA%\Synapse\plugins\` |
| Linux | `~/.config/Synapse/plugins/` |
| From source | `./plugins/` (next to `synapse/`) |

### Available Plugins

| Plugin | Description |
|--------|-------------|
| **Image Analysis** | Filters, thresholding, morphology, segmentation, measurements, ROI |
| **Statistical Analysis** | t-tests, ANOVA, regression, survival analysis, PCA |
| **Figure Plotting** | Scatter, box, violin, heatmap, volcano, regression plots |
| **SAM2 Segmentation** | Interactive click-to-segment, multi-object, auto-segment |
| **Cellpose** | Cell/nucleus segmentation (ONNX, no PyTorch) |
| **Video & Tracking** | Frame extraction, object tracking, trajectory analysis |
| **3D Volume** | Volume rendering and analysis |
| **Cheminformatics** | Molecule editing, docking, protein prep |
| **Filopodia** | Cell protrusion detection and measurement |

All plugin dependencies are bundled or auto-downloaded on first use. No manual `pip install` required.

## Verifying Installation

After launching, you should see:

- The **node canvas** (center) — where you build pipelines
- The **Node Explorer** (left) — browse and search for nodes
- The **Properties panel** (right) — configure the selected node

Press ++ctrl+f++ to search for nodes, or right-click on the canvas to browse the node menu.
