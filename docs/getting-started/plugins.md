# Installing Plugins

Synapse's core provides data I/O, table operations, and display nodes. Domain-specific functionality — image analysis, statistics, plotting, segmentation — is delivered through **plugins**.

## How to Install

### From the App

1. Go to **Plugins > Install Plugin**
2. Select a plugin file (`.py`, `.zip`, or `.synpkg`)
3. Click **Plugins > Reload Plugins**

New nodes will appear in the Node Explorer tree under their respective categories.

### Manual Installation

Copy plugin files into your platform's plugin directory:

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/Synapse/plugins/` |
| Windows | `%APPDATA%\Synapse\plugins\` |
| Linux | `~/.config/Synapse/plugins/` |
| From source | `./plugins/` (next to `synapse/`) |

### Managing Plugins

- **Plugins > Plugin Manager** lets you enable, disable, or uninstall plugins
- Disabled plugins are not loaded on startup but remain on disk
- Plugin settings persist across updates

## Available Plugins

### Image Analysis
Comprehensive image processing toolkit powered by scikit-image. Includes color conversion, exposure adjustment, filters (Gaussian, median, Frangi, etc.), morphological operations, thresholding, segmentation, distance transforms, particle measurements, ROI drawing, and more.

**~70 nodes** covering the full image analysis pipeline.

### Statistical Analysis
Descriptive statistics, normality tests, group comparisons (t-test, Mann-Whitney, Kruskal-Wallis), two-way ANOVA, linear and nonlinear regression, contingency analysis, survival analysis (Kaplan-Meier, Cox), and PCA.

**13 nodes** for statistical testing and modeling.

### Figure Plotting
Publication-ready figures with matplotlib and seaborn. Scatter, box, violin, swarm, bar, histogram, KDE, XY line, heatmap, volcano, regression, and survival plots. All plots are configurable with palettes, axis labels, and styling options.

**16 nodes** for data visualization.

### SAM2 Segmentation
Interactive segmentation using Meta's Segment Anything Model 2. Click on objects to segment them, manage multiple objects, and run automatic full-image segmentation. Runs entirely on ONNX Runtime — no PyTorch required. Models are auto-downloaded on first use.

### Cellpose
Deep learning cell and nucleus segmentation (cyto, cyto2, cyto3, nuclei models). Supports single image and batch folder processing. Runs on ONNX Runtime with GPU acceleration when available. Models auto-download from HuggingFace.

### Video & Tracking
Video frame extraction (imageio + ffmpeg), multi-object tracking via centroid re-prompting, particle linking, trajectory analysis, MSD computation, and track filtering.

### 3D Volume
Volume rendering and analysis for 3D microscopy data. Requires `pyqtgraph`.

### Cheminformatics
RDKit-based chemistry nodes for molecule editing, similarity search, molecular docking (AutoDock Vina), protein preparation, and structure visualization. All dependencies are vendored — no external installs needed.

### Filopodia Analysis
Specialized nodes for detecting and measuring cell protrusions (filopodia) from skeleton-based analysis.

## Packaging Your Own Plugin

See [Creating Plugins](../developing/creating-plugins.md) for how to build and package custom nodes. Use `package_plugin.py` to create `.synpkg` packages for distribution.
