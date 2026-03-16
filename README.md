<p align="center">
  <img src="synapse/icons/synapse_icon.png" alt="Synapse" width="128">
</p>

<h1 align="center">Synapse</h1>

<p align="center">
  A visual node-graph workflow editor for scientific data analysis.
</p>

<p align="center">
  <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg" alt="Platform">
</p>

---

One tool for the whole workflow, from raw data to figures, instead of switching between separate apps for each step. Drag nodes onto a canvas, connect them, and data flows through the pipeline.

## What it does

- **Visual pipeline builder** — connect nodes on a canvas to build analysis workflows
- **Reproducible & shareable** — workflows save as `.json` files that anyone can open and run
- **Batch processing** — iterate over entire folders with automatic result accumulation
- **Plugin system** — extend with custom nodes distributed as `.py`, `.zip`, or `.synpkg` packages
- **Cross-platform** — macOS, Windows, and Linux

## Installation

Tested on Python 3.13 and 3.14.

```bash
git clone https://github.com/Ezra-Nemo/Synapse
cd Synapse
pip install .
```

Then run:

```bash
synapse
```

Or:

```bash
python -m synapse
```

### Rust modules (optional)

Everything works without Rust. But if you want faster OIR file reading and image processing, you can build the optional Rust extensions:

```bash
# Install Rust (one time)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin (Rust-Python build tool)
pip install maturin

# Build each module
cd oir_reader_rs && maturin develop --release && cd ..
cd image_process_rs && maturin develop --release && cd ..
```

## Example workflows

### CSV analysis

`Table Reader` > `Filter Table` > `Single Table Math` > `Aggregate Table` > `Data Table Node`

Load a CSV of cell measurements, filter out debris (`area > 100`), compute circularity (`4 * pi * area / perimeter^2`), aggregate by group to get mean values for Control vs Treatment, and display the summary.

<p align="center">
  <img src="docs/images/Example_1.png" alt="CSV Analysis Pipeline" width="800">
</p>

### Object detection and measurement

`Image Reader` > `Gaussian Blur` > `Binary Threshold` > `Fill Holes` > `Watershed` > `Data Table Node`

Load a coin image, blur to reduce noise, threshold, fill holes, then watershed to separate touching objects. Outputs area, perimeter, and circularity for each detected object.

<p align="center">
  <img src="docs/images/Example_2.png" alt="Image Object Detection" width="800">
</p>

### Statistical comparison

`Table Reader` > `Filter Table` > `Pairwise Comparison` > `Bar Plot` > `Data Figure Node`

Load cell measurement data, filter out debris, run a pairwise comparison on `intensity_mean` between Control and Treatment, and plot the result with significance annotations.

<p align="center">
  <img src="docs/images/Example_3.png" alt="Statistical Comparison" width="800">
</p>

### Batch OIR conversion

```
Folder Iterator --> Image Reader  --> Data Saver
       └---------> Path Modifier --↗
```

Batch-convert Olympus OIR microscopy files to TIFF. The iterator feeds each `.oir` path to both the reader (decodes the image) and the path modifier (swaps the extension to `.tif` and redirects to an output folder). Both connect to the saver.

<p align="center">
  <img src="docs/images/Example_4.png" alt="Batch OIR Conversion" width="800">
</p>

### Collagen area measurement (video)

https://github.com/user-attachments/assets/a3772ee9-da64-4fe1-ad58-ee22ac6f41aa

<p align="center"><i>Color deconvolution of a Masson's trichrome stain, threshold the collagen channel, measure area.</i></p>

## Plugins

The core handles data I/O, table operations, and display. Domain-specific nodes ship as plugins with their dependencies bundled in.

### Installing plugins

1. Go to **Plugins > Install Plugin** and select a `.py`, `.zip`, or `.synpkg` file
2. Or drop plugin files into the `plugins/` directory manually
3. Click **Plugins > Reload Plugins** and the new nodes appear in the Node Explorer

### Available plugins

| Plugin | Nodes | Description |
|--------|-------|-------------|
| Image Analysis | ~70 | Filters, thresholding, morphology, segmentation, measurements, ROI |
| Statistical Analysis | 13 | t-tests, ANOVA, regression, survival analysis, PCA |
| Figure Plotting | 16 | Scatter, box, violin, heatmap, volcano, regression plots |
| SAM2 Segmentation | 4 | Interactive click-to-segment with Meta SAM2 (ONNX, no PyTorch) |
| Cellpose | 2 | Deep learning cell/nucleus segmentation (ONNX) |
| Video & Tracking | 8 | Frame extraction, multi-object tracking, trajectory analysis |
| Cheminformatics | 20+ | RDKit molecule editing, docking, protein prep (fully vendored) |
| 3D Volume | 5 | Volume rendering and analysis |
| Filopodia | 3 | Cell protrusion detection and measurement |

## Other features

- **AI workflow assistant** *(beta)* -- describe what you want and the AI builds the node graph. Supports Ollama, OpenAI, Claude, Gemini, and Groq.
- **Multilingual** -- English and Traditional Chinese.
- **Dynamic column selectors** -- table nodes detect input columns and offer dropdown menus instead of requiring you to type column names.

## Project structure

```
synapse/                Core application package
  app.py                Main window and execution engine
  nodes/                Built-in node definitions (I/O, tables, display, utility)
  llm_assistant.py      AI workflow generation (multi-provider)
  plugin_loader.py      Dynamic plugin discovery and loading
  package_plugin.py     CLI tool for packaging plugins as .synpkg
  data_models.py        Shared data types (TableData, ImageData, etc.)
  i18n.py               Internationalization system
  icons/                Application icons
  translations/         Language files
  ui/                   SVG editor components
NodeGraphQt/            Node graph framework (modified fork)
docs/                   User documentation (mkdocs)
workflows/              Example workflow files
```

## License

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can use, share, and adapt it for non-commercial purposes with attribution.
