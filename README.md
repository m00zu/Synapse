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

Connect processing steps on a canvas to build full analysis pipelines, from loading raw data to generating figures. No code, no app-switching, no reformatting files between steps.

## What it does

- **Visual pipeline builder**: connect nodes on a canvas to build analysis workflows
- **Reproducible & shareable**: workflows save as `.json` files that anyone can open and run
- **Batch processing**: iterate over entire folders with automatic result accumulation
- **Plugin system**: extend with custom nodes distributed as `.py`, `.zip`, or `.synpkg` packages
- **AI workflow assistant** *(beta)*: describe what you want and the AI builds the node graph (Ollama, OpenAI, Claude, Gemini, Groq)
- **Cross-platform**: macOS, Windows, and Linux

## Installation

Tested on Python 3.13 and 3.14.

```bash
git clone https://github.com/m00zu/Synapse
cd Synapse
pip install .
```

Optional but recommended: Install pre-built Rust extensions for faster OIR file reading and image processing:

```bash
pip install oir_reader_rs image_process_rs --find-links https://github.com/m00zu/Synapse/releases/expanded_assets/rust-v0.1.1
```

Then run:

```bash
synapse
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

1. Download `.synpkg` files from [Synapse-Plugins Releases](https://github.com/m00zu/Synapse-Plugins/releases)
2. In Synapse, go to **Plugins > Install Plugin** and select the `.synpkg` file
3. Click **Plugins > Reload Plugins** and the new nodes appear in the Node Explorer

You can also drop `.py` files or extracted plugin folders directly into the `plugins/` directory.

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

## Documentation

Available at [m00zu.github.io/Synapse](https://m00zu.github.io/Synapse/) and built into the app via **Help > Open Manual**.

## License

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can use, share, and adapt it for non-commercial purposes with attribution.
