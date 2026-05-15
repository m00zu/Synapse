# Synapse Manual

Documentation for Synapse, a node-based app for building scientific data analysis workflows.

## What is Synapse?

Synapse lets you build analysis pipelines by connecting nodes on a canvas. Each node handles one step, whether that's reading data, filtering rows, running statistics, or plotting results. Connect them and watch the data flow through!

## What you can do

- Build pipelines visually by connecting nodes on a canvas
- Process entire folders of files in batch
- Save and share workflows as `.json` files; re-run them deterministically
- Strict port-type checking at connection time (Rust-style, with Liskov subtype polymorphism)
- Draw ROIs directly on images for targeted analysis
- Drive Synapse from your favourite chat client via **MCP** (Claude Code / Desktop, Antigravity, Gemini CLI), or use the in-app AI chat panel with 8 providers (including local Ollama / llama.cpp)
- Auto-organize the canvas left-to-right with `Ctrl+L`; run a subset of the graph with `Ctrl+Shift+W`
- Add new nodes through plugins (`.py`, `.zip`, `.synpkg`)

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Installation](getting-started/installation.md) | Install Synapse and get running |
| [Quick Start](getting-started/quick-start.md) | Build your first pipeline in 5 minutes |
| [Interface Overview](getting-started/interface.md) | Tour of the main window |
| [Data Types](concepts/data-types.md) | Understanding data flowing between nodes |
| [Batch Processing](concepts/batch-processing.md) | Process folders of images or files |
| [AI in Synapse](ai/index.md) | In-app AI Chat + MCP server overview |
| [MCP Setup](mcp/README.md) | Connect external chat clients |
| [Installing Plugins](getting-started/plugins.md) | Add image analysis, statistics, and more |
| [Creating Plugins](developing/creating-plugins.md) | Build your own nodes |
| [Keyboard Shortcuts](shortcuts.md) | Hotkeys and shortcuts |

## Node Reference

### Core (always available)

| Category | Pages |
|----------|-------|
| Core | [I/O](nodes/io.md) · [Display](nodes/display.md) · [Utility](nodes/utility.md) · [Data](nodes/data.md) · [Collection](nodes/collection.md) |
| Statistical Analysis | [Descriptive & Comparison](nodes/analysis.md) |
| Plotting | [Plotting](nodes/plotting.md) |
| DataFrame | [I/O](nodes/dataframe/io.md) · [Filter](nodes/dataframe/filter.md) · [Compute](nodes/dataframe/compute.md) · [Transform](nodes/dataframe/transform.md) · [Combine](nodes/dataframe/combine.md) · [Utility](nodes/dataframe/util.md) |
| Image Processing | [I/O](nodes/image_process/io.md) · [Color](nodes/image_process/color.md) · [Exposure](nodes/image_process/exposure.md) · [Filter](nodes/image_process/filter.md) · [Transform](nodes/image_process/transform.md) · [Morphology](nodes/image_process/morphology.md) · [Measure](nodes/image_process/measure.md) · [Visualize](nodes/image_process/visualize.md) |
| 3D Volume | [I/O](nodes/volume/io.md) · [Color](nodes/volume/color.md) · [Exposure](nodes/volume/exposure.md) · [Filters](nodes/volume/filters.md) · [Morphology](nodes/volume/morphology.md) · [Display](nodes/volume/display.md) |
| Cheminformatics | [I/O](nodes/cheminformatics/io.md) · [Molecule](nodes/cheminformatics/mol.md) · [Batch](nodes/cheminformatics/batch.md) · [Convert](nodes/cheminformatics/convert.md) · [Protein](nodes/cheminformatics/protein.md) · [Docking](nodes/cheminformatics/docking.md) · [Viewer](nodes/cheminformatics/viewer.md) |

### Plugins (installed separately)

| Plugin | Pages |
|--------|-------|
| Machine Learning | [Classification](plugins/ml/classification.md) · [Regression](plugins/ml/regression.md) · [Clustering](plugins/ml/clustering.md) · [Embedding](plugins/ml/embedding.md) · [Preprocessing](plugins/ml/preprocessing.md) · [Evaluation](plugins/ml/evaluation.md) · [Visualization](plugins/ml/visualization.md) · [Model I/O](plugins/ml/io.md) |
| Segmentation (SAM2 / Cellpose / Grounding) | [Segmentation](plugins/segmentation.md) |
| Video Analysis & Tracking | [Video Analysis](plugins/video_analysis.md) |
| Filopodia Analysis | [Filopodia](plugins/filopodia.md) |
| Report Generation | [Report](plugins/report.md) |
