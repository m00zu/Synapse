# Synapse Manual

Documentation for Synapse, a node-based app for building scientific data analysis workflows.

## What is Synapse?

Synapse lets you build analysis pipelines by connecting nodes on a canvas. Each node handles one step, whether that's reading data, filtering rows, running statistics, or plotting results. Connect them and watch the data flows through!

## What you can do

- Build pipelines visually by connecting nodes on a canvas
- Process entire folders of files in batch
- Save and share workflows as `.json` files
- Draw ROIs directly on images for targeted analysis
- Have an AI build node graphs from text descriptions *(beta)*
- Add new nodes through plugins (`.py`, `.zip`, `.synpkg`)

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Installation](getting-started/installation.md) | Install Synapse and get running |
| [Quick Start](getting-started/quick-start.md) | Build your first pipeline in 5 minutes |
| [Interface Overview](getting-started/interface.md) | Tour of the main window |
| [Data Types](concepts/data-types.md) | Understanding data flowing between nodes |
| [Batch Processing](concepts/batch-processing.md) | Process folders of images or files |
| [Installing Plugins](getting-started/plugins.md) | Add image analysis, statistics, and more |
| [Creating Plugins](developing/creating-plugins.md) | Build your own nodes |
| [Keyboard Shortcuts](shortcuts.md) | Hotkeys and shortcuts |

## Node Reference

### Built-in (Core)

| Category | Docs |
|----------|------|
| I/O & Display | [IO & Display](nodes/io-display.md) |
| Data & Tables | [Filter](nodes/data-tables/filter.md) · [Compute](nodes/data-tables/compute.md) · [Transform](nodes/data-tables/transform.md) · [Combine](nodes/data-tables/combine.md) · [Utility](nodes/data-tables/util.md) |

### Plugins (install separately)

| Plugin | Docs |
|--------|------|
| Image Analysis | [Color](plugins/image-analysis/color.md) · [Exposure](plugins/image-analysis/exposure.md) · [Filters](plugins/image-analysis/filters.md) · [Transform](plugins/image-analysis/transform.md) · [Morphology](plugins/image-analysis/morphology.md) · [Measurement](plugins/image-analysis/measurement.md) · [ROI & Drawing](plugins/image-analysis/roi-drawing.md) |
| Statistical Analysis | [Descriptive](plugins/statistical-analysis/descriptive.md) · [Regression](plugins/statistical-analysis/regression.md) |
| Figure Plotting | [Plots](plugins/figure-plotting.md) |
| SAM2 & Cellpose | [SAM2](plugins/sam2.md) · [Cellpose](plugins/cellpose.md) · [Video & Tracking](plugins/video-tracking.md) |
| 3D Volume | [Volume](plugins/volume.md) |
| Cheminformatics | [Chemistry](plugins/cheminformatics.md) |
| Filopodia | [Filopodia](plugins/filopodia.md) |
