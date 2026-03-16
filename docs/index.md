# Synapse Manual

Welcome to the **Synapse** documentation — a visual, node-based application for scientific data analysis, image processing, and research workflows.

## What is Synapse?

Synapse lets you build analysis pipelines by connecting visual nodes on a canvas. Each node performs a specific operation — reading data, applying a filter, running statistics, generating a plot — and data flows through connections from one node to the next.

No coding required for standard workflows. For advanced users, a Python scripting node and AI assistant are also available.

## Key Features

- **35+ built-in nodes** for data I/O, table operations, display, and batch processing
- **150+ plugin nodes** for image analysis, segmentation, statistics, plotting, cheminformatics, and more
- **Batch processing** — process hundreds of files with a single click
- **Interactive ROI** — draw regions of interest directly on images
- **AI assistant** — describe a workflow in natural language and let the AI build it (supports Ollama, OpenAI, Claude, Gemini, Groq)
- **Plugin system** — extend with custom nodes (`.py`, `.zip`, `.synpkg`)
- **Multilingual** — English and Traditional Chinese (繁體中文)

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
| SAM2 Segmentation | [SAM2](plugins/sam2.md) |
| Cellpose | [Cellpose](plugins/cellpose.md) |
| Video & Tracking | [Tracking](plugins/video-tracking.md) |
| 3D Volume | [Volume](plugins/volume.md) |
| Cheminformatics | [Chemistry](plugins/cheminformatics.md) |
| Filopodia | [Filopodia](plugins/filopodia.md) |
