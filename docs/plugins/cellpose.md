# Cellpose Segmentation

Cellpose is a deep learning model for cell and nucleus segmentation. Synapse includes a lightweight ONNX-based Cellpose integration that runs without PyTorch.

## Nodes

### Cellpose Segment

Single-image interactive segmentation.

| Direction | Port |
|-----------|------|
| **Input** | `image` |
| **Output** | `mask` |
| **Output** | `label_image` |
| **Output** | `overlay` |

**Properties:**

- `Model` — nuclei, cyto, cyto2, cyto3
- `Channel` — select which channel contains the signal
- `Diameter` — approximate cell diameter in pixels (0 = auto-detect)

### Cellpose Batch

Batch segmentation of a folder of images.

| Direction | Port |
|-----------|------|
| **Input** | `folder` |
| **Output** | `table` |
| **Output** | `label_images` |

**Properties:** Same as Cellpose Segment, plus folder and pattern selection.

## Model Selection Guide

| Model | Best For |
|-------|----------|
| **nuclei** | Round, well-separated nuclei (DAPI, Hoechst) |
| **cyto** | Cytoplasm with nuclear marker in second channel |
| **cyto2** | Improved cytoplasm model |
| **cyto3** | Latest cytoplasm model, best general-purpose |

## Tips

!!! tip "First-time use"
    Models are automatically downloaded from HuggingFace on first use. An internet connection is required for the initial download only.

!!! tip "Diameter"
    If your cells vary significantly in size, set diameter to 0 for auto-detection. For consistent cell sizes, setting an explicit diameter improves accuracy.
