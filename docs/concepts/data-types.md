# Data Types

Synapse uses a set of typed data containers that flow between nodes. Each type has a distinct port color.

## ImageData (Green)

A 2D image represented as a PIL Image.

- **Formats:** Grayscale (L), RGB, RGBA
- **Bit depth:** 8-bit (uint8) or 16-bit (uint16)
- **File types:** TIFF, PNG, JPEG, BMP, and more
- **Conversion:** Use "RGB to Gray" to convert color to grayscale

## MaskData (Forest Green)

A binary mask — a single-channel image where white (255) represents foreground and black (0) represents background.

- Created by threshold nodes, ROI nodes, or segmentation tools
- Used as input for morphological operations and measurements
- Multiple masks can be combined with Image Math

## LabelData (Chartreuse)

An integer-valued image where each connected region has a unique label (1, 2, 3, ...).

- Created by Watershed, Label 2D, or segmentation nodes
- Background pixels have label 0
- Displayed as a color-coded overlay
- Used for per-object measurements

## SkeletonData (Yellow-Green)

A thinned skeleton mask — a special subtype of MaskData produced by skeletonization.

- Created by Skeletonize node
- Used for branch analysis, filopodia measurement, and path tracing

## TableData (Blue)

A tabular dataset (pandas DataFrame) with named columns and rows.

- Created by measurement nodes (Particle Props, Image Stats, etc.)
- Manipulated with Data & Table nodes (Filter, Sort, Aggregate, etc.)
- Visualized with Plot nodes
- Saved to CSV/Excel with the Data Saver node

## StatData (Royal Blue)

A statistical result object containing test statistics, p-values, and summary tables.

- Created by analysis nodes (Grouped Comparison, Pairwise Comparison, etc.)
- Displayed in the node's output or connected to a Data Table Node

## FigureData (Purple)

A matplotlib figure object for scientific plots.

- Created by Plot nodes (Scatter, Histogram, Box Plot, etc.)
- Edited with the Figure Editor node
- Displayed inline with Data Figure Node
- Exported as SVG/PNG/PDF via the SVG Editor node

## PathData (Gray)

A file or folder path string.

- Produced by Folder Iterator (one path per batch iteration)
- Consumed by Image Reader, Table Reader, etc.
- Modified with Path Modifier node
