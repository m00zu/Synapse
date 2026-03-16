# Workflow: Cell Counting

This workflow demonstrates how to count cells in a fluorescence microscopy image.

## Pipeline Overview

```
Image Reader → Gaussian Blur → Binary Threshold → Remove Small Obj → Watershed → Data Table
```

## Step-by-Step

### 1. Load the Image

Add an **Image Reader** node and select your microscopy image.

### 2. Pre-process

- **Gaussian Blur** (sigma = 1-2): Reduces noise
- **Equalize Adapthist** (optional): Enhances contrast for uneven illumination

### 3. Segment

- **Binary Threshold**: Separate cells from background. Adjust the threshold until cells are cleanly separated.
- **Remove Small Obj**: Eliminate noise artifacts (set minimum size in pixels)

### 4. Split Touching Cells

- **Watershed**: Separates touching/overlapping cells using distance-based markers
- Set **Min Object Sep.** to the approximate minimum distance between cell centers

### 5. Measure

The Watershed node outputs both a label image and a table with per-object measurements:

- Area, perimeter, circularity
- Centroid coordinates
- Eccentricity, orientation
- Intensity statistics (if image is connected)

### 6. Visualize

Connect the table to a **Data Table Node** for inline viewing, or to plot nodes for visualization.

<!-- TODO: Add screenshots and example images -->
