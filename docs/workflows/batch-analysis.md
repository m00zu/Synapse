# Workflow: Batch Image Analysis

Process an entire folder of images and collect results into a single table.

## Pipeline Overview

```
Folder Iterator → Image Reader → [Processing] → Particle Props → Batch Accumulator → Data Table
```

## Step-by-Step

### 1. Set Up the Iterator

1. Add a **Folder Iterator** node
2. Select your image folder
3. Set the file pattern (e.g., `*.tif`)

### 2. Build the Processing Pipeline

Connect your standard single-image pipeline between Image Reader and the measurement node.

### 3. Add a Batch Accumulator

Place a **Batch Accumulator** between Particle Props (or any table output) and your downstream analysis nodes.

### 4. Run the Batch

Press ++ctrl+b++ (Batch Run). The status bar shows progress.

### 5. Analyze Combined Results

After the batch completes, the Batch Accumulator outputs a combined table with all results. Each row includes the source filename.

Connect downstream nodes for:

- **Sort Table** / **Filter Table** — organize results
- **Grouped Comparison** — compare between experimental groups
- **Plot nodes** — visualize distributions across samples

<!-- TODO: Add screenshots -->
