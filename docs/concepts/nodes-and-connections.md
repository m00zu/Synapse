# Nodes & Connections

## What is a Node?

A node is a single processing step in your pipeline. Each node:

- Has **input ports** (left side) that receive data
- Has **output ports** (right side) that produce results
- Has **properties** (shown in the Properties panel) that control its behavior

## Data Flow

Data flows **left to right** through connections. When you click **Run Graph**, Synapse:

1. Sorts all nodes in topological order (upstream first)
2. Executes each node in order
3. Passes outputs to downstream inputs through connections

Only **dirty** nodes (whose inputs have changed) are re-executed. If you change a parameter on one node, only that node and its downstream dependents will re-run.

## Port Types

Ports are typed — you can only connect ports of compatible types:

| Type | Color | Description | Typical Nodes |
|------|-------|-------------|---------------|
| **ImageData** | Green | A 2D image (grayscale, RGB, or RGBA) | Image Reader, filters, transforms |
| **MaskData** | Forest green | A binary mask (black & white) | Threshold, ROI, morphology |
| **LabelData** | Chartreuse | An integer label image | Watershed, connected components |
| **SkeletonData** | Yellow-green | A thinned skeleton mask | Skeletonize |
| **TableData** | Blue | A data table (pandas DataFrame) | Particle Props, statistics |
| **StatData** | Royal blue | Statistical test results | Grouped Comparison, ANOVA |
| **FigureData** | Purple | A matplotlib figure | Plot nodes |
| **PathData** | Gray | A file or folder path | Folder Iterator, Image Reader |

## Connection Rules

- One output can connect to **multiple** inputs (fan-out)
- One input can only receive **one** connection
- Connections must match compatible data types
- Circular connections (loops) are not allowed

## Node States

During execution, nodes show their state:

| State | Meaning |
|-------|---------|
| **Clean** | Successfully executed, output is cached |
| **Dirty** | Needs re-execution (input or property changed) |
| **Error** | Execution failed (check error message) |
| **Disabled** | Skipped during execution |
