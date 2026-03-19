# Quick Start

Build your first image analysis pipeline in 5 minutes.

## Step 1: Add an Image Reader

1. Press ++ctrl+f++ to open the Node Explorer search, or right-click on the canvas
2. Search for **"Image Reader"**
3. Double-click or drag to place the node on the canvas
4. In the node's properties, click **Browse** to select an image file

## Step 2: Apply a Filter

1. Add a **Gaussian Blur** node (++ctrl+f++ → search "Gaussian")
2. Connect the `image` output of Image Reader to the `image` input of Gaussian Blur
3. Adjust the **Sigma** property to control blur strength

!!! tip "Making connections"
    Click and drag from an output port (right side) to an input port (left side). Ports are color-coded by data type.

## Step 3: Threshold the Image

1. Add a **Binary Threshold** node
2. Connect the Gaussian Blur output to it
3. Adjust the threshold value

## Step 4: Measure Objects

1. Add a **Particle Props** node
2. Connect the Binary Threshold mask output to the `mask` input
3. Optionally connect the original image to the `image` input (for intensity measurements)

## Step 5: Run the Pipeline

Click **Run Graph** in the toolbar or press ++ctrl+w++.

The pipeline executes node by node. Results appear:

- **Table output** — connect to a **Data Table Node** to view inline
- **Figure output** — connect to a **Data Figure Node** to view plots

## Step 6: Save Your Workflow

Use ++ctrl+s++ to save the workflow as a `.json` file. Reopen it later to continue where you left off.

## Next Steps

- [Interface Overview](interface.md) — learn the UI layout
- [Batch Processing](../concepts/batch-processing.md) — process folders of images
- [Node Reference](../plugins/image-analysis/filters.md) — explore all available nodes

<!-- TODO: Add screenshots for each step -->
