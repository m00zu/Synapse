# Interface Overview

## Main Window Layout

The Synapse window is divided into several areas:

### Canvas (Center)

The main workspace where you create and connect nodes.

- **Pan:** Middle-click drag, or hold ++alt++ and drag
- **Zoom:** Scroll wheel
- **Select:** Click a node, or drag a selection box
- **Multi-select:** Hold ++shift++ and click
- **Delete:** Select node(s) and press ++delete++

### Node Explorer (Left Panel)

A tree of all available nodes organized by category. Double-click or drag a node onto the canvas to add it.

Press ++ctrl+f++ to focus the search bar and filter nodes by name.

You can also press ++tab++ on the canvas to open an inline quick-search popup.

### Properties Panel (Right Panel)

Shows the properties of the currently selected node. Adjust parameters like thresholds, radii, file paths, etc.

### Toolbar (Top)

| Button | Shortcut | Action |
|--------|----------|--------|
| Run Graph | ++ctrl+w++ | Execute the entire pipeline once |
| Batch Run | ++ctrl+b++ | Process all files in a Folder Iterator |
| Stop | — | Cancel a running execution |
| Clear Selected Caches | — | Free memory for selected nodes |
| Clear All Caches | — | Free memory for all nodes |
| Light/Dark Mode | — | Toggle between light and dark theme |

Save and Open are in the **Workflows** menu.

### Node Help Panel (Right Panel, Tab)

Shows documentation, ports, and properties for the currently selected node. Toggle via **Help > Node Help Panel**.

### LLM Assistant (Right Panel, Tab)

An AI assistant that can help you build workflows. Hidden by default — enable via **View > LLM Assistant**. Configure your API key in the assistant panel settings.

### Status Bar (Bottom)

Shows execution progress, batch status, and messages.

## Working with Nodes

### Adding Nodes

- **Search bar:** Press ++ctrl+f++ → type node name → drag and drop onto the canvas to add
- **Node explorer:** Drag from the left panel onto the canvas

### Connecting Nodes

1. Click on an **output port** (right side of a node)
2. Drag to an **input port** (left side of another node)
3. Release to create the connection

Ports are color-coded by data type:

| Color | Data Type |
|-------|-----------|
| Green | Image |
| Forest green | Mask |
| Blue | Table |
| Royal blue | Stat |
| Purple | Figure |
| Chartreuse | Label |
| Yellow-green | Skeleton |
| Gray | Path / Other |


### Disabling Nodes

Right-click a node → **Disable** to skip it during execution without removing it from the graph.

<!-- TODO: Add annotated screenshot of the interface -->
