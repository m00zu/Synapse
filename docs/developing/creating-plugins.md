# Creating Plugins

This guide walks you through building custom Synapse plugins — from a minimal single-node file to a full package with vendored dependencies and custom data types.

---

## How Plugins Work

When Synapse starts, it scans the **plugin directory** for `.py` files and package folders. Any Python class that meets three requirements is auto-registered and appears in the Node Explorer:

1. Inherits from `BaseExecutionNode`
2. Has `__identifier__` starting with `'plugins.'`
3. Has a `NODE_NAME` attribute

No configuration files, no manual registration. Drop a file in, click **View > Reload Plugins**, and your nodes appear.

### Plugin Directory Locations

| Environment | Path |
|-------------|------|
| Development (from source) | `./plugins/` (next to `main.py`) |
| macOS (frozen app) | `~/Library/Application Support/Synapse/plugins/` |
| Windows (frozen app) | `%APPDATA%\Synapse\plugins\` |
| Linux (frozen app) | `~/.synapse/plugins/` |

You can also open the plugin directory from **Plugins > Plugin Manager > Open Folder**.

---

## Quick Start: Minimal Plugin

Create a file called `my_plugin.py` in the plugin directory:

```python
from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import ImageData
import numpy as np

class InvertImageNode(BaseExecutionNode):
    """Invert pixel intensities (output = 1.0 - input)."""

    __identifier__ = 'plugins.Plugins.MyPlugin'
    NODE_NAME      = 'Invert Image'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])

    def evaluate(self):
        # 1. Read from the input port
        in_port = self.inputs().get('image')
        if not (in_port and in_port.connected_ports()):
            return False, "No input connected"

        connected = in_port.connected_ports()[0]
        data = connected.node().output_values.get(connected.name())

        if not isinstance(data, ImageData):
            return False, "Expected ImageData"

        # 2. Process — payload is a float32 numpy array in [0, 1]
        arr = data.payload          # shape: (H, W) grayscale or (H, W, 3) RGB
        result = 1.0 - arr          # invert: black becomes white, white becomes black

        # 3. Write to the output port, carrying over upstream metadata
        #    (bit_depth, scale_um, source_path, etc.)
        upstream_meta = {f: getattr(data, f) for f in data.model_fields if f != 'payload'}
        self.output_values['image'] = ImageData(payload=result, **upstream_meta)
        self.mark_clean()
        return True, None
```

> **Tip:** The two lines that propagate metadata are common enough that there's a
> shortcut: `self._make_image_output(arr)` does the same thing (find upstream ImageData,
> copy all fields except payload, store to `output_values['image']`).
> There's also `self._get_input_image_data()` which replaces the port-reading boilerplate.
> See the [API Reference](#api-reference) at the bottom of this page.

Click **View > Reload Plugins** and your node appears under **Plugins > MyPlugin** in the Node Explorer.

---

## Anatomy of a Node Class

### Required Attributes

```python
class MyNode(BaseExecutionNode):
    __identifier__ = 'plugins.Plugins.Category'  # Must start with 'plugins.'
    NODE_NAME      = 'My Node'                    # Display name in Node Explorer
    PORT_SPEC      = {                            # Port declaration for docs/LLM/tree
        'inputs':  ['image', 'mask'],
        'outputs': ['image']
    }
```

**`__identifier__`** controls the Node Explorer tree hierarchy:

| Identifier | Node Explorer Location |
|------------|----------------------|
| `'plugins.Plugins.MyPlugin'` | Plugins > MyPlugin |
| `'plugins.Plugins.Segmentation'` | Plugins > Segmentation |
| `'plugins.Plugins.Cheminformatics'` | Plugins > Cheminformatics |

All nodes with the same `__identifier__` are grouped together in the tree.

**`PORT_SPEC`** is used by the documentation generator, the LLM assistant's node catalog, and the Node Explorer tree panel (to show colored port indicators next to each node). Each name should be a key from `PORT_COLORS` (detailed in the following section, **Built-in Port Colors**) so the tree shows the correct color. It should mirror your `add_input()` / `add_output()` calls.

### Docstrings

The class docstring is shown in the Node Help panel. The first line is also used as the short description in the LLM assistant's node catalog. When searching for nodes in the Node Explorer, the full docstring text is included in the search, so adding relevant keywords helps users find your node.

```python
class GaussianBlurNode(BaseExecutionNode):
    """Apply a Gaussian blur filter with configurable sigma radius.

    Accepts grayscale or RGB images. The sigma parameter controls
    how much the image is smoothed.

    Keywords: smooth, denoise, blur
    """
```

---

## Ports: Inputs and Outputs

### Adding Ports

Ports are added in `__init__()` with type-specific colors:

```python
def __init__(self):
    super().__init__()
    self.add_input('image',  color=PORT_COLORS['image'])
    self.add_input('mask',   color=PORT_COLORS['mask'])
    self.add_output('image',  color=PORT_COLORS['image'])
    self.add_output('table',  color=PORT_COLORS['table'])
```

### Built-in Port Colors

| Key | Color | Data Type | Python Type |
|-----|-------|-----------|-------------|
| `'image'` | Green | Image | `ImageData` (numpy float32 [0,1]) |
| `'mask'` | Forest green | Binary mask | `MaskData` (numpy uint8, 0 or 255) |
| `'label'` | Chartreuse | Label array | `LabelData` (numpy int32) |
| `'skeleton'` | Yellow-green | Skeleton mask | `SkeletonData` (numpy uint8) |
| `'table'` | Blue | Data table | `TableData` (pandas DataFrame) |
| `'stat'` | Royal blue | Statistics | `StatData` (pandas DataFrame) |
| `'figure'` | Purple | Plot | `FigureData` (matplotlib Figure) |
| `'path'` | Gray | File path | `str` |
| `'any'` | Dark gray | Any type | any Python object |

### Custom Port Colors

If your plugin introduces a new data type, register a color for it **before** your class definitions. The key you add becomes usable in `PORT_SPEC`, `add_input()`, and `add_output()`, and the Node Explorer tree will show the matching color.

```python
from nodes.base import PORT_COLORS

# Register at module level, before any class definitions.
# Use setdefault() so you don't overwrite colors registered by other plugins.
PORT_COLORS.setdefault('spectra', (255, 165, 0))   # Orange
PORT_COLORS.setdefault('tensor',  (220,  80, 220)) # Magenta

class MySpectraNode(BaseExecutionNode):
    PORT_SPEC = {'inputs': ['image'], 'outputs': ['spectra']}

    def __init__(self):
        super().__init__()
        self.add_input('image',    color=PORT_COLORS['image'])
        self.add_output('spectra', color=PORT_COLORS['spectra'])
```

---

## Properties and Widgets

Properties are node parameters that the user can adjust. They are shown on the node card and in the Properties panel. Add them in `__init__()`.

### Integer Spinbox

```python
self._add_int_spinbox('radius', 'Radius', value=3, min_val=1, max_val=100, step=1)
```

### Float Spinbox

```python
self._add_float_spinbox('sigma', 'Sigma', value=1.0,
                         min_val=0.01, max_val=50.0, step=0.1, decimals=2)
```

### Combo Box (Dropdown)

```python
self.add_combo_menu('method', 'Method', items=['gaussian', 'median', 'bilateral'])
```

### File/Directory Selectors

```python
# These are NodeBaseWidget subclasses — add with add_custom_widget()
from nodes.base import NodeFileSelector, NodeDirSelector, NodeFileSaver

file_w = NodeFileSelector(self.view, name='file_path', label='Input File',
                          ext='*.tif *.png *.jpg')
self.add_custom_widget(file_w)

dir_w = NodeDirSelector(self.view, name='folder', label='Folder')
self.add_custom_widget(dir_w)

save_w = NodeFileSaver(self.view, name='save_path', label='Save As',
                       ext='*.csv')
self.add_custom_widget(save_w)
```

### Color Picker

```python
from nodes.base import NodeColorPickerWidget

color_w = NodeColorPickerWidget(self.view, name='dot_color', label='Dot Color')
self.add_custom_widget(color_w)
color_w.value_changed.connect(lambda name, val: self.set_property(name, val))
```

### Compact Row (Multiple Values)

```python
self._add_row('size', 'Size', fields=[
    {'name': 'width',  'label': 'W', 'type': 'int', 'value': 512, 'min_val': 1, 'max_val': 8192},
    {'name': 'height', 'label': 'H', 'type': 'int', 'value': 512, 'min_val': 1, 'max_val': 8192},
])
```

### Reading Property Values

```python
def evaluate(self):
    radius = self.get_property('radius')
    sigma  = self.get_property('sigma')
    method = self.get_property('method')
    ...
```

### Hidden Properties (No Widget)

For internal state that should be serialized with the workflow but not shown to the user:

```python
from NodeGraphQt.constants import NodePropWidgetEnum

self.create_property('internal_state', value='default',
                     widget_type=NodePropWidgetEnum.HIDDEN.value)
```

---

## The `evaluate()` Method

This is where your processing logic lives. It is called by the execution engine whenever the node is dirty (inputs changed or properties changed).

### Return Signature

```python
def evaluate(self) -> tuple[bool, str | None]:
```

- Return `(True, None)` on success
- Return `(False, "Error message")` on failure

### Reading Input Data

For image inputs, use the built-in helper:

```python
def evaluate(self):
    data = self._get_input_image_data()  # returns ImageData or None
    if data is None:
        return False, "No input connected"

    arr = data.payload  # numpy float32 array, values in [0, 1]
    # arr.shape is (H, W) for grayscale, (H, W, 3) for RGB
    ...
```

For non-image inputs, read the port manually:

```python
in_port = self.inputs().get('table')
if not (in_port and in_port.connected_ports()):
    return False, "No input connected"

connected = in_port.connected_ports()[0]
data = connected.node().output_values.get(connected.name())
```

### Writing Output Data

For image outputs, use `_make_image_output()` which propagates upstream metadata (bit_depth, scale_um, etc.):

```python
self._make_image_output(result_array)                  # default port name 'image'
self._make_image_output(result_array, port_name='out') # custom port name
```

For non-image outputs, store directly:

```python
self.output_values['table'] = TableData(payload=dataframe)
self.output_values['mask']  = MaskData(payload=binary_uint8)
```

### Full Pattern

```python
def evaluate(self):
    # 1. Read inputs
    data = self._get_input_image_data()
    if data is None:
        return False, "No input connected"

    # 2. Read properties
    threshold = self.get_property('threshold')

    # 3. Report progress (optional)
    self.set_progress(10)

    # 4. Do work — payload is float32 [0, 1]
    arr = data.payload
    if arr.ndim == 3:
        arr = arr.mean(axis=2)  # convert to grayscale using simple mean
    result = (arr > threshold).astype(np.uint8) * 255

    # 5. Store outputs
    self.output_values['mask'] = MaskData(payload=result)

    # 6. Update display (optional, shows thumbnail on node)
    self.set_display(result)

    # 7. Finish
    self.set_progress(100)
    self.mark_clean()
    return True, None
```

### Important Rules

- Always call `self.mark_clean()` before returning `(True, None)`
- Never call `self.mark_clean()` when returning `(False, ...)`
- `evaluate()` runs on a **background thread** — do not create or modify Qt widgets here
- Use `self.set_progress()` and `self.set_display()` for thread-safe UI updates

---

## Data Types

All data flowing between nodes is wrapped in a `NodeData` subclass (from `data_models.py`).

### Built-in Types

| Type | Parent | Payload |
|------|--------|---------|
| `ImageData` | `NodeData` | `np.ndarray` float32 [0,1] |
| `MaskData` | `ImageData` | `np.ndarray` uint8 (0 or 255) |
| `SkeletonData` | `MaskData` | `np.ndarray` uint8 |
| `LabelData` | `NodeData` | `np.ndarray` int32 |
| `TableData` | `NodeData` | `pd.DataFrame` or `pd.Series` |
| `StatData` | `TableData` | `pd.DataFrame` |
| `FigureData` | `NodeData` | `matplotlib.figure.Figure` |

Since `MaskData` and `SkeletonData` inherit from `ImageData`, any node that accepts `ImageData` input can also accept a mask or skeleton. Similarly, `StatData` inherits from `TableData`, so it works anywhere a table is expected.

All types are imported from `data_models` (e.g. `from data_models import ImageData, MaskData`).

`ImageData` has additional fields beyond `payload`:

- `bit_depth: int` — original bit depth (8, 12, 14, or 16). Default 8.
- `scale_um: float | None` — physical pixel size in micrometers

All types share these optional fields:

- `metadata: dict` — arbitrary key-value pairs (e.g., `{'frame': 1, 'file': 'img_001.tif'}`)
- `source_path: str | None` — originating file path

### Creating Custom Data Types

```python
from data_models import NodeData
from typing import Any

class SpectraData(NodeData):
    """Wraps a 1-D numpy float array (intensity vs channel)."""
    payload: Any   # np.ndarray shape (N,)
```

Use `payload: Any` for types that Pydantic cannot natively validate (numpy arrays, custom objects, etc.).

### Batch Merge Support

If your data type will be used with batch processing, implement `merge()`. The `Batch Accumulator` node calls this method after all iterations finish to combine results from each iteration into a single output (usually a `TableData`):

```python
class SpectraData(NodeData):
    payload: Any

    @classmethod
    def merge(cls, items: list) -> "TableData":
        """Merge spectra from all batch frames into a table."""
        rows = []
        for item in items:
            meta = getattr(item, 'metadata', {}) or {}
            rows.append({
                'frame': meta.get('frame', ''),
                'file': meta.get('file', ''),
                'mean_intensity': float(item.payload.mean()),
            })
        return TableData(payload=pd.DataFrame(rows))
```

---

## Collections

A `CollectionData` is a named bundle of `NodeData` items that travels through the graph as a single wire. Its payload is a `dict[str, NodeData]` — each key is a user-defined name, each value is a data item (typically all the same type, but mixed types are allowed).

```python
from data_models import CollectionData, ImageData

# A collection of three images
col = CollectionData(payload={
    'sample_A': ImageData(payload=arr_a),
    'sample_B': ImageData(payload=arr_b),
    'control':  ImageData(payload=arr_c),
})
```

### Making a Node Collection-Aware

Add `_collection_aware = True` to your class:

```python
class InvertImageNode(BaseExecutionNode):
    _collection_aware = True   # <-- this is all you need

    def evaluate(self):
        # Write this as if it handles a single item.
        # The engine takes care of the rest.
        data = self._get_input_image_data()
        if data is None:
            return False, "No input connected"
        self.output_values['image'] = ImageData(payload=1.0 - data.payload)
        self.mark_clean()
        return True, None
```

When a `CollectionData` arrives at a collection-aware node, the execution engine automatically:

1. Unpacks the collection into individual items
2. Calls `evaluate()` once per item (swapping in each item as though it were a normal single input)
3. Repacks the per-item outputs into a new `CollectionData` on each output port, preserving the original names

Your `evaluate()` code never sees the collection — it just processes one item at a time as usual. The progress bar updates across all items automatically.

### When to Use It

Set `_collection_aware = True` on **stateless processing nodes** — nodes that read input, do some computation, and write output. Gaussian blur, threshold, invert, measure properties, that sort of thing. If your node would work correctly called N times in a row with different inputs, it's a good candidate.

### When NOT to Use It

Do **not** set `_collection_aware = True` on nodes that:

- Have embedded editors, drawing canvases, or interactive widgets (ROI drawing, SAM2 click-to-segment)
- Maintain internal state across calls (accumulators, batch writers)
- Display histograms or plots that depend on seeing all items at once
- Natively handle collections themselves (use `_handles_collection = True` instead)

### Mixed Inputs: Single + Collection

If a collection-aware node has two input ports and one receives a `CollectionData` while the other receives a plain `NodeData`, the single item is **broadcast** to every iteration. For example, a "Mask Image" node with an `image` collection (3 items) and a single `mask` input will apply that same mask to all 3 images.

### Mixed Inputs: Two Collections

When both inputs are collections, items are **paired by name**. If collection A has keys `['sample_1', 'sample_2']` and collection B has keys `['sample_1', 'sample_2']`, then `evaluate()` runs twice — once with `sample_1` from both, once with `sample_2` from both. If a name exists in one collection but not the other, the engine falls back to the first item of the collection that's missing the key.

### Built-in Collection Nodes

| Node | Purpose |
|------|---------|
| **Collect** | Multi-input port. Connect several items, name each one. Outputs a single `CollectionData`. Also accepts existing collections as input and merges their items in. |
| **Select Collection** | Pick one item by name from a collection. Text field + dropdown menu. |
| **Pop Collection** | Extract one item by name. Two outputs: the extracted **item** and the **rest** (collection without it). |
| **Split Collection** | Split a collection into two groups. Type names separated by `\|` or pick from the checkable dropdown. Outputs **selected** and **rest**. |
| **Save Collection** | Save all items to disk. Accepts a path input (from Path Modifier) or manual folder + extension. Each item is saved with its name as a suffix. |

---

## Node Lifecycle

### Dirty State

Nodes track whether they need re-execution:

- **Dirty** (blue) — needs re-execution. Set when inputs connect/disconnect, properties change, or upstream nodes change.
- **Clean** (dark) — successfully executed, output is cached.
- **Error** (red) — execution failed.
- **Disabled** (gray) — skipped during execution.

Property changes automatically trigger `mark_dirty()`, **except** for UI-only properties like `color`, `pos`, `selected`, `name`, `progress`, `table_view`, `image_view`, `show_preview`, `live_preview`.

### Execution Order

The execution engine:

1. Topologically sorts all nodes (upstream first)
2. Skips clean and disabled nodes
3. Calls `evaluate()` on each dirty node
4. On success: `mark_clean()` is called (you must call it yourself)
5. On failure: `mark_error()` is called automatically

### Batch Processing Hooks

When a batch runs (via `Folder Iterator`), the execution engine calls two hooks on every node in the graph:

- `on_batch_start()` — called once before the first iteration. Use it to initialize state, open resources, or clear accumulated results.
- `on_batch_end()` — called once after the last iteration. Use it to finalize output, close resources, or merge collected data.

Between these two calls, `evaluate()` runs once per iteration as usual.

Most nodes don't need to override these. They're useful when your node needs to manage something across the entire batch rather than per-iteration. For example, the built-in `Batch Accumulator` node uses `on_batch_start()` to clear its internal list and `on_batch_end()` to merge all collected items into a single output.

```python
class CsvBatchWriterNode(BaseExecutionNode):
    """Write each batch iteration's table as a row to a single CSV file."""

    __identifier__ = 'plugins.Plugins.MyPlugin'
    NODE_NAME      = 'CSV Batch Writer'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': []}

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'])
        from nodes.base import NodeFileSaver
        save_w = NodeFileSaver(self.view, name='save_path', label='Save As',
                               ext='*.csv')
        self.add_custom_widget(save_w)
        self._rows = []

    def on_batch_start(self):
        # Clear accumulated rows before each batch run
        self._rows = []

    def evaluate(self):
        in_port = self.inputs().get('table')
        if not (in_port and in_port.connected_ports()):
            return False, "No input connected"

        connected = in_port.connected_ports()[0]
        data = connected.node().output_values.get(connected.name())
        if data is None:
            return False, "No data"

        # Collect each iteration's table
        self._rows.append(data.payload)
        self.mark_clean()
        return True, None

    def on_batch_end(self):
        # After all iterations, write everything to one CSV
        import pandas as pd
        if self._rows:
            combined = pd.concat(self._rows, ignore_index=True)
            save_path = self.get_property('save_path')
            if save_path:
                combined.to_csv(save_path, index=False)
        self._rows = []
```

---

## Progress and Display

### Progress Bar

Every node gets a progress bar by default. Update it with:

```python
self.set_progress(50)   # 0-100, thread-safe
```

To create a node without a progress bar, pass `use_progress=False`:

```python
def __init__(self):
    super().__init__(use_progress=False)
```

### Display Thumbnail

Nodes that extend `BaseImageProcessNode` (instead of `BaseExecutionNode`) get a built-in image preview widget on the node card. Call `set_display()` inside `evaluate()` to update it:

```python
from nodes.base import BaseImageProcessNode

class MyImageNode(BaseImageProcessNode):
    ...
    def evaluate(self):
        ...
        self.set_display(numpy_array)  # Thread-safe, accepts numpy arrays
```

`BaseImageProcessNode` also adds a "Show Preview" checkbox that lets the user toggle the thumbnail on/off to save canvas space. If your node doesn't output images, just use `BaseExecutionNode`.

---

## Plugin Formats

### Single File Plugin

A single `.py` file in the plugin directory. Best for simple plugins with 1-3 nodes.

```
plugins/
    my_plugin.py
```

### Package Plugin

A directory with `__init__.py`. Best for complex plugins with multiple modules.

```
plugins/
    my_plugin/
        __init__.py       # Must import/export all node classes
        processing.py     # Helper modules
        models.py
```

Your `__init__.py` must import all node classes so the loader can find them:

```python
# my_plugin/__init__.py
from .processing import MyProcessNode, MyFilterNode
from .models import MyCustomData
```

### Package with Vendored Dependencies

If your plugin needs third-party packages not bundled with Synapse, include them in a `vendor/` subdirectory. The plugin loader automatically prepends `vendor/` to `sys.path` before importing your package.

```
plugins/
    my_plugin/
        __init__.py
        core.py
        vendor/
            some_library/
                __init__.py
                ...
```

This allows `import some_library` to work without system-level installation.

---

## Distribution

### As a `.zip` File

Zip your package directory:

```bash
zip -r my_plugin.zip my_plugin/
```

Users install via **Plugins > Plugin Manager > Install Plugin...** and select the `.zip`.

### As a `.synpkg` File

Use the `synapse package` command to create zstd-compressed tar archives:

```bash
# Full package (includes vendor/ directory)
synapse package plugins/my_plugin

# Skip vendored dependencies (smaller, user installs deps separately)
synapse package plugins/my_plugin --no-vendor

# Source only (no vendor/, no data/)
synapse package plugins/my_plugin --slim

# Custom output directory
synapse package plugins/my_plugin -o ~/Desktop
```

The output filename includes platform, architecture, and Python version automatically (e.g. `my_plugin-darwin-arm64-cp313-20260318.synpkg`).

To install a `.synpkg` from the command line:

```bash
synapse package --install my_plugin.synpkg
```

Users can also install via **Plugins > Plugin Manager > Install Plugin...** in the app.

### Available Libraries

Plugins can only import libraries bundled with the Synapse app:

- **numpy**, **scipy**, **pandas** — numerical computing
- **PIL / Pillow** — image I/O and manipulation
- **scikit-image** (skimage) — image processing and analysis
- **matplotlib** — plotting and figures
- **PySide6** — Qt widgets (for custom node UIs)
- **pydantic** — data validation (used by `NodeData`)

Anything else must be vendored inside your plugin package.

---

## Complete Example: Multi-Output Node with Properties

```python
from nodes.base import BaseExecutionNode, PORT_COLORS
from data_models import ImageData, MaskData, TableData
import numpy as np
import pandas as pd

class ParticleDetectorNode(BaseExecutionNode):
    """Detect bright particles above a threshold and measure their properties."""

    __identifier__ = 'plugins.Plugins.MyPlugin'
    NODE_NAME      = 'Particle Detector'
    PORT_SPEC      = {
        'inputs':  ['image'],
        'outputs': ['mask', 'table']
    }

    def __init__(self):
        super().__init__()
        # Ports
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('mask',  color=PORT_COLORS['mask'])
        self.add_output('table', color=PORT_COLORS['table'])

        # Properties — threshold is in [0, 1] to match the float32 pipeline
        self._add_float_spinbox('threshold', 'Threshold', value=0.5,
                                min_val=0.0, max_val=1.0, step=0.01, decimals=3)
        self._add_int_spinbox('min_area', 'Min Area (px)', value=10,
                              min_val=1, max_val=100000)

    def evaluate(self):
        # Read input — payload is float32 [0, 1]
        data = self._get_input_image_data()
        if data is None:
            return False, "No input connected"

        self.set_progress(10)

        # Convert to grayscale if RGB
        arr = data.payload
        if arr.ndim == 3:
            arr = arr.mean(axis=2)

        # Threshold
        threshold = self.get_property('threshold')
        binary = arr > threshold

        self.set_progress(40)

        # Label connected components
        from skimage.measure import label, regionprops_table
        labels = label(binary)

        # Measure properties
        min_area = self.get_property('min_area')
        props = regionprops_table(labels, intensity_image=arr,
                                  properties=['label', 'area', 'centroid',
                                              'mean_intensity'])
        df = pd.DataFrame(props)

        # Filter by minimum area
        df = df[df['area'] >= min_area].reset_index(drop=True)

        self.set_progress(80)

        # Create output mask (uint8, 0 or 255)
        mask_arr = np.isin(labels, df['label'].values).astype(np.uint8) * 255

        # Rename columns
        df = df.rename(columns={
            'centroid-0': 'centroid_y',
            'centroid-1': 'centroid_x',
        })

        # Store outputs
        self.output_values['mask']  = MaskData(payload=mask_arr)
        self.output_values['table'] = TableData(payload=df)

        self.set_display(mask_arr)
        self.set_progress(100)
        self.mark_clean()
        return True, None
```

---

## Debugging Tips

- **Node doesn't appear?** Check that `__identifier__` starts with `'plugins.'` and `NODE_NAME` is set.
- **Import errors?** Open **Plugins > Plugin Manager** — the Status column shows error messages per file.
- **Evaluate not called?** Make sure the node is connected and marked dirty. Check that you're returning `(True, None)` or `(False, "message")`.
- **UI freezes?** Your `evaluate()` runs on a background thread. Never call Qt widget methods directly — use `self.set_progress()` and `self.set_display()` instead.
- **Property not updating?** Use `self.get_property('name')` inside `evaluate()`, not cached instance variables. Properties auto-trigger `mark_dirty()` when changed.

---

## API Reference

### BaseExecutionNode Methods

**Lifecycle (override these):**

| Method | Description |
|--------|-------------|
| `evaluate()` | Your processing logic. Return `(True, None)` on success, `(False, "message")` on failure. |
| `on_batch_start()` | Called once before the batch loop begins. Override to initialize accumulators. |
| `on_batch_end()` | Called once after all batch items are processed. |

**Port management (call in `__init__`):**

| Method | Description |
|--------|-------------|
| `add_input(name, color=...)` | Add an input port. |
| `add_output(name, color=...)` | Add an output port. |

**Reading inputs:**

| Method | Description |
|--------|-------------|
| `self.inputs()` | Dict of input ports. Use `self.inputs().get('name')` to get a port. |
| `port.connected_ports()` | List of ports connected to this port. Index `[0]` for the first connection. |
| `connected.node().output_values.get(connected.name())` | Read the data from a connected upstream port. |
| `_get_input_image_data()` | Shortcut: find the first connected `ImageData` across all input ports. Returns `None` if not found. |

**Writing outputs:**

| Method | Description |
|--------|-------------|
| `self.output_values['port_name'] = data` | Store output data on a named port. |
| `_make_image_output(arr, port_name='image')` | Shortcut: wrap a numpy array as `ImageData`, propagate upstream metadata (bit_depth, scale_um, etc.), and store on the named port. |

**Properties:**

| Method | Description |
|--------|-------------|
| `get_property(name)` | Read a property value. |
| `set_property(name, value)` | Update a property value. Triggers `mark_dirty()` automatically. |
| `create_property(name, value, ...)` | Create a new property. Raises if it already exists. |

**Node state:**

| Method | Description |
|--------|-------------|
| `mark_clean()` | Mark node as successfully executed. You must call this before returning `(True, None)`. |
| `mark_dirty()` | Mark node as needing re-execution. Cascades to all downstream nodes. |
| `mark_disabled()` | Disable the node so it is skipped during execution. |
| `mark_enabled()` | Re-enable a disabled node. |
| `clear_cache()` | Clear `output_values` and force re-evaluation on next run. |
| `_collection_aware = True` | Class attribute. Enables auto-loop over `CollectionData` inputs — the engine unpacks, runs `evaluate()` per item, and repacks outputs. |

**UI feedback (thread-safe, can be called from `evaluate()`):**

| Method | Description |
|--------|-------------|
| `set_progress(value)` | Update progress bar (0-100). |
| `set_display(data)` | Show a preview thumbnail on the node card. Accepts numpy arrays. |

### Widget Helpers

Call these in `__init__()` to add user-facing controls to the node card.

| Method | Description |
|--------|-------------|
| `_add_int_spinbox(name, label, value, min_val, max_val, step)` | Integer spinbox. |
| `_add_float_spinbox(name, label, value, min_val, max_val, step, decimals)` | Float spinbox. |
| `_add_row(name, label, fields)` | Compact horizontal row of labeled spinboxes. |
| `_add_column_selector(name, label, text, mode)` | Text input with column dropdown. `mode`: `'single'` or `'multi'`. |
| `_refresh_column_selectors(df, *prop_names)` | Update column selector dropdowns with columns from a DataFrame. Thread-safe. |
| `_add_list_input(name, label, items)` | Reorderable list widget. |
| `add_combo_menu(name, label, items)` | Dropdown menu. |
| `add_custom_widget(widget)` | Embed any `NodeBaseWidget` subclass. |

### Available Custom Widgets

Import these from `nodes.base` for use with `add_custom_widget()`:

| Widget | Description |
|--------|-------------|
| `NodeFileSelector(view, name, label, ext)` | File open dialog. `ext`: e.g. `'*.tif *.png *.jpg'`. |
| `NodeFileSaver(view, name, label, ext)` | File save dialog. |
| `NodeDirSelector(view, name, label)` | Directory picker. |
| `NodeColorPickerWidget(view, name, label)` | Color picker button. |
| `NodeColumnSelectorWidget(view, name, label, text, mode)` | Text field with column dropdown. |
| `NodeChannelSelectorWidget(view, name, label, text)` | Channel toggle selector (1-4 + PAD). |
