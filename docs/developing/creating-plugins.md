# Creating Plugins

This guide walks you through building custom Synapse plugins — from a minimal single-node file to a full package with vendored dependencies and custom data types.

---

## How Plugins Work

When Synapse starts, it scans the **plugin directory** for `.py` files and package folders. Any Python class that meets three requirements is auto-registered and appears in the Node Explorer:

1. Inherits from `BaseExecutionNode`
2. Has `__identifier__` starting with `'plugins.'`
3. Has a `NODE_NAME` attribute

No configuration files, no manual registration. Drop a file in, restart, and your nodes appear.

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
from PIL import Image

class InvertImageNode(BaseExecutionNode):
    """Invert pixel intensities (output = 255 - input)."""

    __identifier__ = 'plugins.Plugins.MyPlugin'
    NODE_NAME      = 'Invert Image'
    PORT_SPEC      = {'inputs': ['image'], 'outputs': ['image']}

    def __init__(self):
        super().__init__()
        self.add_input('image',  color=PORT_COLORS['image'])
        self.add_output('image', color=PORT_COLORS['image'])

    def evaluate(self):
        # 1. Get input data
        in_port = self.inputs().get('image')
        if not (in_port and in_port.connected_ports()):
            return False, "No input connected"

        connected = in_port.connected_ports()[0]
        data = connected.node().output_values.get(connected.name())

        if not isinstance(data, ImageData):
            return False, "Expected ImageData"

        # 2. Process
        arr = np.array(data.payload)
        result = (255 - arr).astype(arr.dtype)

        # 3. Store output and finish
        self.output_values['image'] = ImageData(payload=Image.fromarray(result))
        self.mark_clean()
        return True, None
```

Restart Synapse. Your node appears under **Plugins > MyPlugin** in the Node Explorer.

---

## Anatomy of a Node Class

### Required Attributes

```python
class MyNode(BaseExecutionNode):
    __identifier__ = 'plugins.Plugins.Category'  # Must start with 'plugins.'
    NODE_NAME      = 'My Node'                    # Display name in Node Explorer
    PORT_SPEC      = {                            # Port declaration for docs/LLM
        'inputs':  ['image', 'mask'],
        'outputs': ['result']
    }
```

**`__identifier__`** controls the Node Explorer tree hierarchy:

| Identifier | Node Explorer Location |
|------------|----------------------|
| `'plugins.Plugins.MyPlugin'` | Plugins > MyPlugin |
| `'plugins.Plugins.Segmentation'` | Plugins > Segmentation |
| `'plugins.Plugins.Cheminformatics'` | Plugins > Cheminformatics |

All nodes with the same `__identifier__` are grouped together in the tree.

**`PORT_SPEC`** is used by the documentation generator and the LLM assistant's node catalog. It should mirror your `add_input()` / `add_output()` calls.

### Docstrings

The first line of the class docstring is shown in the Node Help panel and the LLM assistant catalog. Write a concise, descriptive one-liner:

```python
class GaussianBlurNode(BaseExecutionNode):
    """Apply a Gaussian blur filter with configurable sigma radius."""
```

Additional lines are shown in the expandable "Details" section of the generated documentation.

---

## Ports: Inputs and Outputs

### Adding Ports

Ports are added in `__init__()` with type-specific colors:

```python
def __init__(self):
    super().__init__()
    self.add_input('image',  color=PORT_COLORS['image'])
    self.add_input('mask',   color=PORT_COLORS['mask'])
    self.add_output('result', color=PORT_COLORS['image'])
    self.add_output('table',  color=PORT_COLORS['table'])
```

### Built-in Port Colors

| Key | Color | Data Type | Python Type |
|-----|-------|-----------|-------------|
| `'image'` | Green | Image | `ImageData` (PIL Image) |
| `'mask'` | Forest green | Binary mask | `MaskData` (PIL Image) |
| `'label'` | Chartreuse | Label array | `LabelData` (numpy int32) |
| `'skeleton'` | Yellow-green | Skeleton mask | `SkeletonData` (PIL Image) |
| `'table'` | Blue | Data table | `TableData` (pandas DataFrame) |
| `'stat'` | Royal blue | Statistics | `StatData` (pandas DataFrame) |
| `'figure'` | Purple | Plot | `FigureData` (matplotlib Figure) |
| `'path'` | Gray | File path | `str` |
| `'any'` | Dark gray | Any type | any Python object |

### Custom Port Colors

Register custom colors **before** your class definitions:

```python
PORT_COLORS['spectra'] = (255, 165, 0)   # Orange
PORT_COLORS['tensor']  = (220,  80, 220) # Magenta
```

### Reading Input Data

The standard pattern for reading an input port:

```python
def evaluate(self):
    in_port = self.inputs().get('image')
    if not (in_port and in_port.connected_ports()):
        return False, "No input connected"

    connected = in_port.connected_ports()[0]
    data = connected.node().output_values.get(connected.name())

    if not isinstance(data, ImageData):
        return False, "Expected ImageData"

    # Use data.payload (PIL Image), data.metadata, data.source_path
    ...
```

### Writing Output Data

Store results in `self.output_values` using the port name as key:

```python
self.output_values['result'] = ImageData(payload=pil_image)
self.output_values['table']  = TableData(payload=dataframe)
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

### Full Pattern

```python
def evaluate(self):
    # 1. Read inputs
    in_port = self.inputs().get('image')
    if not (in_port and in_port.connected_ports()):
        return False, "No input connected"

    connected = in_port.connected_ports()[0]
    data = connected.node().output_values.get(connected.name())
    if not isinstance(data, ImageData):
        return False, "Expected ImageData"

    # 2. Read properties
    threshold = self.get_property('threshold')

    # 3. Report progress (optional)
    self.set_progress(10)

    # 4. Do work
    arr = np.array(data.payload)
    result = arr > threshold

    # 5. Check for cancellation (for long-running operations)
    if self.cancel_requested:
        return False, "Cancelled"

    # 6. Store outputs
    self.output_values['mask'] = MaskData(
        payload=Image.fromarray(result.astype(np.uint8) * 255)
    )

    # 7. Update display (optional, shows thumbnail on node)
    self.set_display(result)

    # 8. Finish
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

## Properties and Widgets

Properties are node parameters that the user can adjust. They are shown on the node card and in the Properties panel.

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

## Data Types

All data flowing between nodes is wrapped in a `NodeData` subclass (from `data_models.py`).

### Built-in Types

| Type | Payload | Import |
|------|---------|--------|
| `ImageData` | `PIL.Image.Image` | `from data_models import ImageData` |
| `MaskData` | `PIL.Image.Image` (binary) | `from data_models import MaskData` |
| `LabelData` | `np.ndarray` (int32) | `from data_models import LabelData` |
| `SkeletonData` | `PIL.Image.Image` (thinned) | `from data_models import SkeletonData` |
| `TableData` | `pd.DataFrame` or `pd.Series` | `from data_models import TableData` |
| `StatData` | `pd.DataFrame` | `from data_models import StatData` |
| `FigureData` | `matplotlib.figure.Figure` | `from data_models import FigureData` |

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

If your data type will be used with batch processing (Folder Iterator), implement `merge()`:

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

If your node participates in batch workflows, override these optional methods:

```python
def on_batch_start(self):
    """Called once before the batch loop begins."""
    pass

def on_batch_end(self):
    """Called once after all batch items are processed."""
    pass
```

### Cancellation

For long-running operations, check `self.cancel_requested` periodically:

```python
def evaluate(self):
    for i, frame in enumerate(frames):
        if self.cancel_requested:
            return False, "Cancelled"
        self.set_progress(int(i / len(frames) * 100))
        # ... process frame ...
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

Show a preview image on the node card:

```python
self.set_display(numpy_array)  # Thread-safe, accepts numpy arrays
```

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

For smaller downloads, use zstd-compressed tar archives:

```bash
tar cf - my_plugin/ | zstd -o my_plugin.synpkg
```

Requires the `zstandard` Python package on the user's machine.

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
from PIL import Image

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

        # Properties
        self._add_int_spinbox('threshold', 'Threshold', value=128,
                              min_val=0, max_val=255)
        self._add_int_spinbox('min_area', 'Min Area (px)', value=10,
                              min_val=1, max_val=100000)

    def evaluate(self):
        # Read input
        in_port = self.inputs().get('image')
        if not (in_port and in_port.connected_ports()):
            return False, "No input connected"

        connected = in_port.connected_ports()[0]
        data = connected.node().output_values.get(connected.name())
        if not isinstance(data, ImageData):
            return False, "Expected ImageData"

        self.set_progress(10)

        # Convert to grayscale numpy array
        img = data.payload.convert('L')
        arr = np.array(img)

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

        # Create output mask
        mask_arr = np.isin(labels, df['label'].values).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_arr, mode='L')

        # Rename columns
        df = df.rename(columns={
            'centroid-0': 'centroid_y',
            'centroid-1': 'centroid_x',
        })

        # Store outputs
        self.output_values['mask']  = MaskData(payload=mask_img)
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

| Method | Description |
|--------|-------------|
| `evaluate()` | Override this. Return `(bool, str \| None)` |
| `add_input(name, color=...)` | Add an input port |
| `add_output(name, color=...)` | Add an output port |
| `get_property(name)` | Read a property value |
| `set_property(name, value)` | Update a property value |
| `create_property(name, value, ...)` | Create a new property |
| `mark_clean()` | Mark node as successfully executed |
| `mark_dirty()` | Mark node as needing re-execution |
| `set_progress(value)` | Update progress bar (0-100) |
| `set_display(data)` | Show preview data on the node card |
| `cancel_requested` | `True` if user clicked Stop |
| `on_batch_start()` | Called before batch loop |
| `on_batch_end()` | Called after batch loop |
| `clear_cache()` | Clear outputs and force re-evaluation |

### Widget Helpers

| Method | Description |
|--------|-------------|
| `_add_int_spinbox(name, label, value, min_val, max_val, step)` | Integer spinbox |
| `_add_float_spinbox(name, label, value, min_val, max_val, step, decimals)` | Float spinbox |
| `_add_row(name, label, fields)` | Compact row of spinboxes |
| `add_combo_menu(name, label, items)` | Dropdown menu |
| `add_custom_widget(widget)` | Embed any `NodeBaseWidget` subclass |
