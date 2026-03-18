"""
export_script.py
================
Export the current node graph as a standalone Python script.
Topologically sorts the nodes, extracts their properties, and generates
a well-structured, runnable script that reproduces the graph's pipeline.
"""

from datetime import datetime


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    """Convert a node name like 'Gaussian Blur 1' to a valid Python identifier."""
    s = name.strip().lower()
    s = s.replace(' ', '_').replace('-', '_').replace('.', '_')
    out = ''.join(c if c.isalnum() or c == '_' else '' for c in s)
    if out and out[0].isdigit():
        out = '_' + out
    return out or '_node'


def _topo_sort(nodes):
    """Topological sort — same DFS approach used by the graph executor."""
    visited = set()
    order = []

    def visit(node):
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)
        for port in node.connected_input_nodes().values():
            for upstream in port:
                visit(upstream)
        order.append(node)

    for n in nodes:
        visit(n)
    return order


def _unique_var(used: set, base: str) -> str:
    """Return a unique variable name by appending _2, _3, ... if needed."""
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f'{base}_{i}' in used:
        i += 1
    name = f'{base}_{i}'
    used.add(name)
    return name


def _get_ignore_props():
    """Properties that are internal / UI-only and should not appear in scripts."""
    return {
        'name', 'color', 'border_color', 'text_color', 'type_', 'id', 'pos',
        'layout_direction', 'selected', 'visible', 'custom', 'progress',
        'table_view', 'image_view', 'show_preview', 'live_preview',
    }


def _collect_props(node) -> dict:
    """Extract user-configurable properties from a node."""
    ignore = _get_ignore_props()
    props = {}
    try:
        custom = node.model.custom_properties
    except Exception:
        return props
    for k, v in custom.items():
        if k in ignore or k.startswith('_'):
            continue
        props[k] = v
    return props


def _repr_val(v) -> str:
    """Produce a Python repr for a property value."""
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    return repr(str(v))


def _is_terminal(node) -> bool:
    """True if none of this node's output ports are connected downstream."""
    for port in node.outputs().values():
        if port.connected_ports():
            return False
    return True


def _classify_node(cls_name):
    """Return a pipeline category string for section grouping."""
    if cls_name in ('FileReadNode', 'ImageReadNode', 'FolderIteratorNode'):
        return 'input'
    if cls_name in ('SaveNode', 'DisplayNode', 'DataTableCellNode',
                     'DataFigureCellNode', 'ImageCellNode'):
        return 'output'
    if 'Plot' in cls_name or cls_name in ('HistogramNode', 'HeatmapNode',
                                           'FigureEditNode'):
        return 'plot'
    if cls_name in ('FilterTableNode', 'MathColumnNode', 'AggregateTableNode',
                     'SortTableNode', 'TopNNode', 'SelectColumnsNode',
                     'ReshapeTableNode', 'CombineTablesNode', 'EditableTableNode',
                     'DataSummaryNode', 'RenameGroupNode',
                     'ColumnValueSplitNode', 'TwoTableMathNode',
                     'RandomSampleNode', 'FileReadNode'):
        return 'table'
    return 'process'


# ── Main export function ─────────────────────────────────────────────────────

def export_graph_to_script(graph) -> str:
    """
    Generate a standalone Python script from the current graph.
    Returns the script as a string.
    """
    all_nodes = graph.all_nodes()
    if not all_nodes:
        return "# Empty graph — nothing to export.\n"

    sorted_nodes = _topo_sort(all_nodes)

    # Assign a unique variable name to each node
    used_vars: set[str] = set()
    node_vars: dict[int, str] = {}  # id(node) → var name
    for node in sorted_nodes:
        base = _sanitize(node.name())
        var = _unique_var(used_vars, base)
        node_vars[id(node)] = var

    # Assign variable names to each output port: (node_id, port_name) → var
    port_vars: dict[tuple[int, str], str] = {}
    for node in sorted_nodes:
        outputs = list(node.outputs().values())
        var = node_vars[id(node)]
        if len(outputs) == 1:
            port_vars[(id(node), outputs[0].name())] = var
        else:
            for p in outputs:
                pname = _sanitize(p.name())
                pvar = f'{var}_{pname}'
                port_vars[(id(node), p.name())] = pvar

    # Track which imports are needed
    imports = set()
    imports.add('import numpy as np')
    imports.add('from PIL import Image')
    imports.add('import os')
    imports.add('import sys')
    imports.add('import time')

    body_lines = []
    has_plots = False
    has_tables = False
    step_num = 0

    for node in sorted_nodes:
        var = node_vars[id(node)]
        cls_name = node.__class__.__name__
        node_name = node.name()
        props = _collect_props(node)

        step_num += 1

        # Resolve inputs
        input_map: dict[str, str] = {}
        for port_name, port in node.inputs().items():
            connected = port.connected_ports()
            if connected:
                cp = connected[0]
                up_key = (id(cp.node()), cp.name())
                up_var = port_vars.get(up_key, '???')
                input_map[port_name] = up_var

        # Section separator with step number
        category = _classify_node(cls_name)
        body_lines.append(f'    # ── Step {step_num}: {node_name} ──────────────────────────────────────')

        # Progress print
        body_lines.append(f'    print(f"[{{step}}/{len(sorted_nodes)}] {node_name}...")')
        body_lines.append(f'    step += 1')
        body_lines.append(f'    t0 = time.time()')

        # Generate code
        code = _generate_node_code(cls_name, var, props, input_map, imports, node)

        if code:
            for line in code:
                body_lines.append(f'    {line}')
        else:
            body_lines.append(f'    # TODO: {cls_name} — manual translation needed')
            if props:
                body_lines.append(f'    # Properties: {props}')
            if input_map:
                body_lines.append(f'    # Inputs: {input_map}')
            body_lines.append(f'    {var} = None  # placeholder')

        # Timing
        body_lines.append(f'    print(f"  done in {{time.time() - t0:.2f}}s")')

        # Track data types for final output
        if 'Plot' in cls_name or cls_name in ('HistogramNode', 'HeatmapNode'):
            has_plots = True
        if category == 'table' or cls_name in ('ParticlePropsNode', 'DataSummaryNode',
                                                 'FileReadNode'):
            has_tables = True

        # For terminal nodes, add preview/save hints
        if _is_terminal(node):
            body_lines.append(f'')
            if category == 'plot' or 'Plot' in cls_name:
                has_plots = True
            elif cls_name == 'DisplayNode':
                pass  # already handled by print()
            elif cls_name == 'SaveNode':
                pass  # already handled by save call
            elif has_tables or 'table' in str(list(node.outputs().keys())).lower():
                body_lines.append(f'    # Terminal output — preview:')
                body_lines.append(f'    if {var} is not None:')
                body_lines.append(f'        if hasattr({var}, "to_string"):')
                body_lines.append(f'            print({var}.to_string())')
                body_lines.append(f'        else:')
                body_lines.append(f'            print(type({var}), {var})')
            else:
                # Image or mask terminal — save it
                body_lines.append(f'    # Terminal output — save result:')
                body_lines.append(f'    if {var} is not None and isinstance({var}, np.ndarray):')
                body_lines.append(f'        Image.fromarray({var}).save(os.path.join(output_dir, "{var}.png"))')
                body_lines.append(f'        print(f"  saved: {{os.path.join(output_dir, \\"{var}.png\\")}}")')

        body_lines.append('')

    # ── Compose the final script ─────────────────────────────────────────

    # Header
    header = [
        '#!/usr/bin/env python3',
        '"""',
        f'Auto-generated from Synapse node graph.',
        f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'Nodes: {len(sorted_nodes)}',
        '',
        'Review and adjust file paths / parameters before running.',
        '"""',
        '',
    ]

    # Imports — group by stdlib / third-party
    stdlib = sorted(i for i in imports if i.startswith('import ') and
                    i.split()[1].split('.')[0] in ('os', 'sys', 'time', 'pathlib', 'glob', 'json'))
    third_party = sorted(i for i in imports if i not in stdlib)
    header.extend(stdlib)
    if stdlib and third_party:
        header.append('')
    header.extend(third_party)
    header.append('')

    # Configuration section
    header.append('')
    header.append('# ═══════════════════════════════════════════════════════════════')
    header.append('#  Configuration — edit these before running')
    header.append('# ═══════════════════════════════════════════════════════════════')
    header.append('')

    # Collect file paths from I/O nodes
    file_params = []
    for node in sorted_nodes:
        cls_name = node.__class__.__name__
        props = _collect_props(node)
        var = node_vars[id(node)]
        if cls_name in ('FileReadNode', 'ImageReadNode'):
            fp = props.get('file_path', '')
            header.append(f'{var.upper()}_PATH = {_repr_val(fp)}')
            file_params.append((var, f'{var.upper()}_PATH'))
        elif cls_name == 'SaveNode':
            fp = props.get('save_path', 'output')
            header.append(f'{var.upper()}_PATH = {_repr_val(fp)}')
            file_params.append((var, f'{var.upper()}_PATH'))
        elif cls_name == 'FolderIteratorNode':
            fp = props.get('folder_path', '')
            header.append(f'{var.upper()}_FOLDER = {_repr_val(fp)}')
            file_params.append((var, f'{var.upper()}_FOLDER'))

    if not file_params:
        header.append('# (no file paths detected — add your input/output paths here)')

    header.append('')
    header.append('OUTPUT_DIR = "output"')
    header.append('')

    # Helper functions
    header.append('')
    header.append('# ═══════════════════════════════════════════════════════════════')
    header.append('#  Helper functions')
    header.append('# ═══════════════════════════════════════════════════════════════')
    header.append('')
    header.append('def to_gray(arr):')
    header.append('    """Convert array to grayscale float64."""')
    header.append('    if arr.ndim == 3:')
    header.append('        return arr.mean(axis=2).astype(np.float64)')
    header.append('    return arr.astype(np.float64)')
    header.append('')
    header.append('')
    header.append('def to_uint8(arr):')
    header.append('    """Normalize to 0-255 uint8."""')
    header.append('    if arr.dtype == np.uint8:')
    header.append('        return arr')
    header.append('    mn, mx = arr.min(), arr.max()')
    header.append('    if mx > mn:')
    header.append('        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)')
    header.append('    return np.zeros_like(arr, dtype=np.uint8)')
    header.append('')

    # Main function
    main_lines = [
        '',
        '# ═══════════════════════════════════════════════════════════════',
        '#  Main pipeline',
        '# ═══════════════════════════════════════════════════════════════',
        '',
        f'def main():',
        f'    """Run the exported pipeline."""',
        f'    total = {step_num}',
        f'    step = 1',
        f'    t_start = time.time()',
        f'',
        f'    # Create output directory',
        f'    output_dir = OUTPUT_DIR',
        f'    os.makedirs(output_dir, exist_ok=True)',
        f'    print(f"Output directory: {{os.path.abspath(output_dir)}}")',
        f'    print(f"Running pipeline ({step_num} steps)...\\n")',
        f'',
    ]

    # Footer
    footer = [
        f'    # ── Done ──────────────────────────────────────────────────────',
        f'    elapsed = time.time() - t_start',
        f'    print(f"\\nPipeline complete in {{elapsed:.1f}}s")',
    ]

    if has_plots:
        imports.add('import matplotlib.pyplot as plt')
        footer.append('')
        footer.append('    # Show all figures')
        footer.append('    plt.show()')

    footer.extend([
        '',
        '',
        'if __name__ == "__main__":',
        '    main()',
        '',
    ])

    # Re-generate imports since generators may have added more
    stdlib = sorted(i for i in imports if i.startswith('import ') and
                    i.split()[1].split('.')[0] in ('os', 'sys', 'time', 'pathlib', 'glob', 'json'))
    third_party = sorted(i for i in imports if i not in stdlib)

    # Rebuild header imports section
    final_header = header[:9]  # shebang + docstring
    final_header.extend(stdlib)
    if stdlib and third_party:
        final_header.append('')
    final_header.extend(third_party)
    final_header.extend(header[9 + len(stdlib) + len(third_party) + (1 if stdlib and third_party else 0):])

    # Actually — simpler: just rebuild cleanly
    out = []
    out.extend([
        '#!/usr/bin/env python3',
        '"""',
        f'Auto-generated from Synapse node graph.',
        f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'Nodes: {len(sorted_nodes)}',
        '',
        'Review and adjust file paths / parameters before running.',
        '"""',
        '',
    ])
    # stdlib imports
    stdlib = sorted(set(i for i in imports if i.startswith('import ') and
                    i.split()[1].split('.')[0] in ('os', 'sys', 'time', 'pathlib', 'glob', 'json')))
    third_party = sorted(set(i for i in imports if i not in set(stdlib)))
    out.extend(stdlib)
    if stdlib and third_party:
        out.append('')
    out.extend(third_party)
    out.append('')

    # Config section
    out.append('')
    out.append('# ═══════════════════════════════════════════════════════════════')
    out.append('#  Configuration — edit these before running')
    out.append('# ═══════════════════════════════════════════════════════════════')
    out.append('')

    for node in sorted_nodes:
        cls_name = node.__class__.__name__
        props = _collect_props(node)
        var = node_vars[id(node)]
        if cls_name in ('FileReadNode', 'ImageReadNode'):
            fp = props.get('file_path', '')
            out.append(f'{var.upper()}_PATH = {_repr_val(fp)}')
        elif cls_name == 'SaveNode':
            fp = props.get('save_path', 'output')
            out.append(f'{var.upper()}_PATH = {_repr_val(fp)}')
        elif cls_name == 'FolderIteratorNode':
            fp = props.get('folder_path', '')
            out.append(f'{var.upper()}_FOLDER = {_repr_val(fp)}')

    out.append('')
    out.append('OUTPUT_DIR = "output"')
    out.append('')

    # Helpers
    out.append('')
    out.append('# ═══════════════════════════════════════════════════════════════')
    out.append('#  Helpers')
    out.append('# ═══════════════════════════════════════════════════════════════')
    out.append('')
    out.append('def to_gray(arr):')
    out.append('    """Convert array to grayscale float64."""')
    out.append('    if arr.ndim == 3:')
    out.append('        return arr.mean(axis=2).astype(np.float64)')
    out.append('    return arr.astype(np.float64)')
    out.append('')
    out.append('')
    out.append('def to_uint8(arr):')
    out.append('    """Normalize to 0-255 uint8."""')
    out.append('    if arr.dtype == np.uint8:')
    out.append('        return arr')
    out.append('    mn, mx = float(arr.min()), float(arr.max())')
    out.append('    if mx > mn:')
    out.append('        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)')
    out.append('    return np.zeros_like(arr, dtype=np.uint8)')
    out.append('')

    # Main function
    out.append('')
    out.append('# ═══════════════════════════════════════════════════════════════')
    out.append('#  Main pipeline')
    out.append('# ═══════════════════════════════════════════════════════════════')
    out.append('')
    out.append('def main():')
    out.append(f'    total = {step_num}')
    out.append('    step = 1')
    out.append('    t_start = time.time()')
    out.append('')
    out.append('    output_dir = OUTPUT_DIR')
    out.append('    os.makedirs(output_dir, exist_ok=True)')
    out.append(f'    print(f"Pipeline: {len(sorted_nodes)} steps")')
    out.append('    print(f"Output:   {os.path.abspath(output_dir)}")')
    out.append('    print()')
    out.append('')
    out.extend(body_lines)
    out.extend(footer)

    return '\n'.join(out)


def _generate_node_code(cls_name, var, props, input_map, imports, node):
    """
    Generate Python code lines for a specific node type.
    Returns a list of strings, or None for unsupported nodes.
    """
    gen = _GENERATORS.get(cls_name)
    if gen:
        return gen(var, props, input_map, imports)
    return _generic_generator(cls_name, var, props, input_map, imports, node)


def _generic_generator(cls_name, var, props, input_map, imports, node):
    """Generate a best-effort comment block for unknown node types."""
    lines = []
    lines.append(f'# {cls_name}: no automatic code generator available')
    for pname, upstream_var in input_map.items():
        lines.append(f'#   input "{pname}" <- {upstream_var}')
    if props:
        lines.append(f'#   properties:')
        for k, v in props.items():
            lines.append(f'#     {k} = {_repr_val(v)}')
    outputs = list(node.outputs().values())
    out_names = [p.name() for p in outputs]
    lines.append(f'#   outputs: {out_names}')
    lines.append(f'{var} = None  # TODO: implement manually')
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
#  Individual node code generators
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_file_read(var, props, inputs, imports):
    imports.add('import pandas as pd')
    fp = props.get('file_path', '')
    sep = props.get('separator', ',')
    in_var = inputs.get('file_path')
    src = in_var if in_var else f'{var.upper()}_PATH'
    lines = [
        f'# Read tabular data (CSV / TSV)',
        f'{var} = pd.read_csv({src}, sep={_repr_val(sep)})',
        f'print(f"  loaded {{len({var})}} rows, {{{var}.shape[1]}} columns")',
    ]
    return lines


def _gen_image_read(var, props, inputs, imports):
    in_var = inputs.get('file_path')
    src = in_var if in_var else f'{var.upper()}_PATH'
    lines = [
        f'# Read image file',
        f'_pil = Image.open({src})',
        f'{var} = np.array(_pil)',
        f'print(f"  shape: {{{var}.shape}}, dtype: {{{var}.dtype}}")',
    ]
    return lines


def _gen_save(var, props, inputs, imports):
    fp = props.get('save_path', 'output')
    fmt = props.get('format', 'png')
    in_var = inputs.get('in') or inputs.get('image') or 'data'
    out_path = f'{var.upper()}_PATH'
    lines = [
        f'# Save result to file',
        f'_save_path = {out_path}',
        f'if isinstance({in_var}, np.ndarray):',
        f'    Image.fromarray(to_uint8({in_var})).save(_save_path)',
        f'elif hasattr({in_var}, "to_csv"):',
        f'    {in_var}.to_csv(_save_path, index=False)',
        f'print(f"  saved: {{_save_path}}")',
    ]
    return lines


def _gen_gaussian_blur(var, props, inputs, imports):
    imports.add('from skimage.filters import gaussian')
    sigma = props.get('sigma', 10.0)
    in_var = inputs.get('image', 'image')
    return [
        f'# Gaussian blur (sigma={sigma})',
        f'{var} = gaussian({in_var}, sigma={sigma}, preserve_range=True)',
        f'{var} = {var}.astype({in_var}.dtype)',
    ]


def _gen_binary_threshold(var, props, inputs, imports):
    in_var = inputs.get('image', 'image')
    state = props.get('thresh_state', [128.0, 1])
    thresh = state[0] if isinstance(state, list) else 128.0
    above = bool(state[1]) if isinstance(state, list) and len(state) > 1 else True
    auto = props.get('auto_otsu_per_image', True)
    op = '>' if above else '<='
    direction = 'above' if above else 'below'
    lines = [
        f'# Binary threshold (direction: keep {direction})',
        f'_gray = to_gray({in_var})',
    ]
    if auto:
        imports.add('from skimage.filters import threshold_otsu')
        lines.extend([
            f'_thresh = threshold_otsu(_gray)',
            f'print(f"  auto Otsu threshold: {{_thresh:.1f}}")',
        ])
    else:
        lines.append(f'_thresh = {thresh}')

    lines.extend([
        f'{var} = (_gray {op} _thresh).astype(np.uint8) * 255',
        f'print(f"  foreground pixels: {{{var}.sum() // 255}}")',
    ])
    return lines


def _gen_remove_small_objects(var, props, inputs, imports):
    imports.add('from skimage.morphology import remove_small_objects')
    in_var = inputs.get('mask', 'mask')
    size = props.get('max_size', 500)
    return [
        f'# Remove small objects (min_size={size} px)',
        f'_binary = {in_var} > 0',
        f'_before = _binary.sum()',
        f'{var} = remove_small_objects(_binary, min_size={size}).astype(np.uint8) * 255',
        f'print(f"  removed {{_before - ({var} > 0).sum()}} small-object pixels")',
    ]


def _gen_remove_small_holes(var, props, inputs, imports):
    imports.add('from skimage.morphology import remove_small_holes')
    in_var = inputs.get('mask', 'mask')
    size = props.get('area_threshold', 500)
    return [
        f'# Fill small holes (area_threshold={size} px)',
        f'{var} = remove_small_holes({in_var} > 0, area_threshold={size}).astype(np.uint8) * 255',
    ]


def _gen_erosion(var, props, inputs, imports):
    imports.add('from skimage.morphology import binary_erosion, disk')
    in_var = inputs.get('mask', 'mask')
    r = props.get('radius', 1)
    return [
        f'# Morphological erosion (radius={r})',
        f'{var} = binary_erosion({in_var} > 0, disk({r})).astype(np.uint8) * 255',
    ]


def _gen_dilation(var, props, inputs, imports):
    imports.add('from skimage.morphology import binary_dilation, disk')
    in_var = inputs.get('mask', 'mask')
    r = props.get('radius', 1)
    return [
        f'# Morphological dilation (radius={r})',
        f'{var} = binary_dilation({in_var} > 0, disk({r})).astype(np.uint8) * 255',
    ]


def _gen_morph_open(var, props, inputs, imports):
    imports.add('from skimage.morphology import binary_opening, disk')
    in_var = inputs.get('mask', 'mask')
    r = props.get('radius', 1)
    return [
        f'# Morphological opening (radius={r})',
        f'{var} = binary_opening({in_var} > 0, disk({r})).astype(np.uint8) * 255',
    ]


def _gen_morph_close(var, props, inputs, imports):
    imports.add('from skimage.morphology import binary_closing, disk')
    in_var = inputs.get('mask', 'mask')
    r = props.get('radius', 1)
    return [
        f'# Morphological closing (radius={r})',
        f'{var} = binary_closing({in_var} > 0, disk({r})).astype(np.uint8) * 255',
    ]


def _gen_fill_holes(var, props, inputs, imports):
    imports.add('from scipy.ndimage import binary_fill_holes')
    in_var = inputs.get('mask', 'mask')
    return [
        f'# Fill all holes in binary mask',
        f'{var} = binary_fill_holes({in_var} > 0).astype(np.uint8) * 255',
    ]


def _gen_skeletonize(var, props, inputs, imports):
    imports.add('from skimage.morphology import skeletonize')
    in_var = inputs.get('mask', 'mask')
    return [
        f'# Skeletonize binary mask to 1-pixel-wide skeleton',
        f'{var} = skeletonize({in_var} > 0).astype(np.uint8) * 255',
    ]


def _gen_watershed(var, props, inputs, imports):
    imports.add('from skimage.segmentation import watershed')
    imports.add('from scipy.ndimage import distance_transform_edt, label as nd_label')
    imports.add('from skimage.feature import peak_local_max')
    in_var = inputs.get('mask', 'mask')
    min_dist = props.get('min_distance', 10)
    return [
        f'# Watershed segmentation (min_distance={min_dist})',
        f'_binary = {in_var} > 0',
        f'_dist = distance_transform_edt(_binary)',
        f'',
        f'# Find seed points at local maxima of distance transform',
        f'_coords = peak_local_max(_dist, min_distance={min_dist}, labels=_binary)',
        f'_markers = np.zeros_like(_binary, dtype=bool)',
        f'if len(_coords):',
        f'    _markers[tuple(_coords.T)] = True',
        f'_markers, _ = nd_label(_markers)',
        f'',
        f'# Run watershed',
        f'{var} = watershed(-_dist, _markers, mask=_binary)',
        f'print(f"  found {{{var}.max()}} regions")',
    ]


def _gen_particle_props(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('from skimage.measure import regionprops_table, label as sk_label')
    in_var = inputs.get('mask', inputs.get('label', 'mask'))
    img_var = inputs.get('image')
    lines = [
        f'# Measure region properties',
        f'_labeled = sk_label({in_var} > 0) if {in_var}.max() <= 1 else {in_var}',
        f'',
        f'_props = regionprops_table(_labeled, properties=[',
        f"    'label', 'area', 'centroid',",
        f"    'axis_major_length', 'axis_minor_length',",
        f"    'eccentricity', 'equivalent_diameter_area',",
        f"    'perimeter', 'solidity', 'orientation', 'extent',",
        f'])',
        f'{var} = pd.DataFrame(_props)',
        f'',
        f'# Rename centroid columns for clarity',
        f'_rename = {{c: c.replace("centroid-0", "centroid_y").replace("centroid-1", "centroid_x")',
        f'           for c in {var}.columns}}',
        f'{var} = {var}.rename(columns=_rename)',
        f'',
        f'# Derived measurements',
        f'if "perimeter" in {var}.columns and "area" in {var}.columns:',
        f'    {var}["circularity"] = 4 * np.pi * {var}["area"] / ({var}["perimeter"] ** 2 + 1e-9)',
        f'',
        f'print(f"  measured {{{var}.shape[0]}} objects")',
    ]
    return lines


def _gen_rgb_to_gray(var, props, inputs, imports):
    imports.add('from skimage.color import rgb2gray')
    in_var = inputs.get('image', 'image')
    return [
        f'# Convert RGB to grayscale',
        f'if {in_var}.ndim == 3:',
        f'    {var} = (rgb2gray({in_var}) * 255).astype(np.uint8)',
        f'else:',
        f'    {var} = {in_var}.copy()',
    ]


def _gen_split_rgb(var, props, inputs, imports):
    in_var = inputs.get('image', 'image')
    return [
        f'# Split into R, G, B channels',
        f'assert {in_var}.ndim == 3, "Input must be RGB"',
        f'{var}_red   = {in_var}[:, :, 0]',
        f'{var}_green = {in_var}[:, :, 1]',
        f'{var}_blue  = {in_var}[:, :, 2]',
    ]


def _gen_merge_rgb(var, props, inputs, imports):
    r = inputs.get('red', 'np.zeros_like(green)')
    g = inputs.get('green', 'np.zeros_like(red)')
    b = inputs.get('blue', 'np.zeros_like(red)')
    return [
        f'# Merge channels into RGB',
        f'{var} = np.stack([{r}, {g}, {b}], axis=-1)',
    ]


def _gen_data_summary(var, props, inputs, imports):
    imports.add('import pandas as pd')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    return [
        f'# Compute descriptive statistics',
        f'{var} = {in_var}.describe()',
        f'print({var}.to_string())',
    ]


def _gen_filter_table(var, props, inputs, imports):
    imports.add('import pandas as pd')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    expr = props.get('filter_expression', '')
    if expr:
        return [
            f'# Filter rows: {expr}',
            f'{var} = {in_var}.query({_repr_val(expr)})',
            f'print(f"  {{len({var})}} / {{len({in_var})}} rows passed filter")',
        ]
    return [
        f'{var} = {in_var}.copy()  # no filter expression set',
    ]


def _gen_math_column(var, props, inputs, imports):
    imports.add('import pandas as pd')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    col = props.get('column_name', 'result')
    expr = props.get('expression', '')
    lines = [
        f'# Add computed column: {col}',
        f'{var} = {in_var}.copy()',
    ]
    if expr:
        lines.append(f'{var}[{_repr_val(col)}] = {var}.eval({_repr_val(expr)})')
    else:
        lines.append(f'# No expression set for column {col}')
    return lines


def _gen_sort_table(var, props, inputs, imports):
    imports.add('import pandas as pd')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    col = props.get('sort_column', '')
    asc = props.get('ascending', True)
    if col:
        direction = 'ascending' if asc else 'descending'
        return [
            f'# Sort by {col} ({direction})',
            f'{var} = {in_var}.sort_values({_repr_val(col)}, ascending={asc}).reset_index(drop=True)',
        ]
    return [f'{var} = {in_var}.copy()  # no sort column set']


def _gen_aggregate_table(var, props, inputs, imports):
    imports.add('import pandas as pd')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    group_col = props.get('group_column', '')
    agg_func = props.get('agg_function', 'mean')
    if group_col:
        return [
            f'# Aggregate: group by {group_col}, apply {agg_func}',
            f'{var} = {in_var}.groupby({_repr_val(group_col)}).agg({_repr_val(agg_func)}).reset_index()',
        ]
    return [
        f'# Aggregate (no group column set)',
        f'{var} = {in_var}.agg("mean").to_frame().T',
    ]


def _gen_display(var, props, inputs, imports):
    in_var = list(inputs.values())[0] if inputs else 'data'
    return [
        f'# Display output',
        f'if hasattr({in_var}, "to_string"):',
        f'    print({in_var}.to_string())',
        f'elif isinstance({in_var}, np.ndarray):',
        f'    print(f"  array shape: {{{in_var}.shape}}, dtype: {{{in_var}.dtype}}")',
        f'else:',
        f'    print({in_var})',
    ]


def _gen_scatter_plot(var, props, inputs, imports):
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    x = props.get('x_column', '')
    y = props.get('y_column', '')
    hue = props.get('hue_column', '') or props.get('group_column', '')
    lines = [
        f'# Scatter plot: {x} vs {y}',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 6))',
    ]
    if hue:
        imports.add('import itertools')
        lines.extend([
            f'for _cat, _grp in {in_var}.groupby({_repr_val(hue)}):',
            f'    {var}_ax.scatter(_grp[{_repr_val(x)}], _grp[{_repr_val(y)}], label=str(_cat), alpha=0.7)',
            f'{var}_ax.legend()',
        ])
    else:
        lines.append(f'{var}_ax.scatter({in_var}[{_repr_val(x)}], {in_var}[{_repr_val(y)}], alpha=0.7)')
    lines.extend([
        f'{var}_ax.set_xlabel({_repr_val(x)})',
        f'{var}_ax.set_ylabel({_repr_val(y)})',
        f'{var}_ax.set_title({_repr_val(f"{y} vs {x}")})',
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


def _gen_histogram(var, props, inputs, imports):
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    col = props.get('column', '')
    bins = props.get('bins', 30)
    lines = [
        f'# Histogram: {col}',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 5))',
    ]
    if col:
        lines.extend([
            f'{var}_ax.hist({in_var}[{_repr_val(col)}], bins={bins}, edgecolor="black", alpha=0.7)',
            f'{var}_ax.set_xlabel({_repr_val(col)})',
            f'{var}_ax.set_ylabel("Count")',
            f'{var}_ax.set_title(f"Distribution of {col}")',
        ])
    else:
        lines.append(f'# Set column name in properties')
    lines.extend([
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


def _gen_box_plot(var, props, inputs, imports):
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    x = props.get('x_column', '') or props.get('group_column', '')
    y = props.get('y_column', '') or props.get('value_column', '')
    lines = [
        f'# Box plot: {y} by {x}',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 6))',
    ]
    if x and y:
        lines.extend([
            f'_groups = [grp[{_repr_val(y)}].dropna().values for _, grp in {in_var}.groupby({_repr_val(x)})]',
            f'_labels = [str(name) for name, _ in {in_var}.groupby({_repr_val(x)})]',
            f'{var}_ax.boxplot(_groups, labels=_labels)',
            f'{var}_ax.set_xlabel({_repr_val(x)})',
            f'{var}_ax.set_ylabel({_repr_val(y)})',
        ])
    else:
        lines.append('# Set x_column and y_column')
    lines.extend([
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


def _gen_violin_plot(var, props, inputs, imports):
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    x = props.get('x_column', '') or props.get('group_column', '')
    y = props.get('y_column', '') or props.get('value_column', '')
    lines = [
        f'# Violin plot: {y} by {x}',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 6))',
    ]
    if x and y:
        lines.extend([
            f'_groups = [grp[{_repr_val(y)}].dropna().values for _, grp in {in_var}.groupby({_repr_val(x)})]',
            f'_labels = [str(name) for name, _ in {in_var}.groupby({_repr_val(x)})]',
            f'{var}_ax.violinplot(_groups, showmeans=True, showmedians=True)',
            f'{var}_ax.set_xticks(range(1, len(_labels) + 1))',
            f'{var}_ax.set_xticklabels(_labels)',
            f'{var}_ax.set_xlabel({_repr_val(x)})',
            f'{var}_ax.set_ylabel({_repr_val(y)})',
        ])
    else:
        lines.append('# Set x_column and y_column')
    lines.extend([
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


def _gen_bar_plot(var, props, inputs, imports):
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    x = props.get('x_column', '') or props.get('group_column', '')
    y = props.get('y_column', '') or props.get('value_column', '')
    lines = [
        f'# Bar plot: {y} by {x}',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 6))',
    ]
    if x and y:
        lines.extend([
            f'_summary = {in_var}.groupby({_repr_val(x)})[{_repr_val(y)}].mean()',
            f'{var}_ax.bar(_summary.index.astype(str), _summary.values)',
            f'{var}_ax.set_xlabel({_repr_val(x)})',
            f'{var}_ax.set_ylabel({_repr_val(y)})',
        ])
    else:
        lines.append('# Set x_column and y_column')
    lines.extend([
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


def _gen_heatmap(var, props, inputs, imports):
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    lines = [
        f'# Heatmap of correlation matrix',
        f'_numeric = {in_var}.select_dtypes(include=[np.number])',
        f'_corr = _numeric.corr()',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(10, 8))',
        f'_im = {var}_ax.imshow(_corr, cmap="coolwarm", vmin=-1, vmax=1)',
        f'{var}_ax.set_xticks(range(len(_corr.columns)))',
        f'{var}_ax.set_yticks(range(len(_corr.columns)))',
        f'{var}_ax.set_xticklabels(_corr.columns, rotation=45, ha="right")',
        f'{var}_ax.set_yticklabels(_corr.columns)',
        f'plt.colorbar(_im, ax={var}_ax)',
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ]
    return lines


def _gen_xy_line_plot(var, props, inputs, imports):
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    x = props.get('x_column', '')
    y = props.get('y_column', '')
    lines = [
        f'# XY Line plot: {y} vs {x}',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 5))',
    ]
    if x and y:
        lines.extend([
            f'{var}_ax.plot({in_var}[{_repr_val(x)}], {in_var}[{_repr_val(y)}], marker="o", markersize=3)',
            f'{var}_ax.set_xlabel({_repr_val(x)})',
            f'{var}_ax.set_ylabel({_repr_val(y)})',
        ])
    else:
        lines.append('# Set x_column and y_column')
    lines.extend([
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


def _gen_crop(var, props, inputs, imports):
    in_var = inputs.get('image', 'image')
    return [
        f'# CropNode — requires ROI coordinates; adjust slicing manually',
        f'# Example: {var} = {in_var}[y1:y2, x1:x2]',
        f'{var} = {in_var}  # TODO: add crop coordinates',
    ]


def _gen_rotate(var, props, inputs, imports):
    imports.add('from scipy.ndimage import rotate as ndi_rotate')
    in_var = inputs.get('image', 'image')
    angle = props.get('angle', 0)
    return [
        f'# Rotate image by {angle} degrees',
        f'{var} = ndi_rotate({in_var}, angle={angle}, reshape=False)',
    ]


def _gen_resize(var, props, inputs, imports):
    imports.add('from skimage.transform import resize')
    in_var = inputs.get('image', 'image')
    w = props.get('width', 256)
    h = props.get('height', 256)
    return [
        f'# Resize to {w}x{h}',
        f'{var} = resize({in_var}, ({h}, {w}), preserve_range=True).astype({in_var}.dtype)',
    ]


def _gen_multi_otsu(var, props, inputs, imports):
    imports.add('from skimage.filters import threshold_multiotsu')
    in_var = inputs.get('image', 'image')
    n = props.get('n_classes', 3)
    return [
        f'# Multi-Otsu threshold into {n} classes',
        f'_gray = to_gray({in_var})',
        f'_thresholds = threshold_multiotsu(_gray, classes={n})',
        f'{var} = np.digitize(_gray, bins=_thresholds).astype(np.int32)',
        f'print(f"  thresholds: {{[f\\"{{t:.1f}}\\" for t in _thresholds]}}")',
        f'print(f"  {n} classes, label range: 0..{{{var}.max()}}")',
    ]


def _gen_equalize_adapthist(var, props, inputs, imports):
    imports.add('from skimage.exposure import equalize_adapthist')
    in_var = inputs.get('image', 'image')
    clip = props.get('clip_limit', 0.01)
    return [
        f'# Adaptive histogram equalization (CLAHE, clip={clip})',
        f'{var} = (equalize_adapthist({in_var}, clip_limit={clip}) * 255).astype(np.uint8)',
    ]


def _gen_brightness_contrast(var, props, inputs, imports):
    in_var = inputs.get('image', 'image')
    brightness = props.get('brightness', 0)
    contrast = props.get('contrast', 1.0)
    return [
        f'# Adjust brightness ({brightness:+}) and contrast (x{contrast})',
        f'{var} = np.clip({in_var}.astype(np.float32) * {contrast} + {brightness}, 0, 255).astype(np.uint8)',
    ]


def _gen_image_math(var, props, inputs, imports):
    a_var = inputs.get('image_a', inputs.get('A', 'image_a'))
    b_var = inputs.get('image_b', inputs.get('B', 'image_b'))
    op = props.get('operation', 'Add')
    lines = [f'# Image math: {op}']
    ops = {
        'Add': f'{var} = np.clip({a_var}.astype(np.int16) + {b_var}.astype(np.int16), 0, 255).astype(np.uint8)',
        'Subtract': f'{var} = np.clip({a_var}.astype(np.int16) - {b_var}.astype(np.int16), 0, 255).astype(np.uint8)',
        'Multiply': f'{var} = np.clip({a_var}.astype(np.float32) * {b_var}.astype(np.float32) / 255, 0, 255).astype(np.uint8)',
        'Divide': f'{var} = np.clip({a_var}.astype(np.float32) / np.maximum({b_var}.astype(np.float32), 1), 0, 255).astype(np.uint8)',
        'Max': f'{var} = np.maximum({a_var}, {b_var})',
        'Min': f'{var} = np.minimum({a_var}, {b_var})',
    }
    lines.append(ops.get(op, ops['Add']))
    return lines


def _gen_local_threshold(var, props, inputs, imports):
    imports.add('from skimage.filters import threshold_local')
    in_var = inputs.get('image', 'image')
    block = props.get('block_size', 35)
    offset = props.get('offset', 0)
    return [
        f'# Local (adaptive) threshold (block_size={block}, offset={offset})',
        f'_gray = to_gray({in_var})',
        f'_thresh = threshold_local(_gray, block_size={block}, offset={offset})',
        f'{var} = (_gray > _thresh).astype(np.uint8) * 255',
    ]


def _gen_gamma_contrast(var, props, inputs, imports):
    imports.add('from skimage.exposure import adjust_gamma')
    in_var = inputs.get('image', 'image')
    gamma = props.get('gamma', 1.0)
    return [
        f'# Gamma correction (gamma={gamma})',
        f'{var} = adjust_gamma({in_var}, gamma={gamma})',
    ]


def _gen_rolling_ball(var, props, inputs, imports):
    imports.add('from skimage.restoration import rolling_ball')
    in_var = inputs.get('image', 'image')
    radius = props.get('radius', 50)
    return [
        f'# Rolling ball background subtraction (radius={radius})',
        f'_bg = rolling_ball({in_var}, radius={radius})',
        f'{var} = np.clip({in_var}.astype(np.float32) - _bg, 0, 255).astype(np.uint8)',
    ]


def _gen_mirror(var, props, inputs, imports):
    in_var = inputs.get('image', 'image')
    axis = props.get('axis', 'Horizontal')
    ax = 1 if axis == 'Horizontal' else 0
    return [
        f'# Mirror image ({axis.lower()})',
        f'{var} = np.flip({in_var}, axis={ax}).copy()',
    ]


def _gen_zoom(var, props, inputs, imports):
    imports.add('from scipy.ndimage import zoom as ndi_zoom')
    in_var = inputs.get('image', 'image')
    factor = props.get('zoom_factor', 2.0)
    return [
        f'# Zoom (factor={factor}x)',
        f'_zoom_factors = [{factor}, {factor}] + ([1] if {in_var}.ndim == 3 else [])',
        f'{var} = ndi_zoom({in_var}, _zoom_factors, order=1)',
    ]


def _gen_canny_edge(var, props, inputs, imports):
    imports.add('from skimage.feature import canny')
    in_var = inputs.get('image', 'image')
    sigma = props.get('sigma', 1.0)
    return [
        f'# Canny edge detection (sigma={sigma})',
        f'_gray = to_gray({in_var})',
        f'{var} = canny(_gray, sigma={sigma}).astype(np.uint8) * 255',
    ]


def _gen_sobel_edge(var, props, inputs, imports):
    imports.add('from skimage.filters import sobel')
    in_var = inputs.get('image', 'image')
    return [
        f'# Sobel edge detection',
        f'_gray = to_gray({in_var})',
        f'{var} = to_uint8(sobel(_gray))',
    ]


def _gen_prewitt_edge(var, props, inputs, imports):
    imports.add('from skimage.filters import prewitt')
    in_var = inputs.get('image', 'image')
    return [
        f'# Prewitt edge detection',
        f'_gray = to_gray({in_var})',
        f'{var} = to_uint8(prewitt(_gray))',
    ]


def _gen_laplacian_edge(var, props, inputs, imports):
    imports.add('from skimage.filters import laplace')
    in_var = inputs.get('image', 'image')
    return [
        f'# Laplacian edge detection',
        f'_gray = to_gray({in_var})',
        f'{var} = to_uint8(np.abs(laplace(_gray)))',
    ]


def _gen_combine_tables(var, props, inputs, imports):
    imports.add('import pandas as pd')
    parts = list(inputs.values())
    if len(parts) >= 2:
        return [
            f'# Combine {len(parts)} tables',
            f'{var} = pd.concat([{", ".join(parts)}], ignore_index=True)',
            f'print(f"  combined: {{len({var})}} total rows")',
        ]
    elif parts:
        return [f'{var} = {parts[0]}.copy()']
    return [f'{var} = pd.DataFrame()  # no inputs connected']


def _gen_select_columns(var, props, inputs, imports):
    imports.add('import pandas as pd')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    cols = props.get('columns', '')
    if cols:
        col_list = [c.strip() for c in cols.split(',') if c.strip()]
        return [
            f'# Select columns: {", ".join(col_list)}',
            f'{var} = {in_var}[{repr(col_list)}].copy()',
        ]
    return [f'{var} = {in_var}.copy()  # no columns specified']


def _gen_outlier_detection(var, props, inputs, imports):
    imports.add('import pandas as pd')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    method = props.get('method', 'IQR')
    lines = [
        f'# Outlier detection ({method})',
        f'{var} = {in_var}.copy()',
    ]
    if method == 'IQR':
        lines.extend([
            f'_numeric = {var}.select_dtypes(include=[np.number])',
            f'_Q1 = _numeric.quantile(0.25)',
            f'_Q3 = _numeric.quantile(0.75)',
            f'_IQR = _Q3 - _Q1',
            f'_mask = ~((_numeric < (_Q1 - 1.5 * _IQR)) | (_numeric > (_Q3 + 1.5 * _IQR))).any(axis=1)',
            f'{var} = {var}[_mask].reset_index(drop=True)',
            f'print(f"  kept {{len({var})}} / {{len({in_var})}} rows")',
        ])
    else:
        lines.append(f'# Method "{method}" — implement manually')
    return lines


def _gen_normality_test(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('from scipy import stats')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    return [
        f'# Normality test (Shapiro-Wilk)',
        f'_results = []',
        f'for _col in {in_var}.select_dtypes(include=[np.number]).columns:',
        f'    _data = {in_var}[_col].dropna()',
        f'    if len(_data) >= 3:',
        f'        _stat, _p = stats.shapiro(_data)',
        f'        _results.append({{"column": _col, "W_statistic": _stat, "p_value": _p, "normal": _p > 0.05}})',
        f'{var} = pd.DataFrame(_results)',
        f'print({var}.to_string())',
    ]


def _gen_pca(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('from sklearn.decomposition import PCA')
    imports.add('from sklearn.preprocessing import StandardScaler')
    in_var = inputs.get('table', inputs.get('in', 'df'))
    n = props.get('n_components', 2)
    return [
        f'# PCA (n_components={n})',
        f'_numeric = {in_var}.select_dtypes(include=[np.number]).dropna()',
        f'_scaled = StandardScaler().fit_transform(_numeric)',
        f'_pca = PCA(n_components={n})',
        f'_transformed = _pca.fit_transform(_scaled)',
        f'{var} = pd.DataFrame(_transformed, columns=[f"PC{{i+1}}" for i in range({n})])',
        f'print(f"  explained variance ratio: {{_pca.explained_variance_ratio_}}")',
    ]


def _gen_white_top_hat(var, props, inputs, imports):
    imports.add('from skimage.morphology import white_tophat, disk')
    in_var = inputs.get('image', 'image')
    r = props.get('radius', 10)
    return [
        f'# White top-hat filter (radius={r})',
        f'_gray = to_gray({in_var})',
        f'{var} = to_uint8(white_tophat(_gray, disk({r})))',
    ]


def _gen_black_top_hat(var, props, inputs, imports):
    imports.add('from skimage.morphology import black_tophat, disk')
    in_var = inputs.get('image', 'image')
    r = props.get('radius', 10)
    return [
        f'# Black top-hat filter (radius={r})',
        f'_gray = to_gray({in_var})',
        f'{var} = to_uint8(black_tophat(_gray, disk({r})))',
    ]


def _gen_frangi(var, props, inputs, imports):
    imports.add('from skimage.filters import frangi')
    in_var = inputs.get('image', 'image')
    return [
        f'# Frangi vesselness filter',
        f'_gray = to_gray({in_var})',
        f'{var} = to_uint8(frangi(_gray))',
    ]


def _gen_blob_detect(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('from skimage.feature import blob_log')
    in_var = inputs.get('image', 'image')
    return [
        f'# Blob detection (Laplacian of Gaussian)',
        f'_gray = to_gray({in_var})',
        f'_blobs = blob_log(_gray, max_sigma=30, threshold=0.1)',
        f'{var} = pd.DataFrame(_blobs, columns=["y", "x", "sigma"])',
        f'{var}["radius_px"] = {var}["sigma"] * np.sqrt(2)',
        f'print(f"  detected {{{var}.shape[0]}} blobs")',
    ]


def _gen_find_contours(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('from skimage.measure import find_contours')
    in_var = inputs.get('mask', inputs.get('image', 'mask'))
    level = props.get('level', 0.5)
    return [
        f'# Find contours (level={level})',
        f'_gray = to_gray({in_var}) if {in_var}.ndim == 3 else {in_var}.astype(np.float64)',
        f'_contours = find_contours(_gray, level={level})',
        f'print(f"  found {{len(_contours)}} contours")',
        f'{var} = _contours  # list of (N, 2) arrays',
    ]


def _gen_folder_iterator(var, props, inputs, imports):
    imports.add('from pathlib import Path')
    fp = props.get('folder_path', '')
    pattern = props.get('pattern', '*.*')
    return [
        f'# Load file list from folder',
        f'_folder = Path({var.upper()}_FOLDER)',
        f'{var}_files = sorted(_folder.glob({_repr_val(pattern)}))',
        f'print(f"  found {{len({var}_files)}} files matching {_repr_val(pattern)}")',
        f'{var} = [str(f) for f in {var}_files]',
    ]


def _gen_grouped_comparison(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('from scipy.stats import f_oneway, kruskal')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    method = props.get('method', 'One-Way ANOVA')
    target = props.get('target_column', '')
    group = props.get('group_column', 'Group')
    is_anova = 'ANOVA' in method
    test_fn = 'f_oneway' if is_anova else 'kruskal'
    test_name = 'One-Way ANOVA' if is_anova else 'Kruskal-Wallis'
    stat_name = 'F-Statistic' if is_anova else 'H-Statistic'
    lines = [
        f'# {test_name}: compare groups',
        f'_group_col = {_repr_val(group)}',
        f'_target_col = {_repr_val(target)}',
        f'',
        f'# Auto-detect columns if not specified',
        f'if not _target_col or _target_col not in {in_var}.columns:',
        f'    _num_cols = {in_var}.select_dtypes(include=[np.number]).columns',
        f'    _target_col = [c for c in _num_cols if c != _group_col][0] if len(_num_cols) > 0 else None',
        f'',
        f'_df_clean = {in_var}[[_group_col, _target_col]].dropna()',
        f'_groups = [grp[_target_col].values for _, grp in _df_clean.groupby(_group_col)]',
        f'_stat, _p = {test_fn}(*_groups)',
        f'',
        f'{var} = pd.DataFrame([{{',
        f'    "Test": {_repr_val(test_name)},',
        f'    "Target": _target_col,',
        f'    {_repr_val(stat_name)}: round(_stat, 4),',
        f'    "p-value": round(_p, 6),',
        f'    "Significant": _p < 0.05,',
        f'}}])',
        f'print({var}.to_string(index=False))',
    ]
    return lines


def _gen_pairwise_comparison(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('from itertools import combinations')
    imports.add('from scipy.stats import ttest_ind, mannwhitneyu')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    method = props.get('method', "Welch's T-test")
    target = props.get('target_column', '')
    group = props.get('group_column', '')
    p_adj = props.get('p_adj_method', 'bonferroni')
    comp_type = props.get('comparison_type', 'All-vs-All')
    ref = props.get('reference_group', '')

    lines = [
        f'# Pairwise comparison ({method})',
        f'_group_col = {_repr_val(group)}',
        f'_target_col = {_repr_val(target)}',
        f'',
        f'# Auto-detect columns if not specified',
        f'if not _group_col or _group_col not in {in_var}.columns:',
        f'    for _c in {in_var}.columns:',
        f'        if _c.lower() in ("group", "class", "treatment"):',
        f'            _group_col = _c; break',
        f'if not _target_col or _target_col not in {in_var}.columns:',
        f'    _num = {in_var}.select_dtypes(include=[np.number]).columns',
        f'    _target_col = [c for c in _num if c != _group_col][0]',
        f'',
        f'_df = {in_var}[[_group_col, _target_col]].dropna()',
        f'_df[_group_col] = _df[_group_col].astype(str).str.strip()',
        f'_unique = _df[_group_col].unique()',
        f'',
    ]

    if 'Mann' in method:
        test_call = 'mannwhitneyu(_d1, _d2, alternative="two-sided")'
    else:
        equal_var = 'True' if "Student" in method else 'False'
        test_call = f'ttest_ind(_d1, _d2, equal_var={equal_var})'

    if comp_type == 'Reference-vs-All' and ref:
        lines.extend([
            f'# Reference-vs-All: compare each group to {_repr_val(ref)}',
            f'_ref_data = _df[_df[_group_col] == {_repr_val(ref)}][_target_col].values',
            f'_results = []',
            f'for _g in _unique:',
            f'    if _g == {_repr_val(ref)}: continue',
            f'    _d1 = _ref_data',
            f'    _d2 = _df[_df[_group_col] == _g][_target_col].values',
            f'    _stat, _p = {test_call}',
            f'    _results.append({{"Group A": {_repr_val(ref)}, "Group B": _g,',
            f'                     "Statistic": round(_stat, 4), "p-value": round(_p, 6)}})',
        ])
    else:
        lines.extend([
            f'# All-vs-All pairwise comparisons',
            f'_results = []',
            f'for _g1, _g2 in combinations(_unique, 2):',
            f'    _d1 = _df[_df[_group_col] == _g1][_target_col].values',
            f'    _d2 = _df[_df[_group_col] == _g2][_target_col].values',
            f'    _stat, _p = {test_call}',
            f'    _results.append({{"Group A": _g1, "Group B": _g2,',
            f'                     "Statistic": round(_stat, 4), "p-value": round(_p, 6)}})',
        ])

    lines.extend([
        f'',
        f'{var} = pd.DataFrame(_results)',
        f'',
    ])

    if p_adj != 'none':
        imports.add('from statsmodels.stats.multitest import multipletests')
        lines.extend([
            f'# P-value correction ({p_adj})',
            f'if len({var}) > 1:',
            f'    _reject, _padj, _, _ = multipletests({var}["p-value"], method={_repr_val(p_adj)})',
            f'    {var}["p-adjusted"] = _padj',
            f'    {var}["Significant"] = _reject',
            f'else:',
            f'    {var}["p-adjusted"] = {var}["p-value"]',
            f'    {var}["Significant"] = {var}["p-value"] < 0.05',
        ])
    else:
        lines.append(f'{var}["Significant"] = {var}["p-value"] < 0.05')

    lines.extend([
        f'',
        f'print(f"  {{len({var})}} pairwise comparisons:")',
        f'print({var}.to_string(index=False))',
    ])
    return lines


def _gen_group_normalization(var, props, inputs, imports):
    imports.add('import pandas as pd')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    control = props.get('control_group', '')
    target = props.get('target_column', 'Group')
    lines = [
        f'# Normalize to control group mean',
        f'_group_col = {_repr_val(target)}',
        f'_control = {_repr_val(control)}',
        f'',
        f'# Find group column',
        f'if _group_col not in {in_var}.columns:',
        f'    for _c in {in_var}.columns:',
        f'        if _c.lower() in ("group", "class", "treatment"):',
        f'            _group_col = _c; break',
        f'',
        f'{var} = {in_var}.copy()',
        f'_num_cols = {var}.select_dtypes(include=[np.number]).columns',
        f'_ctrl_rows = {var}[{var}[_group_col].astype(str) == _control]',
        f'for _col in _num_cols:',
        f'    _ctrl_mean = _ctrl_rows[_col].mean()',
        f'    if _ctrl_mean != 0:',
        f'        {var}[_col] = {var}[_col] / _ctrl_mean',
        f'print(f"  normalized {{len(_num_cols)}} columns to {{_control}} mean")',
    ]
    return lines


def _gen_linear_regression(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('import statsmodels.api as sm')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    x_cols = props.get('x_cols', '')
    y_col = props.get('y_col', '')
    intercept = props.get('intercept', True)
    lines = [
        f'# Linear regression (OLS)',
        f'_x_cols_raw = {_repr_val(x_cols)}',
        f'_y_col = {_repr_val(y_col)}',
        f'',
        f'_num_cols = {in_var}.select_dtypes(include=[np.number]).columns.tolist()',
        f'if not _y_col or _y_col not in {in_var}.columns:',
        f'    _y_col = _num_cols[-1]',
        f'if not _x_cols_raw:',
        f'    _x_cols = [c for c in _num_cols if c != _y_col][:1]',
        f'else:',
        f'    _x_cols = [c.strip() for c in _x_cols_raw.split(",") if c.strip() in {in_var}.columns]',
        f'',
        f'_df_c = {in_var}[_x_cols + [_y_col]].dropna()',
        f'_X = _df_c[_x_cols].astype(float)',
        f'_y = _df_c[_y_col].astype(float)',
    ]
    if intercept:
        lines.append(f'_X = sm.add_constant(_X, has_constant="add")')
    lines.extend([
        f'',
        f'_model = sm.OLS(_y, _X).fit()',
        f'',
        f'# Coefficients table',
        f'_ci = _model.conf_int(alpha=0.05)',
        f'{var}_coef = pd.DataFrame({{',
        f'    "Parameter": _model.params.index,',
        f'    "Coefficient": _model.params.round(6).values,',
        f'    "Std Error": _model.bse.round(6).values,',
        f'    "t-value": _model.tvalues.round(4).values,',
        f'    "p-value": _model.pvalues.round(6).values,',
        f'    "95% CI Lo": _ci.iloc[:, 0].round(6).values,',
        f'    "95% CI Hi": _ci.iloc[:, 1].round(6).values,',
        f'}})',
        f'',
        f'# Residuals',
        f'{var}_resid = _df_c.copy()',
        f'{var}_resid["Predicted"] = _model.fittedvalues.values',
        f'{var}_resid["Residual"] = _model.resid.values',
        f'',
        f'print(f"  R² = {{_model.rsquared:.4f}}, Adj R² = {{_model.rsquared_adj:.4f}}")',
        f'print(f"  F = {{_model.fvalue:.2f}}, p = {{_model.f_pvalue:.2e}}")',
        f'print({var}_coef.to_string(index=False))',
        f'{var} = {var}_coef  # coefficients table (residuals in {var}_resid)',
    ])
    return lines


def _gen_two_way_anova(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('import statsmodels.formula.api as smf')
    imports.add('from statsmodels.stats.anova import anova_lm')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    f1 = props.get('factor1', '')
    f2 = props.get('factor2', '')
    val = props.get('value_col', '')
    lines = [
        f'# Two-Way ANOVA with interaction',
        f'_factor1 = {_repr_val(f1)}',
        f'_factor2 = {_repr_val(f2)}',
        f'_value_col = {_repr_val(val)}',
        f'',
        f'# Auto-detect columns if not specified',
        f'_cat_cols = {in_var}.select_dtypes(exclude=[np.number]).columns.tolist()',
        f'_num_cols = {in_var}.select_dtypes(include=[np.number]).columns.tolist()',
        f'if not _factor1 or _factor1 not in {in_var}.columns:',
        f'    _factor1 = _cat_cols[0] if _cat_cols else _num_cols[0]',
        f'if not _factor2 or _factor2 not in {in_var}.columns:',
        f'    _factor2 = _cat_cols[1] if len(_cat_cols) > 1 else _num_cols[1]',
        f'if not _value_col or _value_col not in {in_var}.columns:',
        f'    _value_col = next((c for c in _num_cols if c not in (_factor1, _factor2)), None)',
        f'',
        f'_df_c = {in_var}[[_factor1, _factor2, _value_col]].dropna().copy()',
        f'_df_c.columns = ["_F1", "_F2", "_V"]',
        f'_df_c["_F1"] = _df_c["_F1"].astype(str)',
        f'_df_c["_F2"] = _df_c["_F2"].astype(str)',
        f'',
        f'_formula = "_V ~ C(_F1) + C(_F2) + C(_F1):C(_F2)"',
        f'_model = smf.ols(_formula, data=_df_c).fit()',
        f'_aov = anova_lm(_model, typ=2).reset_index()',
        f'_aov.columns = ["Source", "Sum of Squares", "df", "F", "p-value"]',
        f'_aov["Source"] = (_aov["Source"]',
        f'    .str.replace("C(_F1)", _factor1, regex=False)',
        f'    .str.replace("C(_F2)", _factor2, regex=False)',
        f'    .str.replace(":", " x ", regex=False))',
        f'_aov["Significant"] = _aov["p-value"] < 0.05',
        f'',
        f'# Group means summary',
        f'{var}_means = (_df_c.groupby(["_F1", "_F2"])["_V"]',
        f'    .agg(Mean="mean", SD="std", N="count")',
        f'    .reset_index())',
        f'{var}_means.columns = [_factor1, _factor2, "Mean", "SD", "N"]',
        f'{var}_means["SEM"] = {var}_means["SD"] / np.sqrt({var}_means["N"])',
        f'',
        f'{var} = _aov.round(6)',
        f'print({var}.to_string(index=False))',
    ]
    return lines


def _gen_contingency(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('from scipy.stats import chi2_contingency, fisher_exact')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    col1 = props.get('col1', '')
    col2 = props.get('col2', '')
    lines = [
        f'# Contingency analysis (Chi-square / Fisher\'s exact)',
        f'_col1 = {_repr_val(col1)}',
        f'_col2 = {_repr_val(col2)}',
        f'',
        f'# Auto-detect categorical columns if not specified',
        f'_cat = {in_var}.select_dtypes(exclude=[np.number]).columns.tolist()',
        f'if not _col1 or _col1 not in {in_var}.columns:',
        f'    _col1 = _cat[0] if _cat else {in_var}.columns[0]',
        f'if not _col2 or _col2 not in {in_var}.columns:',
        f'    _col2 = _cat[1] if len(_cat) > 1 else {in_var}.columns[1]',
        f'',
        f'_ct = pd.crosstab({in_var}[_col1], {in_var}[_col2])',
        f'_chi2, _p, _dof, _expected = chi2_contingency(_ct)',
        f'',
        f'{var} = pd.DataFrame([{{',
        f'    "Test": "Chi-square",',
        f'    "Chi2": round(_chi2, 4),',
        f'    "df": _dof,',
        f'    "p-value": round(_p, 6),',
        f'    "Significant": _p < 0.05,',
        f'}}])',
        f'',
        f'# Fisher\'s exact test (2x2 only)',
        f'if _ct.shape == (2, 2):',
        f'    _odds, _p_fisher = fisher_exact(_ct)',
        f'    {var} = pd.concat([{var}, pd.DataFrame([{{',
        f'        "Test": "Fisher exact", "Chi2": np.nan,',
        f'        "df": np.nan, "p-value": round(_p_fisher, 6),',
        f'        "Significant": _p_fisher < 0.05,',
        f'    }}])], ignore_index=True)',
        f'',
        f'print(f"  Observed counts:")',
        f'print(_ct.to_string())',
        f'print(f"\\n  Test results:")',
        f'print({var}.to_string(index=False))',
    ]
    return lines


def _gen_pairwise_matrix(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    method = props.get('method', 'pearson')
    cmap = props.get('colormap', 'coolwarm')
    lines = [
        f'# Pairwise correlation/distance matrix ({method})',
        f'_numeric = {in_var}.select_dtypes(include=[np.number])',
    ]
    if 'distance' in method:
        actual = method.split(' ')[0]
        imports.add('from scipy.spatial.distance import pdist, squareform')
        lines.extend([
            f'_dist = pdist(_numeric.T.values, metric={_repr_val(actual)})',
            f'{var} = pd.DataFrame(squareform(_dist),',
            f'                     index=_numeric.columns, columns=_numeric.columns).round(3)',
        ])
    else:
        lines.append(f'{var} = _numeric.corr(method={_repr_val(method)}).round(3)')
    lines.extend([
        f'',
        f'# Heatmap visualization',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(10, 8))',
        f'_im = {var}_ax.imshow({var}.values, cmap={_repr_val(cmap)}, aspect="auto")',
        f'{var}_ax.set_xticks(range(len({var}.columns)))',
        f'{var}_ax.set_yticks(range(len({var}.index)))',
        f'{var}_ax.set_xticklabels({var}.columns, rotation=45, ha="right")',
        f'{var}_ax.set_yticklabels({var}.index)',
        f'plt.colorbar(_im, ax={var}_ax)',
        f'plt.tight_layout()',
        f'',
        f'print({var}.to_string())',
    ])
    return lines


def _gen_swarm_plot(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    x = props.get('x_column', '') or props.get('group_column', '')
    y = props.get('y_column', '') or props.get('value_column', '')
    lines = [
        f'# Swarm / strip plot: {y} by {x}',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 6))',
    ]
    if x and y:
        lines.extend([
            f'_categories = {in_var}[{_repr_val(x)}].astype(str).unique()',
            f'for _i, _cat in enumerate(_categories):',
            f'    _vals = {in_var}[{in_var}[{_repr_val(x)}].astype(str) == _cat][{_repr_val(y)}].dropna()',
            f'    _jitter = np.random.uniform(-0.2, 0.2, size=len(_vals))',
            f'    {var}_ax.scatter(_i + _jitter, _vals, alpha=0.6, s=20)',
            f'{var}_ax.set_xticks(range(len(_categories)))',
            f'{var}_ax.set_xticklabels(_categories)',
            f'{var}_ax.set_xlabel({_repr_val(x)})',
            f'{var}_ax.set_ylabel({_repr_val(y)})',
        ])
    else:
        lines.append('# Set x_column (group) and y_column (value)')
    lines.extend([
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


def _gen_kde_plot(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('import matplotlib.pyplot as plt')
    imports.add('from scipy.stats import gaussian_kde')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    col = props.get('column', '') or props.get('value_column', '')
    lines = [
        f'# KDE density plot',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 5))',
    ]
    if col:
        lines.extend([
            f'_data = {in_var}[{_repr_val(col)}].dropna().values',
            f'_kde = gaussian_kde(_data)',
            f'_x = np.linspace(_data.min(), _data.max(), 300)',
            f'{var}_ax.plot(_x, _kde(_x))',
            f'{var}_ax.fill_between(_x, _kde(_x), alpha=0.3)',
            f'{var}_ax.set_xlabel({_repr_val(col)})',
            f'{var}_ax.set_ylabel("Density")',
        ])
    else:
        lines.append('# Set column property')
    lines.extend([
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


def _gen_regression_plot(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    x = props.get('x_column', '')
    y = props.get('y_column', '')
    lines = [
        f'# Regression plot: {y} vs {x}',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 6))',
    ]
    if x and y:
        lines.extend([
            f'_x = {in_var}[{_repr_val(x)}].astype(float)',
            f'_y = {in_var}[{_repr_val(y)}].astype(float)',
            f'_mask = ~(np.isnan(_x) | np.isnan(_y))',
            f'_x, _y = _x[_mask], _y[_mask]',
            f'',
            f'# Scatter + regression line',
            f'{var}_ax.scatter(_x, _y, alpha=0.5)',
            f'_coeffs = np.polyfit(_x, _y, 1)',
            f'_fit_x = np.linspace(_x.min(), _x.max(), 100)',
            f'{var}_ax.plot(_fit_x, np.polyval(_coeffs, _fit_x), "r-", lw=2,',
            f'             label=f"y = {{_coeffs[0]:.3f}}x + {{_coeffs[1]:.3f}}")',
            f'_r2 = 1 - np.sum((_y - np.polyval(_coeffs, _x))**2) / np.sum((_y - _y.mean())**2)',
            f'{var}_ax.set_title(f"R² = {{_r2:.4f}}")',
            f'{var}_ax.set_xlabel({_repr_val(x)})',
            f'{var}_ax.set_ylabel({_repr_val(y)})',
            f'{var}_ax.legend()',
        ])
    else:
        lines.append('# Set x_column and y_column')
    lines.extend([
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


def _gen_volcano_plot(var, props, inputs, imports):
    imports.add('import pandas as pd')
    imports.add('import matplotlib.pyplot as plt')
    in_var = inputs.get('in', inputs.get('table', 'df'))
    fc_col = props.get('fc_column', '') or props.get('log2fc_column', '')
    p_col = props.get('p_column', '') or props.get('pvalue_column', '')
    fc_thresh = props.get('fc_threshold', 1.0)
    p_thresh = props.get('p_threshold', 0.05)
    lines = [
        f'# Volcano plot',
        f'{var}_fig, {var}_ax = plt.subplots(figsize=(8, 6))',
    ]
    if fc_col and p_col:
        lines.extend([
            f'_fc = {in_var}[{_repr_val(fc_col)}].astype(float)',
            f'_pv = {in_var}[{_repr_val(p_col)}].astype(float)',
            f'_logp = -np.log10(_pv + 1e-300)',
            f'',
            f'# Color by significance',
            f'_sig_up   = (_fc >  {fc_thresh}) & (_pv < {p_thresh})',
            f'_sig_down = (_fc < -{fc_thresh}) & (_pv < {p_thresh})',
            f'_ns = ~(_sig_up | _sig_down)',
            f'{var}_ax.scatter(_fc[_ns],       _logp[_ns],       c="gray",  alpha=0.4, s=10, label="NS")',
            f'{var}_ax.scatter(_fc[_sig_up],   _logp[_sig_up],   c="red",   alpha=0.6, s=15, label="Up")',
            f'{var}_ax.scatter(_fc[_sig_down], _logp[_sig_down], c="blue",  alpha=0.6, s=15, label="Down")',
            f'{var}_ax.axhline(-np.log10({p_thresh}), ls="--", c="gray", lw=0.8)',
            f'{var}_ax.axvline( {fc_thresh}, ls="--", c="gray", lw=0.8)',
            f'{var}_ax.axvline(-{fc_thresh}, ls="--", c="gray", lw=0.8)',
            f'{var}_ax.set_xlabel("log2 Fold Change")',
            f'{var}_ax.set_ylabel("-log10(p-value)")',
            f'{var}_ax.legend()',
        ])
    else:
        lines.append('# Set fc_column and p_column')
    lines.extend([
        f'plt.tight_layout()',
        f'{var} = {var}_fig',
    ])
    return lines


# ── Generator dispatch table ─────────────────────────────────────────────────

_GENERATORS = {
    # I/O
    'FileReadNode': _gen_file_read,
    'ImageReadNode': _gen_image_read,
    'SaveNode': _gen_save,
    'FolderIteratorNode': _gen_folder_iterator,

    # Image processing
    'GaussianBlurNode': _gen_gaussian_blur,
    'BinaryThresholdNode': _gen_binary_threshold,
    'RemoveSmallObjectsNode': _gen_remove_small_objects,
    'RemoveSmallHolesNode': _gen_remove_small_holes,
    'ErosionNode': _gen_erosion,
    'DilationNode': _gen_dilation,
    'MorphOpenNode': _gen_morph_open,
    'MorphCloseNode': _gen_morph_close,
    'FillHolesNode': _gen_fill_holes,
    'SkeletonizeNode': _gen_skeletonize,
    'WatershedNode': _gen_watershed,
    'ParticlePropsNode': _gen_particle_props,
    'RGBToGrayNode': _gen_rgb_to_gray,
    'SplitRGBNode': _gen_split_rgb,
    'MergeRGBNode': _gen_merge_rgb,
    'MultiOtsuNode': _gen_multi_otsu,
    'EqualizeAdapthistNode': _gen_equalize_adapthist,
    'BrightnessContrastNode': _gen_brightness_contrast,
    'ImageMathNode': _gen_image_math,
    'ThresholdLocalNode': _gen_local_threshold,
    'GammaContrastNode': _gen_gamma_contrast,
    'RollingBallNode': _gen_rolling_ball,
    'MirrorNode': _gen_mirror,
    'ZoomNode': _gen_zoom,
    'ResizeNode': _gen_resize,
    'RotateNode': _gen_rotate,
    'CropNode': _gen_crop,
    'CannyEdgeNode': _gen_canny_edge,
    'SobelEdgeNode': _gen_sobel_edge,
    'PrewittEdgeNode': _gen_prewitt_edge,
    'LaplacianEdgeNode': _gen_laplacian_edge,
    'WhiteTopHatNode': _gen_white_top_hat,
    'BlackTopHatNode': _gen_black_top_hat,
    'FrangiNode': _gen_frangi,
    'BlobDetectNode': _gen_blob_detect,
    'FindContoursNode': _gen_find_contours,

    # Table processing
    'DataSummaryNode': _gen_data_summary,
    'FilterTableNode': _gen_filter_table,
    'MathColumnNode': _gen_math_column,
    'SortTableNode': _gen_sort_table,
    'AggregateTableNode': _gen_aggregate_table,
    'CombineTablesNode': _gen_combine_tables,
    'SelectColumnsNode': _gen_select_columns,
    'OutlierDetectionNode': _gen_outlier_detection,
    'NormalityTestNode': _gen_normality_test,
    'PCANode': _gen_pca,

    # Analysis / Statistics
    'GroupedComparisonNode': _gen_grouped_comparison,
    'PairwiseComparisonNode': _gen_pairwise_comparison,
    'GroupNormalizationNode': _gen_group_normalization,
    'LinearRegressionNode': _gen_linear_regression,
    'TwoWayANOVANode': _gen_two_way_anova,
    'ContingencyAnalysisNode': _gen_contingency,
    'PairwiseMatrixNode': _gen_pairwise_matrix,

    # Plotting
    'ScatterPlotNode': _gen_scatter_plot,
    'HistogramNode': _gen_histogram,
    'BoxPlotNode': _gen_box_plot,
    'ViolinPlotNode': _gen_violin_plot,
    'BarPlotNode': _gen_bar_plot,
    'HeatmapNode': _gen_heatmap,
    'XYLinePlotNode': _gen_xy_line_plot,
    'SwarmPlotNode': _gen_swarm_plot,
    'KdePlotNode': _gen_kde_plot,
    'RegressionPlotNode': _gen_regression_plot,
    'VolcanoPlotNode': _gen_volcano_plot,

    # Display
    'DisplayNode': _gen_display,
}
