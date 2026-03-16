#!/usr/bin/env python3
"""Auto-generate node reference Markdown pages from source docstrings.

Run:  python generate_docs.py
Outputs to docs/nodes/ and docs/plugins/
"""

import ast
import os
import re
import textwrap

# Project root is one level up from synapse/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")


def _extract_nodes(filepath):
    """Parse a Python file and extract node class info via AST."""
    with open(filepath, encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    nodes = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue

        info = {
            "class_name": cls.name,
            "node_name": None,
            "identifier": None,
            "docstring": None,
            "inputs": [],   # list of (name, type)
            "outputs": [],  # list of (name, type)
            "properties": [],
        }

        # Docstring
        ds = ast.get_docstring(cls)
        if ds:
            # Strip keyword lines and clean up
            lines = ds.strip().split("\n")
            clean = []
            for line in lines:
                stripped = line.strip()
                if stripped.lower().startswith("keywords:"):
                    continue
                if stripped.lower().startswith("keyword:"):
                    continue
                clean.append(line)
            info["docstring"] = textwrap.dedent("\n".join(clean)).strip()

        # Walk class body for assignments and method calls
        for stmt in cls.body:
            # NODE_NAME = '...'
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "NODE_NAME" and isinstance(
                            stmt.value, (ast.Constant,)
                        ):
                            info["node_name"] = stmt.value.value
                        elif target.id == "__identifier__" and isinstance(
                            stmt.value, (ast.Constant,)
                        ):
                            info["identifier"] = stmt.value.value
                        elif target.id == "PORT_SPEC" and isinstance(
                            stmt.value, ast.Dict
                        ):
                            for key, val in zip(stmt.value.keys, stmt.value.values):
                                if isinstance(key, ast.Constant):
                                    if isinstance(val, ast.List):
                                        names = [
                                            e.value
                                            for e in val.elts
                                            if isinstance(e, ast.Constant)
                                        ]
                                        # PORT_SPEC names are also the type
                                        if key.value == "inputs":
                                            info["inputs"] = [(n, n) for n in names]
                                        elif key.value == "outputs":
                                            info["outputs"] = [(n, n) for n in names]

            # __init__ method — look for add_input, add_output, _add_*_spinbox, etc.
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                for node in ast.walk(stmt):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    fname = None
                    if isinstance(func, ast.Attribute):
                        fname = func.attr
                    elif isinstance(func, ast.Name):
                        fname = func.id

                    if fname in ("add_input", "add_output") and node.args:
                        arg = node.args[0]
                        if isinstance(arg, ast.Constant):
                            port_name = arg.value
                            # Extract type from color=PORT_COLORS['type']
                            # or color=PORT_COLORS.get('type', ...)
                            port_type = port_name  # default: name is the type
                            for kw in node.keywords:
                                if kw.arg == "color":
                                    if isinstance(kw.value, ast.Subscript):
                                        # PORT_COLORS['table'] → 'table'
                                        sl = kw.value.slice
                                        if isinstance(sl, ast.Constant):
                                            port_type = sl.value
                                    elif isinstance(kw.value, ast.Call):
                                        # PORT_COLORS.get('table', ...) → 'table'
                                        if (kw.value.args
                                                and isinstance(kw.value.args[0], ast.Constant)):
                                            port_type = kw.value.args[0].value
                            target_list = info["inputs"] if fname == "add_input" else info["outputs"]
                            existing_names = [n for n, _ in target_list]
                            if port_name not in existing_names:
                                target_list.append((port_name, port_type))

                    elif fname and fname.startswith("_add_") and "spinbox" in fname:
                        if len(node.args) >= 2:
                            prop_name = (
                                node.args[1].value
                                if isinstance(node.args[1], ast.Constant)
                                else None
                            )
                            if prop_name:
                                info["properties"].append(prop_name)

                    elif fname == "add_checkbox":
                        # add_checkbox(prop_id, label, text=..., state=...)
                        for kw in node.keywords:
                            pass
                        if len(node.args) >= 3 and isinstance(
                            node.args[2], ast.Constant
                        ):
                            info["properties"].append(node.args[2].value)
                        elif len(node.args) >= 2 and isinstance(
                            node.args[1], ast.Constant
                        ):
                            info["properties"].append(node.args[1].value)

                    elif fname == "add_combo_menu":
                        if len(node.args) >= 2 and isinstance(
                            node.args[1], ast.Constant
                        ):
                            info["properties"].append(node.args[1].value)

        # If __init__ add_input/add_output calls found real port names (name != type),
        # those are authoritative — drop the PORT_SPEC placeholder entries (name == type).
        for direction in ("inputs", "outputs"):
            real = [(n, t) for n, t in info[direction] if n != t]
            if real:
                info[direction] = real

        if info["node_name"]:
            nodes.append(info)

    return nodes


def _node_to_md(info):
    """Convert a node info dict to a Markdown section."""
    lines = []
    name = info["node_name"]
    lines.append(f"### {name}")
    lines.append("")

    if info["docstring"]:
        # Use first paragraph only for the summary
        paragraphs = info["docstring"].split("\n\n")
        summary = paragraphs[0].replace("\n", " ").strip()
        lines.append(summary)
        lines.append("")

        # If there are more paragraphs, add them as details
        if len(paragraphs) > 1:
            rest = "\n\n".join(paragraphs[1:]).strip()
            if rest:
                lines.append("??? note \"Details\"")
                detail_lines = rest.split("\n")
                prev_was_bullet = False
                for i, rline in enumerate(detail_lines):
                    stripped = rline.strip()
                    is_bullet = stripped.startswith("- ") or stripped.startswith("* ")
                    is_empty = not stripped

                    # Convert standalone **param** — lines to bullet items
                    if (stripped.startswith("**") and "—" in stripped
                            and not stripped.startswith("- ")):
                        stripped = f"- {stripped}"
                        is_bullet = True

                    # Insert blank line before first bullet when preceded
                    # by non-bullet text (required by mkdocs admonitions)
                    if is_bullet and not prev_was_bullet and i > 0:
                        prev_line = detail_lines[i - 1].strip()
                        if prev_line and not prev_line.startswith("- ") and not prev_line.startswith("* "):
                            lines.append("")

                    if is_bullet:
                        lines.append(f"    {stripped}")
                    else:
                        lines.append(f"    {rline}")

                    if not is_empty:
                        prev_was_bullet = is_bullet
                lines.append("")

    # Ports table
    if info["inputs"] or info["outputs"]:
        lines.append("| Direction | Port | Type |")
        lines.append("|-----------|------|------|")
        for name, ptype in info["inputs"]:
            lines.append(f"| **Input** | `{name}` | {ptype} |")
        for name, ptype in info["outputs"]:
            lines.append(f"| **Output** | `{name}` | {ptype} |")
        lines.append("")

    # Properties
    if info["properties"]:
        lines.append("**Properties:** " + ", ".join(f"`{p}`" for p in info["properties"]))
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ── File → category mapping ──────────────────────────────────────────────

FILE_MAP = {
    # Core nodes (in synapse/nodes/)
    "synapse/nodes/dataframe_nodes.py": {
        "filter": ("nodes/data-tables/filter.md", "Filter"),
        "compute": ("nodes/data-tables/compute.md", "Compute"),
        "transform": ("nodes/data-tables/transform.md", "Transform"),
        "combine": ("nodes/data-tables/combine.md", "Combine"),
        "util": ("nodes/data-tables/util.md", "Utility"),
    },
    "synapse/nodes/io_nodes.py": {
        "_default": ("nodes/io-display.md", "IO & Display"),
    },
    "synapse/nodes/display_nodes.py": {
        "_default": ("nodes/io-display.md", "IO & Display"),
    },
    "synapse/nodes/utility_nodes.py": {
        "_default": ("nodes/io-display.md", "IO & Display"),
    },
    # ── image_analysis plugin ──────────────────────────────────────────────
    "plugins/image_analysis/image_process_nodes.py": {
        "color": ("plugins/image-analysis/color.md", "Color"),
        "exposure": ("plugins/image-analysis/exposure.md", "Exposure & Contrast"),
        "filter": ("plugins/image-analysis/filters.md", "Filters"),
        "transform": ("plugins/image-analysis/transform.md", "Transform"),
        "morphology": ("plugins/image-analysis/morphology.md", "Morphology"),
        "_default": ("plugins/image-analysis/filters.md", "Filters"),
    },
    "plugins/image_analysis/mask_nodes.py": {
        "_default": ("plugins/image-analysis/morphology.md", "Morphology"),
    },
    "plugins/image_analysis/vision_nodes.py": {
        "morphology": ("plugins/image-analysis/morphology.md", "Morphology"),
        "filter": ("plugins/image-analysis/filters.md", "Filters"),
        "measure": ("plugins/image-analysis/measurement.md", "Measurement & Analysis"),
        "_default": ("plugins/image-analysis/measurement.md", "Measurement & Analysis"),
    },
    "plugins/image_analysis/roi_nodes.py": {
        "_default": ("plugins/image-analysis/roi-drawing.md", "ROI & Drawing"),
    },
    # ── statistical_analysis plugin ────────────────────────────────────────
    "plugins/statistical_analysis/analysis_nodes.py": {
        "_default": ("plugins/statistical-analysis/descriptive.md", "Descriptive & Comparison"),
    },
    "plugins/statistical_analysis/stats_nodes.py": {
        "_default": ("plugins/statistical-analysis/regression.md", "Regression & Advanced"),
    },
    # ── figure_plotting plugin ─────────────────────────────────────────────
    "plugins/figure_plotting/plot_nodes.py": {
        "_default": ("plugins/figure-plotting.md", "Plotting"),
    },
    # ── filopodia plugin ───────────────────────────────────────────────────
    "plugins/filopodia_nodes/filopodia_nodes.py": {
        "_default": ("plugins/filopodia.md", "Filopodia Analysis"),
    },
    "plugins/filopodia_nodes/confocal_nodes.py": {
        "_default": ("plugins/filopodia.md", "Filopodia Analysis"),
    },
    # ── SAM2 / Cellpose / Tracking plugins ─────────────────────────────────
    "plugins/sam2_nodes/sam2_segment.py": {
        "_default": ("plugins/sam2.md", "SAM2 Segmentation"),
    },
    "plugins/sam2_nodes/node.py": {
        "_default": ("plugins/sam2.md", "SAM2 Segmentation"),
    },
    "plugins/sam2_nodes/tracking.py": {
        "_default": ("plugins/video-tracking.md", "Video & Tracking"),
    },
    "plugins/sam2_nodes/grounding.py": {
        "_default": ("plugins/sam2.md", "SAM2 Segmentation"),
    },
    "plugins/sam2_nodes/video_utils.py": {
        "_default": ("plugins/video-tracking.md", "Video & Tracking"),
    },
    "plugins/sam2_nodes/video_analyze_node.py": {
        "_default": ("plugins/video-tracking.md", "Video & Tracking"),
    },
    "plugins/sam2_nodes/particle_tracking.py": {
        "_default": ("plugins/video-tracking.md", "Video & Tracking"),
    },
    "plugins/sam2_nodes/cellpose_node.py": {
        "_default": ("plugins/cellpose.md", "Cellpose Segmentation"),
    },
    "plugins/sam2_nodes/cellpose_segment_node.py": {
        "_default": ("plugins/cellpose.md", "Cellpose Segmentation"),
    },
}

# Volume nodes
for vf in [
    "plugins/volume_nodes/io_nodes.py",
    "plugins/volume_nodes/viewer_nodes.py",
    "plugins/volume_nodes/image_ops_nodes.py",
    "plugins/volume_nodes/process_nodes.py",
    "plugins/volume_nodes/segment_nodes.py",
]:
    FILE_MAP[vf] = {"_default": ("plugins/volume.md", "3D Volume Processing")}

# Cheminformatics nodes
for rf in [
    "plugins/rdkit_nodes/chem_nodes.py",
    "plugins/rdkit_nodes/docking_nodes.py",
    "plugins/rdkit_nodes/viewer_nodes.py",
]:
    FILE_MAP[rf] = {"_default": ("plugins/cheminformatics.md", "Cheminformatics")}


def _get_category_key(identifier):
    """Extract the last meaningful part of the identifier for categorization.

    Case-insensitive: identifiers use mixed case (Exposure vs exposure,
    Transform vs transform) so we normalise to lower.
    """
    if not identifier:
        return "_default"
    parts = identifier.split(".")
    return parts[-1].lower() if parts else "_default"


def main():
    # Collect all node info grouped by output file
    output_files = {}  # path -> {title, nodes[]}
    core_count = 0
    plugin_count = 0

    for rel_path, cat_map in FILE_MAP.items():
        full_path = os.path.join(ROOT, rel_path)
        if not os.path.isfile(full_path):
            continue

        # Normalise cat_map keys to lower for matching
        cat_map_lower = {k.lower(): v for k, v in cat_map.items()}

        is_plugin = rel_path.startswith("plugins/")
        nodes = _extract_nodes(full_path)
        for info in nodes:
            cat_key = _get_category_key(info["identifier"])
            if cat_key in cat_map_lower:
                out_path, title = cat_map_lower[cat_key]
            elif "_default" in cat_map_lower:
                out_path, title = cat_map_lower["_default"]
            else:
                continue

            if out_path not in output_files:
                output_files[out_path] = {"title": title, "nodes": []}
            output_files[out_path]["nodes"].append(info)

            if is_plugin:
                plugin_count += 1
            else:
                core_count += 1

    # Write output files
    for rel_out, data in output_files.items():
        out_path = os.path.join(DOCS, rel_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        lines = [f"# {data['title']}", ""]

        for info in data["nodes"]:
            lines.append(_node_to_md(info))

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  wrote {rel_out} ({len(data['nodes'])} nodes)")

    total = core_count + plugin_count
    print(f"\nGenerated {len(output_files)} reference pages.")
    print(f"  Core nodes: {core_count}")
    print(f"  Plugin nodes: {plugin_count}")
    print(f"  Total: {total}")


if __name__ == "__main__":
    main()
