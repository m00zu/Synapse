#!/usr/bin/env python3
"""Auto-generate node reference Markdown pages from source __identifier__.

Output paths are derived directly from each node's ``__identifier__``
attribute -- no manual file-to-page mapping.  Adding a new node
anywhere under ``synapse/nodes/`` or ``Synapse-Plugins/`` automatically
creates or updates its docs page on the next run.

Identifier -> output path:

    nodes.io                      ->  docs/nodes/io.md
    nodes.dataframe.Compute       ->  docs/nodes/dataframe/compute.md
    nodes.image_process.filter    ->  docs/nodes/image_process/filter.md
    plugins.ML.Classification     ->  docs/plugins/ml/classification.md

Folder names and filenames are lowercased for cross-platform URL
stability.  Page titles come from the last identifier component
(underscores split into words; known acronyms like ``io`` stay
upper-case).

Source roots:

    PySide_Node/synapse/nodes/    (core nodes)
    Synapse-Plugins/              (canonical plugin repo; sibling dir)

Override the plugins root via the ``SYNAPSE_PLUGINS_DIR`` env var.

Run:  python synapse/generate_docs.py
"""

import ast
import os
import re
import textwrap


# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
PLUGINS_ROOT = os.environ.get(
    "SYNAPSE_PLUGINS_DIR",
    os.path.normpath(os.path.join(ROOT, "..", "Synapse-Plugins")),
)
CORE_NODES_ROOT = os.path.join(ROOT, "synapse", "nodes")

# Directories to skip while walking source roots.
_SKIP_DIRS = {"__pycache__", "rust", "vendor", ".git", "build", "dist"}

# Acronyms kept upper-case when converting identifier components to
# page titles.  Anything not in here is title-cased normally.
_ACRONYMS = {
    "io", "ml", "ai", "ui", "api", "url", "json", "csv", "rdkit",
    "sam2", "shap", "pca", "umap", "id", "rgb", "rgba", "svg", "html",
    "cli", "gui", "mcp", "ngs", "pdb",
}


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
            # Strip keyword lines (and any continuation lines after them)
            lines = ds.strip().split("\n")
            clean = []
            in_keywords = False
            for line in lines:
                stripped = line.strip()
                if stripped.lower().startswith("keywords:") or stripped.lower().startswith("keyword:"):
                    in_keywords = True
                    continue
                if in_keywords:
                    # Continuation lines are indented or contain CJK/comma-heavy content
                    if not stripped or (stripped and not stripped[0].isupper() and not stripped.startswith("-") and not stripped.startswith("*")):
                        continue
                    in_keywords = False
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

            # __init__ method -- look for add_input, add_output, _add_*_spinbox, etc.
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
                                        # PORT_COLORS['table'] -> 'table'
                                        sl = kw.value.slice
                                        if isinstance(sl, ast.Constant):
                                            port_type = sl.value
                                    elif isinstance(kw.value, ast.Call):
                                        # PORT_COLORS.get('table', ...) -> 'table'
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
        # those are authoritative -- drop the PORT_SPEC placeholder entries (name == type).
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

                    # Convert standalone **param** -- lines to bullet items
                    if (stripped.startswith("**") and "--" in stripped
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


# ── Routing: identifier -> output path ───────────────────────────────────

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_camel(name):
    """Split CamelCase boundaries: ``'VideoAnalysis'`` -> ``['Video', 'Analysis']``.

    Leaves snake_case and lowercase strings alone (so ``'image_process'``
    returns ``['image_process']`` and ``'io'`` returns ``['io']``).
    Acronyms like ``'IO'`` or ``'ML'`` stay together because the regex
    requires a lowercase letter after the second uppercase.
    """
    return _CAMEL_BOUNDARY.split(name)


def _slugify(name):
    """Lowercase identifier component for filesystem path use.

    ``'VideoAnalysis'`` -> ``'video_analysis'``
    ``'image_process'`` -> ``'image_process'``
    ``'IO'``            -> ``'io'``
    """
    return "_".join(part.lower() for part in _split_camel(name))


def _make_title(name):
    """Convert an identifier component to a page title.

    Handles snake_case (``'image_process'`` -> ``'Image Process'``),
    CamelCase (``'VideoAnalysis'`` -> ``'Video Analysis'``), and acronyms
    (``'io'`` -> ``'IO'``).
    """
    # First split CamelCase, then split each chunk on underscores.
    words = []
    for chunk in _split_camel(name):
        words.extend(chunk.split("_"))
    return " ".join(
        w.upper() if w.lower() in _ACRONYMS else w.capitalize()
        for w in words if w
    )


def _identifier_to_output_path(identifier):
    """Map a node ``__identifier__`` to ``(relative_doc_path, page_title)``.

    Returns ``None`` if the identifier is missing or malformed.

    Examples::

        'nodes.io'                  -> ('nodes/io.md', 'IO')
        'nodes.dataframe.Compute'   -> ('nodes/dataframe/compute.md', 'Compute')
        'plugins.ML.Classification' -> ('plugins/ml/classification.md', 'Classification')
        'plugins.VideoAnalysis'     -> ('plugins/video_analysis.md', 'Video Analysis')
    """
    if not identifier or "." not in identifier:
        return None
    parts = identifier.split(".")
    if len(parts) < 2:
        return None
    namespace = _slugify(parts[0])
    rest_slugs = [_slugify(p) for p in parts[1:]]
    title_source = parts[-1]  # original case for nicer titles
    if len(rest_slugs) == 1:
        rel_path = f"{namespace}/{rest_slugs[0]}.md"
    else:
        folder = "/".join(rest_slugs[:-1])
        filename = rest_slugs[-1]
        rel_path = f"{namespace}/{folder}/{filename}.md"
    return rel_path, _make_title(title_source)


def _walk_source_files(root):
    """Yield .py file paths under ``root``, skipping junk directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # Group node infos by output path.
    output_pages: dict[str, dict] = {}  # rel_path -> {title, nodes: []}
    files_scanned = 0
    nodes_routed = 0
    nodes_skipped = 0
    skipped_details: list[tuple[str, str, object]] = []

    for source_root in [CORE_NODES_ROOT, PLUGINS_ROOT]:
        if not os.path.isdir(source_root):
            print(f"  warn: source root not found: {source_root}")
            continue
        for py_file in _walk_source_files(source_root):
            files_scanned += 1
            nodes = _extract_nodes(py_file)
            for info in nodes:
                identifier = info.get("identifier")
                routed = _identifier_to_output_path(identifier)
                if routed is None:
                    nodes_skipped += 1
                    skipped_details.append(
                        (info["class_name"], py_file, identifier))
                    continue
                rel_path, title = routed
                if rel_path not in output_pages:
                    output_pages[rel_path] = {"title": title, "nodes": []}
                output_pages[rel_path]["nodes"].append(info)
                nodes_routed += 1

    # Sort each page's nodes alphabetically by NODE_NAME for stable output.
    for page in output_pages.values():
        page["nodes"].sort(key=lambda n: (n.get("node_name") or "").lower())

    # Write pages.
    written = []
    for rel_path, data in sorted(output_pages.items()):
        out_path = os.path.join(DOCS, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        lines = [f"# {data['title']}", ""]
        for info in data["nodes"]:
            lines.append(_node_to_md(info))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        written.append(rel_path)

    # Summary.
    print(f"\nGenerated {len(written)} reference pages.")
    print(f"  Files scanned:  {files_scanned}")
    print(f"  Nodes routed:   {nodes_routed}")
    print(f"  Nodes skipped:  {nodes_skipped}")
    if skipped_details:
        print("\nSkipped nodes (missing or malformed __identifier__):")
        for cls, fp, ident in skipped_details[:20]:
            short = os.path.relpath(fp, ROOT)
            print(f"  - {cls} in {short}  (__identifier__={ident!r})")
        if len(skipped_details) > 20:
            print(f"  ... and {len(skipped_details) - 20} more")

    # Detect potentially-stale pages: any .md under docs/nodes/ or
    # docs/plugins/ that this run did NOT generate.  Leftover pages
    # from a previous (FILE_MAP-based) layout end up here.  We report
    # them but don't delete -- review and rm manually.
    auto_roots = ["nodes", "plugins"]
    stale = []
    written_set = set(written)
    for root_name in auto_roots:
        abs_root = os.path.join(DOCS, root_name)
        if not os.path.isdir(abs_root):
            continue
        for dirpath, _, filenames in os.walk(abs_root):
            for fn in filenames:
                if not fn.endswith(".md"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), DOCS)
                # Normalise to forward slashes for comparison.
                rel = rel.replace(os.sep, "/")
                if rel not in written_set:
                    stale.append(rel)

    if stale:
        print(f"\nPotentially-stale auto-generated pages ({len(stale)}):")
        for s in sorted(stale):
            print(f"  - {s}")
        print("\n  These weren't generated by this run.  Review and delete if")
        print("  they're orphans from the previous (FILE_MAP-based) layout.")


if __name__ == "__main__":
    main()
