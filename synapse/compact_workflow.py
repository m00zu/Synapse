"""
compact_workflow.py
===================
Compact workflow format for small LLMs.

Instead of generating full JSON with port names and properties, the LLM
produces a lightweight two-line format:

    Line 1: space-separated node class names (positional IDs: 1, 2, 3, …)
    Line 2: space-separated edge pairs as src>dst (integer IDs)

Example:
    ImageReadNode GaussianBlurNode BinaryThresholdNode ParticlePropsNode DataTableCellNode
    1>2 2>3 3>4 4>5

The auto-wiring engine resolves which specific ports to connect based on
type compatibility, handling fan-out, multi-output nodes, and skip connections.

Usage:

    from compact_workflow import parse_compact, compact_to_full_json, convert_full_to_compact

    # LLM output → full JSON workflow
    text = "ImageReadNode GaussianBlurNode\\n1>2"
    workflow = compact_to_full_json(text, schema_path="llm_node_schema.json")

    # Existing full JSON → compact format (for training data conversion)
    compact = convert_full_to_compact({"nodes": [...], "edges": [...]})
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Port type compatibility
# ---------------------------------------------------------------------------

# Forward compatibility: an output of type X can connect to an input of type Y
# if Y is in COMPAT[X].
_COMPAT: dict[str, set[str]] = {
    "image":       {"image", "any"},
    "mask":        {"mask", "image", "any", "image/mask"},
    "label_image": {"label_image", "any"},
    "skeleton":    {"skeleton", "mask", "any"},
    "table":       {"table", "any", "in"},
    "stat":        {"stat", "table", "any", "in"},
    "figure":      {"figure", "any"},
    "confocal":    {"confocal", "any"},
    "path":        {"path", "any"},
    "any":         {"any", "image", "mask", "table", "figure", "path",
                    "label_image", "skeleton", "stat", "confocal", "in",
                    "image/mask"},
}


def _types_compatible(out_type: str, in_type: str) -> bool:
    """Check if an output port type can connect to an input port type."""
    if out_type == in_type:
        return True
    return in_type in _COMPAT.get(out_type, {out_type, "any"})


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

_cached_catalog: dict | None = None
_cached_path: str | None = None


def _load_catalog(schema_path: str | Path) -> dict[str, dict]:
    """Load and cache the node catalog from llm_node_schema.json."""
    global _cached_catalog, _cached_path
    schema_path = str(schema_path)
    if _cached_catalog is not None and _cached_path == schema_path:
        return _cached_catalog

    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)

    catalog = {}
    for name, info in schema.get("node_catalog", {}).items():
        inputs = []
        for p in info.get("inputs", []):
            if isinstance(p, dict):
                inputs.append({"name": p["name"], "type": p.get("type", "any")})
            else:
                inputs.append({"name": str(p), "type": "any"})

        outputs = []
        for p in info.get("outputs", []):
            if isinstance(p, dict):
                outputs.append({"name": p["name"], "type": p.get("type", "any")})
            else:
                outputs.append({"name": str(p), "type": "any"})

        catalog[name] = {"inputs": inputs, "outputs": outputs}

    _cached_catalog = catalog
    _cached_path = schema_path
    return catalog


# ---------------------------------------------------------------------------
# Parser: compact text → node list + edge pairs
# ---------------------------------------------------------------------------

def parse_compact(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Parse compact format text into (node_types, edge_pairs).

    Args:
        text: Two-line format — line 1 is node names, line 2 is edge pairs.
              Can also be a single line (nodes only, no edges).

    Returns:
        (node_types, edge_pairs) where edge_pairs are 1-based integer tuples.
    """
    text = text.strip()
    if not text:
        return [], []

    lines = text.split("\n")
    line1 = lines[0].strip()
    line2 = lines[1].strip() if len(lines) > 1 else ""

    # Parse node names
    node_types = line1.split()

    # Parse edge pairs
    edge_pairs = []
    if line2:
        for pair in line2.split():
            pair = pair.strip()
            if ">" not in pair:
                continue
            parts = pair.split(">")
            if len(parts) != 2:
                continue
            try:
                src = int(parts[0].lstrip("en"))  # allow "e1>2" or "n1>n2" or "1>2"
                dst = int(parts[1].lstrip("en"))
                edge_pairs.append((src, dst))
            except ValueError:
                continue

    return node_types, edge_pairs


# ---------------------------------------------------------------------------
# Auto-wiring: resolve edge pairs to specific port names
# ---------------------------------------------------------------------------

def _resolve_edge(
    src_info: dict,
    dst_info: dict,
    src_used_outputs: dict[str, int],
    dst_used_inputs: set[str],
) -> tuple[str | None, str | None]:
    """
    Find the best output→input port pair for one edge.

    Handles multi-output nodes (SplitRGBNode → red, green, blue) by tracking
    which output ports have already been used via src_used_outputs counter.

    Args:
        src_info:  {"inputs": [...], "outputs": [...]} for source node
        dst_info:  {"inputs": [...], "outputs": [...]} for destination node
        src_used_outputs: {port_name: use_count} — tracks how many times
                          each output port of this source has been wired
        dst_used_inputs:  set of input port names already connected on dst

    Returns:
        (out_port_name, in_port_name) or (None, None) if no match found.
    """
    # Build candidate pairs: (out_port, in_port, priority)
    # Priority: 0 = exact type match, 1 = compatible match
    candidates = []
    for out_port in src_info["outputs"]:
        out_name = out_port["name"]
        out_type = out_port["type"]
        for in_port in dst_info["inputs"]:
            in_name = in_port["name"]
            in_type = in_port["type"]
            if in_name in dst_used_inputs:
                continue  # already connected
            if not _types_compatible(out_type, in_type):
                continue
            priority = 0 if out_type == in_type else 1
            candidates.append((out_name, in_name, priority))

    if not candidates:
        return None, None

    # Sort by priority (exact match first), then by least-used output port
    candidates.sort(key=lambda c: (c[2], src_used_outputs.get(c[0], 0)))
    return candidates[0][0], candidates[0][1]


def auto_wire(
    node_types: list[str],
    edge_pairs: list[tuple[int, int]],
    catalog: dict[str, dict],
) -> dict:
    """
    Convert compact format (node types + edge pairs) to a full JSON workflow.

    Args:
        node_types:  List of node class names (1-indexed positionally).
        edge_pairs:  List of (src_idx, dst_idx) 1-based pairs.
        catalog:     Node catalog from _load_catalog().

    Returns:
        Full workflow dict: {"nodes": [...], "edges": [...]}
    """
    # Build nodes
    nodes = []
    for i, cls_name in enumerate(node_types, 1):
        nodes.append({"id": f"n{i}", "type": cls_name, "custom": {}})

    # Track port usage per node for round-robin multi-output assignment
    # Key: (node_idx, port_name) → use count
    out_usage: dict[int, dict[str, int]] = {i: {} for i in range(1, len(node_types) + 1)}
    in_used: dict[int, set[str]] = {i: set() for i in range(1, len(node_types) + 1)}

    edges = []
    for src_idx, dst_idx in edge_pairs:
        if src_idx < 1 or src_idx > len(node_types):
            continue
        if dst_idx < 1 or dst_idx > len(node_types):
            continue

        src_cls = node_types[src_idx - 1]
        dst_cls = node_types[dst_idx - 1]

        src_info = catalog.get(src_cls)
        dst_info = catalog.get(dst_cls)
        if src_info is None or dst_info is None:
            continue

        out_port, in_port = _resolve_edge(
            src_info, dst_info,
            out_usage[src_idx], in_used[dst_idx],
        )
        if out_port is None or in_port is None:
            continue

        edges.append({
            "from_node_id": f"n{src_idx}",
            "from_port": out_port,
            "to_node_id": f"n{dst_idx}",
            "to_port": in_port,
        })

        # Track usage
        out_usage[src_idx][out_port] = out_usage[src_idx].get(out_port, 0) + 1
        in_used[dst_idx].add(in_port)

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# High-level: compact text → full JSON workflow
# ---------------------------------------------------------------------------

def compact_to_full_json(
    text: str,
    schema_path: str | Path = "llm_node_schema.json",
) -> dict:
    """
    Convert compact LLM output to a full JSON workflow.

    Args:
        text:         Compact format text (2 lines: nodes, edges)
        schema_path:  Path to llm_node_schema.json

    Returns:
        Full workflow dict ready for WorkflowLoader.build()
    """
    catalog = _load_catalog(schema_path)
    node_types, edge_pairs = parse_compact(text)

    if not edge_pairs and len(node_types) > 1:
        # No edges specified → assume linear chain
        edge_pairs = [(i, i + 1) for i in range(1, len(node_types))]

    return auto_wire(node_types, edge_pairs, catalog)


# ---------------------------------------------------------------------------
# Reverse: full JSON workflow → compact format (for training data conversion)
# ---------------------------------------------------------------------------

def convert_full_to_compact(workflow: dict) -> str:
    """
    Convert a full JSON workflow to compact format.

    Args:
        workflow: {"nodes": [...], "edges": [...]}

    Returns:
        Two-line compact format string.
    """
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])

    # Map node IDs to 1-based positions
    id_to_pos = {}
    for i, node in enumerate(nodes):
        id_to_pos[node.get("id", f"n{i+1}")] = i + 1

    # Line 1: node types
    node_types = [n.get("type", "Unknown") for n in nodes]
    line1 = " ".join(node_types)

    # Line 2: edge pairs
    edge_strs = []
    for edge in edges:
        src_pos = id_to_pos.get(edge.get("from_node_id"))
        dst_pos = id_to_pos.get(edge.get("to_node_id"))
        if src_pos is not None and dst_pos is not None:
            edge_strs.append(f"{src_pos}>{dst_pos}")
    line2 = " ".join(edge_strs)

    return f"{line1}\n{line2}"


# ---------------------------------------------------------------------------
# Batch conversion: convert all training examples
# ---------------------------------------------------------------------------

def convert_training_data(
    input_path: str | Path,
    output_path: str | Path,
    schema_path: str | Path = "llm_node_schema.json",
) -> dict:
    """
    Convert a full-JSON training file to compact format.

    Args:
        input_path:   Path to examples*.json (list of {user, assistant} dicts)
        output_path:  Path to write compact JSONL
        schema_path:  Path to llm_node_schema.json

    Returns:
        Stats dict: {total, converted, skipped, round_trip_ok, round_trip_fail}
    """
    catalog = _load_catalog(schema_path)

    with open(input_path, encoding="utf-8") as fh:
        examples = json.load(fh)

    stats = {"total": len(examples), "converted": 0, "skipped": 0,
             "round_trip_ok": 0, "round_trip_fail": 0}
    rows = []

    for ex in examples:
        user = ex.get("user", "")
        assistant = ex.get("assistant")
        if not user or not isinstance(assistant, dict):
            stats["skipped"] += 1
            continue

        compact = convert_full_to_compact(assistant)

        # Round-trip test: compact → full JSON → compare edges
        try:
            node_types, edge_pairs = parse_compact(compact)
            reconstructed = auto_wire(node_types, edge_pairs, catalog)

            # Compare: do the reconstructed edges match the original?
            orig_edges = set()
            for e in assistant.get("edges", []):
                orig_edges.add((
                    e.get("from_node_id"), e.get("from_port"),
                    e.get("to_node_id"), e.get("to_port"),
                ))
            recon_edges = set()
            for e in reconstructed.get("edges", []):
                recon_edges.add((
                    e.get("from_node_id"), e.get("from_port"),
                    e.get("to_node_id"), e.get("to_port"),
                ))

            if orig_edges == recon_edges:
                stats["round_trip_ok"] += 1
            else:
                stats["round_trip_fail"] += 1
        except Exception:
            stats["round_trip_fail"] += 1

        rows.append({
            "prompt": user,
            "completion": compact,
        })
        stats["converted"] += 1

    with open(output_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description="Convert full JSON training data to compact format, "
                    "or test round-trip accuracy."
    )
    sub = parser.add_subparsers(dest="command")

    # convert command
    conv = sub.add_parser("convert", help="Convert training data")
    conv.add_argument("input", help="Input examples*.json file or glob pattern")
    conv.add_argument("-o", "--output", default="finetune/data/compact_train.jsonl",
                      help="Output JSONL path")
    conv.add_argument("--schema", default="llm_node_schema.json",
                      help="Path to llm_node_schema.json")

    # test command
    test = sub.add_parser("test", help="Test compact format on a single prompt")
    test.add_argument("text", nargs="?",
                      default="ImageReadNode GaussianBlurNode BinaryThresholdNode "
                              "ParticlePropsNode DataTableCellNode\n1>2 2>3 3>4 4>5")
    test.add_argument("--schema", default="llm_node_schema.json")

    # round-trip command
    rt = sub.add_parser("roundtrip", help="Test round-trip accuracy on training data")
    rt.add_argument("input", help="Input examples*.json")
    rt.add_argument("--schema", default="llm_node_schema.json")

    args = parser.parse_args()

    if args.command == "convert":
        files = sorted(glob.glob(args.input))
        if not files:
            files = [args.input]
        total_stats = {"total": 0, "converted": 0, "skipped": 0,
                       "round_trip_ok": 0, "round_trip_fail": 0}
        all_rows = []

        for f in files:
            print(f"Processing {f}...")
            stats = convert_training_data(f, "/dev/null", args.schema)
            for k in total_stats:
                total_stats[k] += stats[k]

            # Re-read to collect rows
            with open(f, encoding="utf-8") as fh:
                examples = json.load(fh)
            catalog = _load_catalog(args.schema)
            for ex in examples:
                user = ex.get("user", "")
                assistant = ex.get("assistant")
                if user and isinstance(assistant, dict):
                    all_rows.append({
                        "prompt": user,
                        "completion": convert_full_to_compact(assistant),
                    })

        with open(args.output, "w", encoding="utf-8") as fh:
            for row in all_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"\nResults:")
        print(f"  Total examples:     {total_stats['total']}")
        print(f"  Converted:          {total_stats['converted']}")
        print(f"  Skipped:            {total_stats['skipped']}")
        print(f"  Round-trip OK:      {total_stats['round_trip_ok']}")
        print(f"  Round-trip FAIL:    {total_stats['round_trip_fail']}")
        print(f"  Accuracy:           {total_stats['round_trip_ok']}/{total_stats['converted']}"
              f" ({100*total_stats['round_trip_ok']/max(1,total_stats['converted']):.1f}%)")
        print(f"\n  Wrote {len(all_rows)} rows → {args.output}")

    elif args.command == "test":
        catalog = _load_catalog(args.schema)
        node_types, edge_pairs = parse_compact(args.text)
        print(f"Nodes: {node_types}")
        print(f"Edges: {edge_pairs}")
        result = auto_wire(node_types, edge_pairs, catalog)
        print(f"\nFull JSON:")
        print(json.dumps(result, indent=2))

    elif args.command == "roundtrip":
        stats = convert_training_data(args.input, "/dev/null", args.schema)
        total = stats["converted"]
        ok = stats["round_trip_ok"]
        fail = stats["round_trip_fail"]
        print(f"Round-trip test on {args.input}:")
        print(f"  Total:    {total}")
        print(f"  OK:       {ok} ({100*ok/max(1,total):.1f}%)")
        print(f"  FAIL:     {fail} ({100*fail/max(1,total):.1f}%)")

    else:
        parser.print_help()
