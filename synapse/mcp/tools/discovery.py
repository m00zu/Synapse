"""Discovery tools: list_nodes, describe_node, search_nodes."""
from __future__ import annotations

from typing import Any

from ..controller import GraphController, NodeInfo


def list_nodes(controller: GraphController) -> list[dict[str, Any]]:
    """Return the catalog of all registered node types.

    Each entry has ``{name, type, category, summary}``.  Designed to fit
    into a single chat-context block (~5–15 KB at current Synapse scale).
    Use ``describe_node`` for full port/property details on a specific
    type before calling ``add_node``.
    """
    return [
        {'name': n.name, 'type': n.type_id,
         'category': n.category, 'summary': n.summary}
        for n in controller.list_registered()
    ]


def describe_node(controller: GraphController,
                  node_type: str) -> dict[str, Any]:
    """Return full info for a single registered node type.

    Returns ``{name, category, summary, inputs, outputs, properties}``.
    Each entry in ``properties`` is a dict::

        {
          "name":    "method",            # property identifier
          "kind":    "combo",             # one of: combo, bool, int, float,
                                          #         text, file_open, file_save,
                                          #         slider_int, slider_float,
                                          #         color, color_rgba, hidden,
                                          #         label, textarea, unknown
          "default": <current default value>,
          "options": ["IQR", "Z-score"],  # combo only
          "range":   [0.0, 100.0],        # numeric only (when defined)
          "hint":    "..."                # author-supplied per-property doc
        }

    **Use the ``options`` field for combo properties when calling
    ``set_property`` — the values must match exactly (case-sensitive).**
    """
    try:
        info: NodeInfo = controller.describe_registered(node_type)
    except KeyError:
        raise ValueError(
            f"Unknown node type: {node_type!r}. "
            f"Call list_nodes() to see all registered types.")
    # Prefer the rich property_specs if available; fall back to the
    # legacy bare-name list so older controllers (the Fake in tests)
    # still produce a sensible response.
    if info.property_specs:
        props_out = info.property_specs
    else:
        props_out = [{'name': n} for n in info.properties]
    return {
        'name': info.name,
        'category': info.category,
        'summary': info.summary,
        'inputs': list(info.input_ports),
        'outputs': list(info.output_ports),
        'properties': props_out,
    }


def search_nodes(controller: GraphController,
                 query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Substring search over node name + summary + category.

    Case-insensitive; whitespace-trimmed.  Empty query returns ``[]``.
    Designed as a fallback when ``list_nodes()`` returns too many entries
    to scan; v0 uses keyword matching (no embeddings).
    """
    q = (query or '').strip().lower()
    if not q:
        return []
    hits: list[tuple[int, dict]] = []  # (score, entry)
    for n in controller.list_registered():
        haystack = f'{n.name} {n.category} {n.summary}'.lower()
        if q not in haystack:
            continue
        # Score: name match beats summary match.
        score = 0
        if q in n.name.lower():
            score += 10
        if q in n.summary.lower():
            score += 1
        hits.append((score, {
            'name': n.name, 'type': n.type_id,
            'category': n.category, 'summary': n.summary,
        }))
    hits.sort(key=lambda x: -x[0])
    return [entry for _, entry in hits[:top_k]]
