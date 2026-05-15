"""Backward-compatibility migration for serialized workflow data.

When a node class is renamed (its ``__identifier__`` or class name
changes), workflows saved under the old name would otherwise fail to
load -- ``NodeGraphQt`` can't find the type, and the user just sees
"unknown node type" with no recourse.

This module patches the layout dict in-place at load time, rewriting
the ``type_`` field of any node that matches a known rename rule.
Apply it via ``migrate_layout(data)`` before calling
``graph.deserialize_session(...)``.

Returns the number of nodes that were migrated, so the caller can show
a status message ("migrated N nodes").
"""
from __future__ import annotations

from typing import Any


# ── Identifier-prefix renames ───────────────────────────────────────────
#
# Order matters: longer prefixes first, so 'plugins.Plugins.confocal'
# matches before the more general 'plugins.Plugins.'.  All renames are
# applied as left-anchored prefix substitutions on the node's
# ``type_`` string (which has the form ``<__identifier__>.<ClassName>``).

_IDENTIFIER_RENAMES = (
    # plugins.Plugins.<sub>  ->  plugins.<Renamed>
    ('plugins.Plugins.filopodia.',     'plugins.Filopodia.'),
    ('plugins.Plugins.Report.',        'plugins.Report.'),
    ('plugins.Plugins.Segmentation.',  'plugins.Segmentation.'),
    ('plugins.Plugins.VideoAnalysis.', 'plugins.VideoAnalysis.'),
    ('plugins.Plugins.confocal.',      'plugins.Confocal.'),
    ('plugins.Plugins.example.',       'plugins.Example.'),
)


# ── Per-class renames (whole type_ string) ──────────────────────────────
#
# Use this for one-off class renames that aren't covered by the prefix
# table above (e.g., GlobalMaskPropsNode -> ImageStatsNode).

_CLASS_RENAMES = {
    # Old behaviour from synapse.app -- GlobalMaskProps node became
    # ImageStats; the renamed node also needs a default property.
    # We can't set the property here easily (would need to know the
    # node's full type_), so we handle this as a tuple
    # (old_suffix, new_suffix, optional_property_seed).
}


# ── Suffix renames that need a property seeded ──────────────────────────
#
# (old_suffix, new_suffix, property_seed_dict).  Applied if the
# node's type_ endswith old_suffix.

_SUFFIX_RENAMES_WITH_PROPS = (
    ('.GlobalMaskPropsNode', '.ImageStatsNode', {'per_channel': False}),
)


# ── Port-name renames (per node-type suffix) ──────────────────────────
#
# When a node renames one of its ports, saved connections referencing
# the old name need rewriting on load.  Format::
#
#     {<node_type_suffix>: {<old_port_name>: <new_port_name>}}
#
# A connection references a port via ``[node_id, port_name]`` (both
# ``out`` and ``in`` arrays), so the migrator looks up each
# referenced node's type and applies the matching rename rule.

_PORT_RENAMES = {
    # ImageMathNode: B port retyped image (was mask-only); renamed for clarity.
    '.ImageMathNode': {
        'B (mask)': 'B (image/mask)',
    },
}


def migrate_layout(layout_data: dict[str, Any]) -> int:
    """Migrate deprecated node types in serialized workflow data.

    Mutates ``layout_data`` in place.  Returns the number of nodes
    that were migrated (for status-message purposes).
    """
    if not isinstance(layout_data, dict):
        return 0
    nodes = layout_data.get('nodes', {})
    if not isinstance(nodes, dict):
        return 0

    migrated = 0
    for n_data in nodes.values():
        if not isinstance(n_data, dict):
            continue
        t = str(n_data.get('type_', ''))
        if not t:
            continue

        # 1. Suffix renames that need a property seeded.
        new_t = None
        for old_suffix, new_suffix, prop_seed in _SUFFIX_RENAMES_WITH_PROPS:
            if t.endswith(old_suffix):
                new_t = t[:-len(old_suffix)] + new_suffix
                custom = n_data.get('custom')
                if not isinstance(custom, dict):
                    custom = {}
                for k, v in prop_seed.items():
                    custom.setdefault(k, v)
                n_data['custom'] = custom
                break

        # 2. Identifier-prefix renames.
        if new_t is None:
            for old_prefix, new_prefix in _IDENTIFIER_RENAMES:
                if t.startswith(old_prefix):
                    new_t = new_prefix + t[len(old_prefix):]
                    break

        # 3. Full type_ renames (if we ever need them).
        if new_t is None and t in _CLASS_RENAMES:
            new_t = _CLASS_RENAMES[t]

        if new_t is not None and new_t != t:
            n_data['type_'] = new_t
            migrated += 1

    # ── Port-name renames -----------------------------------------------
    # Walk connections and rewrite port-name references on either end
    # whose owning node matches a _PORT_RENAMES suffix rule.  Uses each
    # node's CURRENT (post-rename) ``type_`` so the rule keys can target
    # the post-rename class name.
    connections = layout_data.get('connections')
    if isinstance(connections, list) and _PORT_RENAMES:
        for conn in connections:
            if not isinstance(conn, dict):
                continue
            for side in ('out', 'in'):
                ref = conn.get(side)
                if not (isinstance(ref, list) and len(ref) == 2):
                    continue
                node_id, port_name = ref
                node_data = nodes.get(node_id) if isinstance(node_id, str) else None
                if not isinstance(node_data, dict):
                    continue
                node_type = str(node_data.get('type_', ''))
                for type_suffix, renames in _PORT_RENAMES.items():
                    if node_type.endswith(type_suffix) and port_name in renames:
                        ref[1] = renames[port_name]
                        migrated += 1
                        break

    return migrated
