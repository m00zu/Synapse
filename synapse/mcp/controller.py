"""GraphController: abstract facade over NodeGraph + fake for tests.

Real impl lives in the same file (NodeGraphController) and is wired by
``server.py`` once Synapse has constructed its NodeGraph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class NodeInfo:
    """Description of a node *type* (template, not an instance)."""
    category: str
    name: str            # NODE_NAME (human label, may contain spaces)
    type_id: str         # NodeGraphQt's type_ identifier
    properties: list[str]
    input_ports: list[str]
    output_ports: list[str]
    summary: str         # one-line description for catalog


@dataclass
class NodeRecord:
    """State of an active node *instance* in the current graph."""
    id: str
    type_id: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    status: str = 'pending'   # 'pending' | 'running' | 'clean' | 'error'
    last_message: str | None = None


class GraphController(Protocol):
    """Operations the MCP tools call.  All methods run on the Qt main thread."""

    def list_registered(self) -> list[NodeInfo]: ...
    def describe_registered(self, type_id: str) -> NodeInfo: ...
    def list_active(self) -> list[NodeRecord]: ...
    def list_connections(self) -> list[tuple[str, str, str, str]]: ...
    def get_node(self, node_id: str) -> NodeRecord: ...
    def add_node(self, type_id: str, properties: dict | None = None,
                 position: tuple[float, float] | None = None) -> str: ...
    def delete_node(self, node_id: str) -> None: ...
    def set_property(self, node_id: str, prop: str, value: Any) -> None: ...
    def connect(self, src_id: str, src_port: str,
                dst_id: str, dst_port: str) -> None: ...
    def disconnect(self, src_id: str, src_port: str,
                   dst_id: str, dst_port: str) -> None: ...
    def run_node(self, node_id: str) -> dict: ...
    def get_node_output(self, node_id: str, port_name: str) -> Any: ...
    def replace_node(self, node_id: str, new_type_id: str,
                     properties: dict | None = None) -> dict: ...


# ── Fake implementation for unit tests ──────────────────────────────────────

class FakeGraphController:
    """In-memory GraphController for unit-testing tools without Qt."""

    def __init__(self, registered: list[NodeInfo] | None = None) -> None:
        self._registered = {n.type_id: n for n in (registered or [])}
        self._active: dict[str, NodeRecord] = {}
        self._connections: list[tuple[str, str, str, str]] = []
        self._counter = 0
        self._run_results: dict[str, dict] = {}
        self._outputs: dict[tuple[str, str], Any] = {}

    # ── inspection / query ──────────────────────────────────────────────
    def list_registered(self) -> list[NodeInfo]:
        return list(self._registered.values())

    def describe_registered(self, type_id: str) -> NodeInfo:
        if type_id not in self._registered:
            raise KeyError(f"unknown node type: {type_id}")
        return self._registered[type_id]

    def list_active(self) -> list[NodeRecord]:
        return list(self._active.values())

    def list_connections(self) -> list[tuple[str, str, str, str]]:
        return list(self._connections)

    def get_node(self, node_id: str) -> NodeRecord:
        if node_id not in self._active:
            raise KeyError(f"unknown node id: {node_id}")
        return self._active[node_id]

    # ── mutation ────────────────────────────────────────────────────────
    def add_node(self, type_id: str, properties: dict | None = None,
                 position: tuple[float, float] | None = None) -> str:
        if type_id not in self._registered:
            raise KeyError(f"unknown node type: {type_id}")
        self._counter += 1
        nid = f'n{self._counter}'
        info = self._registered[type_id]
        self._active[nid] = NodeRecord(
            id=nid, type_id=type_id, name=info.name,
            properties=dict(properties or {}),
        )
        return nid

    def set_property(self, node_id: str, prop: str, value: Any) -> None:
        rec = self.get_node(node_id)
        rec.properties[prop] = value

    def connect(self, src_id: str, src_port: str,
                dst_id: str, dst_port: str) -> None:
        self.get_node(src_id)  # raises if missing
        self.get_node(dst_id)
        self._connections.append((src_id, src_port, dst_id, dst_port))

    def disconnect(self, src_id: str, src_port: str,
                   dst_id: str, dst_port: str) -> None:
        edge = (src_id, src_port, dst_id, dst_port)
        if edge not in self._connections:
            raise KeyError(f"no such connection: {edge}")
        self._connections.remove(edge)

    def delete_node(self, node_id: str) -> None:
        if node_id not in self._active:
            raise KeyError(f"unknown node id: {node_id}")
        del self._active[node_id]
        # Drop any edges touching this node.
        self._connections = [
            (s, sp, d, dp) for (s, sp, d, dp) in self._connections
            if s != node_id and d != node_id
        ]

    # ── execution ───────────────────────────────────────────────────────
    def run_node(self, node_id: str) -> dict:
        self.get_node(node_id)
        return self._run_results.get(node_id,
            {'success': True, 'message': None, 'duration_ms': 0.0})

    def set_run_result(self, node_id: str, *, success: bool,
                       message: str | None, duration_ms: float) -> None:
        """Test helper: pre-program what run_node returns for a node."""
        self._run_results[node_id] = {
            'success': success, 'message': message,
            'duration_ms': duration_ms,
        }

    def get_node_output(self, node_id: str, port_name: str) -> Any:
        self.get_node(node_id)   # raises KeyError if missing
        key = (node_id, port_name)
        if key not in self._outputs:
            raise KeyError(
                f"node {node_id!r} has no output value on port "
                f"{port_name!r} — run the node first")
        return self._outputs[key]

    def set_output(self, node_id: str, port_name: str, value: Any) -> None:
        """Test helper: stash an output value on a port."""
        self._outputs[(node_id, port_name)] = value

    def replace_node(self, node_id: str, new_type_id: str,
                     properties: dict | None = None) -> dict:
        """Swap a node's type; carry over compatible properties + edges."""
        old = self.get_node(node_id)            # raises KeyError if missing
        if new_type_id not in self._registered:
            raise KeyError(f"unknown node type: {new_type_id}")

        new_info = self._registered[new_type_id]
        # Property carry-over (best-effort: keep keys that are still
        # defined on the new type).
        carried_props = {k: v for k, v in old.properties.items()
                         if k in new_info.properties}
        if properties:
            carried_props.update(properties)

        # Snapshot edges + figure out which survive.
        in_edges  = [(s, sp, d, dp) for (s, sp, d, dp) in self._connections
                     if d == node_id]
        out_edges = [(s, sp, d, dp) for (s, sp, d, dp) in self._connections
                     if s == node_id]

        kept_in  = [e for e in in_edges  if e[3] in new_info.input_ports]
        kept_out = [e for e in out_edges if e[1] in new_info.output_ports]
        dropped: list[dict] = []
        for (s, sp, d, dp) in in_edges:
            if dp not in new_info.input_ports:
                dropped.append({'src_node_id': s, 'src_port': sp,
                                 'dst_node_id': d, 'dst_port': dp,
                                 'reason': f"new type has no input port "
                                           f"{dp!r}"})
        for (s, sp, d, dp) in out_edges:
            if sp not in new_info.output_ports:
                dropped.append({'src_node_id': s, 'src_port': sp,
                                 'dst_node_id': d, 'dst_port': dp,
                                 'reason': f"new type has no output port "
                                           f"{sp!r}"})

        # Delete the old node (drops ALL its edges).
        del self._active[node_id]
        self._connections = [(s, sp, d, dp) for (s, sp, d, dp) in self._connections
                             if s != node_id and d != node_id]

        # Re-create with the same id so referencing edges still match.
        self._active[node_id] = NodeRecord(
            id=node_id, type_id=new_type_id, name=new_info.name,
            properties=carried_props,
        )
        # Restore the compatible edges (same id is reused).
        self._connections.extend(kept_in + kept_out)

        return {'node_id': node_id, 'new_type': new_type_id,
                'carried_properties': list(carried_props.keys()),
                'dropped_connections': dropped}


# ── Real Qt-backed implementation ───────────────────────────────────────────

class NodeGraphController:
    """GraphController backed by a live NodeGraphQt NodeGraph.

    Must be called only from the Qt main thread (use ``ThreadHop`` from
    the asyncio side).  Owns no threading itself.
    """

    def __init__(self, graph) -> None:  # NodeGraphQt.NodeGraph
        self._graph = graph
        # Cache of registered types so we can answer list_registered() fast.
        self._type_summaries: dict[str, NodeInfo] = {}

    # ── helpers ─────────────────────────────────────────────────────────
    def _build_node_info(self, type_id: str, cls,
                          inst=None) -> NodeInfo:
        port_spec = getattr(cls, 'PORT_SPEC', {}) or {}
        category = getattr(cls, '__identifier__', '').split('.')[-1] or 'misc'
        doc = (cls.__doc__ or '').strip()
        summary = next((ln.strip() for ln in doc.splitlines()
                        if ln.strip()), '')

        # If we have an instance, read actual ports and custom properties.
        # Otherwise fall back to class-level PORT_SPEC (often stale) and
        # leave properties empty until describe_registered fills them in.
        if inst is not None:
            try:
                inputs = [p.name() for p in inst.input_ports()]
                outputs = [p.name() for p in inst.output_ports()]
            except Exception:
                inputs = list(port_spec.get('inputs', []))
                outputs = list(port_spec.get('outputs', []))
            try:
                custom = getattr(inst.model, 'custom_properties', {}) or {}
                props = [k for k in custom.keys() if not k.startswith('_')]
            except Exception:
                props = []
        else:
            inputs = list(port_spec.get('inputs', []))
            outputs = list(port_spec.get('outputs', []))
            props = []

        return NodeInfo(
            category=category,
            name=getattr(cls, 'NODE_NAME', type_id),
            type_id=type_id,
            properties=props,
            input_ports=inputs,
            output_ports=outputs,
            summary=summary,
        )

    def _ensure_info_cached(self, type_id: str, cls) -> NodeInfo:
        """Populate cache for ``type_id`` if missing.  Tries instance
        introspection; falls back to class-only on instantiation failure.
        """
        cached = self._type_summaries.get(type_id)
        if cached is not None and cached.properties:
            # Already cached with instance info (properties non-empty
            # implies we successfully instantiated).
            return cached
        # Try the heavyweight path first.
        try:
            inst = cls()
            info = self._build_node_info(type_id, cls, inst=inst)
        except Exception:
            info = self._build_node_info(type_id, cls, inst=None)
        self._type_summaries[type_id] = info
        return info

    def _safe_props(self, node) -> dict:
        """Return only user-defined custom properties (no framework state).

        ``node.properties()`` includes 'inputs', 'outputs', 'pos', etc. with
        values that contain ``PortModel`` instances and aren't JSON-serializable.
        ``node.model.custom_properties`` is the right surface — exactly the
        spinboxes/combos/checkboxes the user defined.
        """
        custom = getattr(node.model, 'custom_properties', {}) or {}
        out: dict = {}
        for k, v in custom.items():
            if k.startswith('_'):
                continue
            out[k] = v
        return out

    # ── inspection / query ──────────────────────────────────────────────
    def list_registered(self) -> list[NodeInfo]:
        out = []
        for type_id, cls in self._graph.node_factory.nodes.items():
            out.append(self._ensure_info_cached(type_id, cls))
        return out

    def describe_registered(self, type_id: str) -> NodeInfo:
        cls = self._graph.node_factory.nodes.get(type_id)
        if cls is None:
            raise KeyError(f"unknown node type: {type_id}")
        return self._ensure_info_cached(type_id, cls)

    def list_active(self) -> list[NodeRecord]:
        out = []
        for node in self._graph.all_nodes():
            out.append(NodeRecord(
                id=node.id, type_id=node.type_, name=node.name(),
                properties=self._safe_props(node),
                status=getattr(node, '_status', 'pending') or 'pending',
                last_message=getattr(node, '_last_message', None),
            ))
        return out

    def list_connections(self) -> list[tuple[str, str, str, str]]:
        out = []
        for node in self._graph.all_nodes():
            for out_port in node.output_ports():
                for in_port in out_port.connected_ports():
                    out.append((node.id, out_port.name(),
                                in_port.node().id, in_port.name()))
        return out

    def get_node(self, node_id: str) -> NodeRecord:
        node = self._graph.get_node_by_id(node_id)
        if node is None:
            raise KeyError(f"unknown node id: {node_id}")
        return NodeRecord(
            id=node.id, type_id=node.type_, name=node.name(),
            properties=self._safe_props(node),
            status=getattr(node, '_status', 'pending') or 'pending',
            last_message=getattr(node, '_last_message', None),
        )

    # Fallback when actual node width can't be measured.
    _AUTO_LAYOUT_X_PAD = 300.0
    _AUTO_LAYOUT_Y_PAD = 120.0
    _AUTO_LAYOUT_GAP = 60.0

    @staticmethod
    def _measured_width(node) -> float:
        """Return the rendered width of a NodeGraphQt node, after forcing
        its view to lay out.  Falls back to the X_PAD default on failure.
        """
        try:
            view = getattr(node, 'view', None)
            if view is not None and hasattr(view, 'draw_node'):
                try:
                    view.draw_node()
                except Exception:
                    pass
            if view is not None and hasattr(view, 'boundingRect'):
                w = float(view.boundingRect().width())
                if w > 0:
                    return w
        except Exception:
            pass
        return NodeGraphController._AUTO_LAYOUT_X_PAD

    def _next_auto_position(self) -> tuple[float, float]:
        """Pick a default x,y that doesn't overlap existing nodes.

        Places the new node one column to the right of the rightmost
        existing node (accounting for that node's actual width).
        """
        rightmost = None  # (x_right, y)
        for n in self._graph.all_nodes():
            try:
                p = n.pos()
            except Exception:
                continue
            x, y = (p[0], p[1]) if not hasattr(p, 'x') else (p.x(), p.y())
            x_right = x + self._measured_width(n)
            if rightmost is None or x_right > rightmost[0]:
                rightmost = (x_right, y)
        if rightmost is None:
            return (0.0, 0.0)
        return (rightmost[0] + self._AUTO_LAYOUT_GAP, rightmost[1])

    # ── mutation ────────────────────────────────────────────────────────
    def add_node(self, type_id: str, properties: dict | None = None,
                 position: tuple[float, float] | None = None) -> str:
        node = self._graph.create_node(type_id)
        if node is None:
            raise KeyError(f"unknown node type: {type_id}")
        if position is None:
            position = self._next_auto_position()
        node.set_pos(*position)
        for k, v in (properties or {}).items():
            node.set_property(k, v)
        return node.id

    def set_property(self, node_id: str, prop: str, value: Any) -> None:
        node = self._graph.get_node_by_id(node_id)
        if node is None:
            raise KeyError(f"unknown node id: {node_id}")
        node.set_property(prop, value)

    def connect(self, src_id: str, src_port: str,
                dst_id: str, dst_port: str) -> None:
        src = self._graph.get_node_by_id(src_id)
        dst = self._graph.get_node_by_id(dst_id)
        if src is None or dst is None:
            raise KeyError(f"unknown node id: {src_id} or {dst_id}")
        out = src.get_output(src_port)
        inp = dst.get_input(dst_port)
        if out is None:
            raise KeyError(f"{src_id} has no output port '{src_port}'")
        if inp is None:
            raise KeyError(f"{dst_id} has no input port '{dst_port}'")
        out.connect_to(inp)

    def disconnect(self, src_id: str, src_port: str,
                   dst_id: str, dst_port: str) -> None:
        src = self._graph.get_node_by_id(src_id)
        dst = self._graph.get_node_by_id(dst_id)
        if src is None or dst is None:
            raise KeyError(f"unknown node id: {src_id} or {dst_id}")
        out = src.get_output(src_port)
        inp = dst.get_input(dst_port)
        if out is None or inp is None:
            raise KeyError(
                f"no such port: {src_id}.{src_port} or {dst_id}.{dst_port}")
        if inp not in out.connected_ports():
            raise KeyError(
                f"no connection: {src_id}.{src_port} -> {dst_id}.{dst_port}")
        out.disconnect_from(inp)

    def delete_node(self, node_id: str) -> None:
        node = self._graph.get_node_by_id(node_id)
        if node is None:
            raise KeyError(f"unknown node id: {node_id}")
        # NodeGraphQt's delete_node also tears down attached edges.
        self._graph.delete_node(node)

    def replace_node(self, node_id: str, new_type_id: str,
                     properties: dict | None = None) -> dict:
        """Swap a NodeGraphQt node's type, preserving compatible state.

        Properties whose name still exists on the new type are carried
        over.  Edges whose port name still exists on the new type are
        reconnected.  Returns metadata about what was kept and dropped.
        """
        old = self._graph.get_node_by_id(node_id)
        if old is None:
            raise KeyError(f"unknown node id: {node_id}")
        if new_type_id not in self._graph.node_factory.nodes:
            raise KeyError(f"unknown node type: {new_type_id}")

        # Snapshot what we want to preserve.
        old_pos = None
        try:
            p = old.pos()
            old_pos = (p[0], p[1]) if not hasattr(p, 'x') else (p.x(), p.y())
        except Exception:
            pass
        old_props = self._safe_props(old)

        # Snapshot all attached edges.
        in_edges:  list[tuple[str, str, str, str]] = []  # (src_id, src_port, dst_id, dst_port)
        out_edges: list[tuple[str, str, str, str]] = []
        for in_port in old.input_ports():
            for src in in_port.connected_ports():
                in_edges.append((src.node().id, src.name(),
                                  old.id, in_port.name()))
        for out_port in old.output_ports():
            for dst in out_port.connected_ports():
                out_edges.append((old.id, out_port.name(),
                                   dst.node().id, dst.name()))

        # Learn the new type's port + property surface.
        new_info = self.describe_registered(new_type_id)
        new_in  = set(new_info.input_ports)
        new_out = set(new_info.output_ports)
        new_props_set = set(new_info.properties)

        # Delete the old (also drops its edges).
        self._graph.delete_node(old)

        # Create the replacement.
        new_node = self._graph.create_node(new_type_id)
        if new_node is None:
            raise RuntimeError(
                f"create_node({new_type_id!r}) returned None — "
                f"the replacement was not registered.")
        new_id = new_node.id
        if old_pos is not None:
            try:
                new_node.set_pos(*old_pos)
            except Exception:
                pass

        # Carry over properties (best-effort).
        carried: list[str] = []
        merged_props = {k: v for k, v in old_props.items()
                        if k in new_props_set}
        if properties:
            merged_props.update(properties)
        for k, v in merged_props.items():
            try:
                new_node.set_property(k, v)
                carried.append(k)
            except Exception:
                pass

        # Re-wire compatible edges.
        dropped: list[dict] = []
        for (s_id, s_port, _, d_port) in in_edges:
            if d_port not in new_in:
                dropped.append({'src_node_id': s_id, 'src_port': s_port,
                                 'dst_node_id': new_id, 'dst_port': d_port,
                                 'reason': f"new type has no input port "
                                           f"{d_port!r}"})
                continue
            try:
                self.connect(s_id, s_port, new_id, d_port)
            except KeyError as e:
                dropped.append({'src_node_id': s_id, 'src_port': s_port,
                                 'dst_node_id': new_id, 'dst_port': d_port,
                                 'reason': str(e)})
        for (_, s_port, d_id, d_port) in out_edges:
            if s_port not in new_out:
                dropped.append({'src_node_id': new_id, 'src_port': s_port,
                                 'dst_node_id': d_id, 'dst_port': d_port,
                                 'reason': f"new type has no output port "
                                           f"{s_port!r}"})
                continue
            try:
                self.connect(new_id, s_port, d_id, d_port)
            except KeyError as e:
                dropped.append({'src_node_id': new_id, 'src_port': s_port,
                                 'dst_node_id': d_id, 'dst_port': d_port,
                                 'reason': str(e)})

        return {'node_id': new_id, 'new_type': new_type_id,
                'carried_properties': carried,
                'dropped_connections': dropped}

    def get_node_output(self, node_id: str, port_name: str) -> Any:
        node = self._graph.get_node_by_id(node_id)
        if node is None:
            raise KeyError(f"unknown node id: {node_id}")
        outputs = getattr(node, 'output_values', None) or {}
        if port_name not in outputs:
            raise KeyError(
                f"node {node_id!r} has no value on output port "
                f"{port_name!r} — run the node first (current ports: "
                f"{list(outputs.keys())})")
        return outputs[port_name]

    # ── execution ───────────────────────────────────────────────────────
    def run_node(self, node_id: str) -> dict:
        import time
        node = self._graph.get_node_by_id(node_id)
        if node is None:
            raise KeyError(f"unknown node id: {node_id}")
        # Mirror Synapse's "Run" semantics: re-evaluate dirty upstream too.
        # Delegate to the existing helper if available; otherwise call evaluate.
        t0 = time.perf_counter()
        try:
            if hasattr(node, 'evaluate_with_upstream'):
                success, msg = node.evaluate_with_upstream()
            else:
                success, msg = node.evaluate()
            success = bool(success)
        except Exception as exc:
            return {'success': False,
                    'message': f'{type(exc).__name__}: {exc}',
                    'duration_ms': (time.perf_counter() - t0) * 1000.0}
        return {'success': success, 'message': msg,
                'duration_ms': (time.perf_counter() - t0) * 1000.0}
