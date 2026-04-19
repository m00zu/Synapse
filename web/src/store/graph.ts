import { create } from "zustand";
import { api } from "../api/client";
import type { WidgetCatalog, WsEvent } from "../api/types";

export interface Node {
  id: string;
  type: string;
  x: number;
  y: number;
  props: Record<string, unknown>;
}

export interface Edge {
  src: string;
  dst: string;
  src_port?: string;
  dst_port?: string;
}

interface GraphState {
  catalog: WidgetCatalog | null;
  nodes: Node[];
  edges: Edge[];
  selectedId: string | null;
  loading: boolean;
  error: string | null;
  runStatus: Record<string, "running" | "ok" | "error">;
  runActive: boolean;

  // Actions
  loadCatalog: () => Promise<void>;
  refreshGraph: () => Promise<void>;
  addNode: (type: string, x?: number, y?: number) => Promise<string>;
  removeNode: (id: string) => Promise<void>;
  patchProp: (id: string, prop: string, value: unknown) => Promise<void>;
  addEdge: (e: Edge) => Promise<void>;
  removeEdge: (e: Edge) => Promise<void>;
  select: (id: string | null) => void;
  applyWsEvent: (ev: WsEvent) => void;
}

export const useGraph = create<GraphState>((set, get) => ({
  catalog: null,
  nodes: [],
  edges: [],
  selectedId: null,
  loading: false,
  error: null,
  runStatus: {},
  runActive: false,

  loadCatalog: async () => {
    set({ loading: true, error: null });
    try {
      const catalog = await api.getCatalog();
      set({ catalog, loading: false });
    } catch (e) {
      set({ error: String(e), loading: false });
    }
  },

  refreshGraph: async () => {
    // Phase 1c: server is source of truth for structure. We re-read the
    // nodes array from our local mirror; /api/graph's NodeGraphQt format
    // is too nested to adapt here, so we keep a shadow list of
    // { id, type, props } updated via addNode/removeNode/patchProp.
  },

  addNode: async (type, x = 0, y = 0) => {
    const { id } = await api.addNode({ type, x, y });
    // Use catalog default props for the optimistic client-side copy.
    const cat = get().catalog;
    const specs = cat?.[type] ?? [];
    const props: Record<string, unknown> = {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const walk = (s: any) => {
      if (s.children) s.children.forEach(walk);
      else if ("prop" in s) props[s.prop] = s.default;
    };
    specs.forEach(walk);
    set({ nodes: [...get().nodes, { id, type, x, y, props }] });
    return id;
  },

  removeNode: async (id) => {
    await api.deleteNode(id);
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.src !== id && e.dst !== id),
      selectedId: get().selectedId === id ? null : get().selectedId,
    });
  },

  patchProp: async (id, prop, value) => {
    await api.patchProps(id, { [prop]: value });
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, props: { ...n.props, [prop]: value } } : n
      ),
    });
  },

  addEdge: async (e) => {
    await api.connect(e);
    set({ edges: [...get().edges, e] });
  },

  removeEdge: async (e) => {
    await api.disconnect(e);
    set({
      edges: get().edges.filter(
        (x) => !(x.src === e.src && x.dst === e.dst &&
                 x.src_port === e.src_port && x.dst_port === e.dst_port)
      ),
    });
  },

  select: (id) => set({ selectedId: id }),

  applyWsEvent: (ev) => {
    if (ev.kind === "node_started") {
      set({
        runStatus: { ...get().runStatus, [ev.node_id]: "running" },
        runActive: true,
      });
    } else if (ev.kind === "node_finished") {
      set({
        runStatus: {
          ...get().runStatus,
          [ev.node_id]: ev.success ? "ok" : "error",
        },
      });
    } else if (ev.kind === "run_finished") {
      set({ runActive: false });
    }
    // node_progress + preview_available: ignored in Phase 1c
  },
}));
