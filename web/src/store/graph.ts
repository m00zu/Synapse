import { create } from "zustand";
import { api } from "../api/client";
import type { NodeCategories, WidgetCatalog, WsEvent } from "../api/types";

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
  categories: NodeCategories | null;
  nodes: Node[];
  edges: Edge[];
  selectedId: string | null;
  loading: boolean;
  error: string | null;
  runStatus: Record<string, "running" | "ok" | "error">;
  runActive: boolean;
  previewVersions: Record<string, number>; // key = `${nodeId}:${port}`

  // Actions
  loadCatalog: () => Promise<void>;
  refreshGraph: () => Promise<void>;
  addNode: (type: string, x?: number, y?: number) => Promise<string>;
  removeNode: (id: string) => Promise<void>;
  patchProp: (id: string, prop: string, value: unknown) => Promise<void>;
  /** Update local position immediately (for smooth dragging). Server PATCH
   * is debounced separately — see commitNodePos. */
  setNodePos: (id: string, x: number, y: number) => void;
  /** Persist a node's current position to the server. Call from a debounced
   * drag-stop handler so we don't PATCH on every pixel of movement. */
  commitNodePos: (id: string) => Promise<void>;
  addEdge: (e: Edge) => Promise<void>;
  removeEdge: (e: Edge) => Promise<void>;
  select: (id: string | null) => void;
  applyWsEvent: (ev: WsEvent) => void;
}

export const useGraph = create<GraphState>((set, get) => ({
  catalog: null,
  categories: null,
  nodes: [],
  edges: [],
  selectedId: null,
  loading: false,
  error: null,
  runStatus: {},
  runActive: false,
  previewVersions: {},

  loadCatalog: async () => {
    set({ loading: true, error: null });
    try {
      const [catalog, categories] = await Promise.all([
        api.getCatalog(),
        api.getCategories(),
      ]);
      set({ catalog, categories, loading: false });
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

  setNodePos: (id, x, y) => {
    set({
      nodes: get().nodes.map((n) => (n.id === id ? { ...n, x, y } : n)),
    });
  },

  commitNodePos: async (id) => {
    const n = get().nodes.find((x) => x.id === id);
    if (!n) return;
    try {
      await api.patchPos(id, n.x, n.y);
    } catch (e) {
      console.error(`patchPos(${id}) failed:`, e);
    }
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
    } else if (ev.kind === "preview_available") {
      const key = `${ev.node_id}:${ev.port}`;
      set({
        previewVersions: {
          ...get().previewVersions,
          [key]: (get().previewVersions[key] ?? 0) + 1,
        },
      });
    }
  },
}));
