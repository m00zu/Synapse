import type { NodeCategories, WidgetCatalog, WsEvent } from "./types";

const API = ""; // same-origin; Vite proxies /api/* in dev

async function jfetch<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}: ${await resp.text()}`);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  // Catalog
  getCatalog: () => jfetch<WidgetCatalog>("/api/nodes"),
  getCategories: () => jfetch<NodeCategories>("/api/nodes/categories"),

  // Graph CRUD
  getGraph: () => jfetch<Record<string, unknown>>("/api/graph"),
  addNode: (body: { type: string; x?: number; y?: number }) =>
    jfetch<{ id: string }>("/api/graph/nodes", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteNode: (id: string) =>
    fetch(`/api/graph/nodes/${encodeURIComponent(id)}`, { method: "DELETE" })
      .then((r) => { if (!r.ok && r.status !== 204) throw new Error(String(r.status)); }),
  patchProps: (id: string, props: Record<string, unknown>) =>
    jfetch<{ ok: true }>(`/api/graph/nodes/${encodeURIComponent(id)}/props`, {
      method: "PATCH",
      body: JSON.stringify(props),
    }),
  patchPos: (id: string, x: number, y: number) =>
    jfetch<{ ok: true }>(`/api/graph/nodes/${encodeURIComponent(id)}/pos`, {
      method: "PATCH",
      body: JSON.stringify({ x, y }),
    }),
  connect: (body: { src: string; dst: string; src_port?: string; dst_port?: string }) =>
    jfetch<{ ok: true }>("/api/graph/edges", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  disconnect: (body: { src: string; dst: string; src_port?: string; dst_port?: string }) =>
    jfetch<{ ok: true }>("/api/graph/edges", {
      method: "DELETE",
      body: JSON.stringify(body),
    }),

  // Execution
  runGraph: () => jfetch<{ run_id: string }>("/api/exec/run", { method: "POST" }),
  stopGraph: () =>
    fetch("/api/exec/stop", { method: "POST" })
      .then((r) => { if (!r.ok && r.status !== 204) throw new Error(String(r.status)); }),

  // Files
  uploadFile: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/files/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`upload failed: ${r.status}`);
    return (await r.json()) as { server_path: string };
  },
  browseDir: (path: string) =>
    jfetch<{ root: string; entries: { name: string; is_dir: boolean; path: string }[] }>(
      `/api/files/browse?path=${encodeURIComponent(path)}`
    ),

  // WebSocket — reconnects with jittered backoff on close.
  openWs: (onEvent: (ev: WsEvent) => void): { close: () => void } => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/api/ws`;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let closed = false;
    // Track whether we've ever been connected + are currently connected.
    // Use these to emit one disconnect toast (not a spam per reconnect
    // attempt) and one reconnect toast when we actually recover.
    let hasBeenOpen = false;
    let isOpen = false;

    const connect = () => {
      if (closed) return;
      ws = new WebSocket(url);
      ws.onopen = () => {
        const wasReconnect = hasBeenOpen && !isOpen;
        hasBeenOpen = true;
        isOpen = true;
        if (wasReconnect) {
          void import("../store/toasts").then(({ useToasts }) => {
            useToasts.getState().push({ kind: "info", text: "Reconnected to server" });
          });
        }
      };
      ws.onmessage = (m) => onEvent(JSON.parse(m.data) as WsEvent);
      ws.onclose = () => {
        if (closed) return;
        const wasOpen = isOpen;
        isOpen = false;
        if (wasOpen && hasBeenOpen) {
          void import("../store/toasts").then(({ useToasts }) => {
            useToasts.getState().push({
              kind: "error",
              text: "Disconnected from server, reconnecting…",
            });
          });
        }
        const jitter = 500 + Math.random() * 1000;
        reconnectTimer = setTimeout(connect, jitter);
      };
    };

    connect();
    return {
      close: () => {
        closed = true;
        if (reconnectTimer) clearTimeout(reconnectTimer);
        ws?.close();
      },
    };
  },
};
