import { create } from "zustand";

export interface Toast { id: string; kind: "info" | "error"; text: string; }

interface ToastState {
  toasts: Toast[];
  push: (t: Omit<Toast, "id">) => void;
  dismiss: (id: string) => void;
}

export const useToasts = create<ToastState>((set) => ({
  toasts: [],
  push: (t) => {
    const id = crypto.randomUUID();
    set((s) => ({ toasts: [...s.toasts, { id, ...t }] }));
    // Auto-dismiss after 5s.
    setTimeout(
      () => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })),
      5000,
    );
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })),
}));

/** Convenience: push an error toast without importing the hook. */
export function pushError(text: string) {
  useToasts.getState().push({ kind: "error", text });
}
