import { useCallback, useEffect, useRef, useState } from "react";
import PropertiesPanel from "./PropertiesPanel";
import ChatPanel from "../../ai/ChatPanel";

const TAB_KEY = "synapse-web.rightTab";
const WIDTH_KEY = "synapse-web.rightWidth";
const MIN_W = 240;
const MAX_W = 900;
const DEFAULT_W = 384;

type Tab = "props" | "chat";

export default function RightPanel() {
  const [tab, setTab] = useState<Tab>(() => {
    try {
      const saved = localStorage.getItem(TAB_KEY);
      return saved === "chat" ? "chat" : "props";
    } catch { return "props"; }
  });
  const [width, setWidth] = useState<number>(() => {
    try {
      const saved = parseInt(localStorage.getItem(WIDTH_KEY) ?? "", 10);
      if (Number.isFinite(saved)) return Math.min(MAX_W, Math.max(MIN_W, saved));
    } catch { /* ignore */ }
    return DEFAULT_W;
  });
  const dragState = useRef<{ startX: number; startW: number } | null>(null);

  const pick = (t: Tab) => {
    setTab(t);
    try { localStorage.setItem(TAB_KEY, t); } catch { /* ignore */ }
  };

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragState.current = { startX: e.clientX, startW: width };
  }, [width]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = dragState.current;
      if (!d) return;
      // Panel is on the right; drag left = wider.
      const next = Math.min(MAX_W, Math.max(MIN_W, d.startW - (e.clientX - d.startX)));
      setWidth(next);
    };
    const onUp = () => {
      if (dragState.current) {
        try { localStorage.setItem(WIDTH_KEY, String(width)); } catch { /* ignore */ }
      }
      dragState.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [width]);

  return (
    <aside
      style={{ width }}
      className="border-l border-border shrink-0 flex flex-col relative"
    >
      <div
        onMouseDown={onMouseDown}
        title="Drag to resize"
        className="absolute left-0 top-0 bottom-0 w-1 -translate-x-1/2 cursor-col-resize hover:bg-accent/40 z-10"
      />
      <div className="flex border-b border-border">
        <button
          onClick={() => pick("props")}
          className={`flex-1 text-xs py-2 ${tab === "props"
            ? "bg-bg2 text-fg font-semibold" : "text-fg/60 hover:bg-bg2"}`}>
          Properties
        </button>
        <button
          onClick={() => pick("chat")}
          className={`flex-1 text-xs py-2 ${tab === "chat"
            ? "bg-bg2 text-fg font-semibold" : "text-fg/60 hover:bg-bg2"}`}>
          AI Chat
        </button>
      </div>
      <div className="flex-1 overflow-hidden">
        {tab === "props" ? <PropertiesPanel /> : <ChatPanel />}
      </div>
    </aside>
  );
}
