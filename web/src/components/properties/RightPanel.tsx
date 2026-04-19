import { useState } from "react";
import PropertiesPanel from "./PropertiesPanel";
import ChatPanel from "../../ai/ChatPanel";

const TAB_KEY = "synapse-web.rightTab";

type Tab = "props" | "chat";

export default function RightPanel() {
  const [tab, setTab] = useState<Tab>(() => {
    try {
      const saved = localStorage.getItem(TAB_KEY);
      return saved === "chat" ? "chat" : "props";
    } catch { return "props"; }
  });
  const pick = (t: Tab) => {
    setTab(t);
    try { localStorage.setItem(TAB_KEY, t); } catch { /* ignore */ }
  };
  return (
    <aside className="w-96 border-l border-border shrink-0 flex flex-col">
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
