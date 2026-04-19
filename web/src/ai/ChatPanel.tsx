// web/src/ai/ChatPanel.tsx
import { useState, useEffect } from "react";
import { api } from "../api/client";
import { useGraph } from "../store/graph";
import BubbleLog from "./BubbleLog";
import ProviderSelect from "./providerSelect";
import type { ChatProvider } from "../api/types";

export default function ChatPanel() {
  const bubbles = useGraph((s) => s.chatBubbles);
  const [text, setText] = useState("");
  const [providers, setProviders] = useState<ChatProvider[]>([]);
  const [provider, setProvider] = useState("Ollama");
  const [model, setModel] = useState("gemma3:12b");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.listProviders().then((r) => setProviders(r.providers)).catch(() => {});
  }, []);

  const send = async () => {
    const t = text.trim();
    if (!t || busy) return;
    setText(""); setBusy(true);
    try {
      await api.startChatTurn({ user_text: t, provider, model });
    } finally {
      setBusy(false);
    }
  };

  const stop = () => { api.stopChatTurn(); };

  return (
    <div className="flex flex-col h-full">
      <ProviderSelect
        providers={providers}
        provider={provider}
        model={model}
        onProvider={setProvider}
        onModel={setModel}
      />
      <BubbleLog bubbles={bubbles} />
      <div className="border-t border-border p-2 flex gap-1">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask the AI…"
          className="flex-1 bg-bg border border-border rounded px-2 py-1 text-xs text-fg resize-none"
          rows={2}
        />
        <div className="flex flex-col gap-1">
          <button
            onClick={send} disabled={busy || !text.trim()}
            className="px-3 py-1 bg-accent text-bg text-xs rounded disabled:opacity-50"
          >
            Send
          </button>
          <button
            onClick={stop}
            className="px-3 py-1 bg-red-700 text-white text-xs rounded disabled:opacity-50"
            disabled={!busy}
          >
            Stop
          </button>
        </div>
      </div>
    </div>
  );
}
