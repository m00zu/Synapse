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

  // `streaming` is driven by store state, not a local flag. The /api/chat/turn
  // POST returns almost immediately (it kicks off a daemon thread server-side),
  // so a local `busy` flag would clear before any token arrived — making the
  // Stop button useless. Watch the last assistant bubble instead.
  const streaming = bubbles.some((b) => b.role === "assistant" && b.streaming);

  useEffect(() => {
    api.listProviders().then((r) => setProviders(r.providers)).catch(() => {});
  }, []);

  const send = async () => {
    const t = text.trim();
    if (!t || streaming) return;
    setText("");
    try {
      await api.startChatTurn({ user_text: t, provider, model });
    } catch (e) {
      console.error("startChatTurn failed:", e);
    }
  };

  const stop = () => { api.stopChatTurn().catch(() => {}); };

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
            onClick={send} disabled={streaming || !text.trim()}
            className="px-3 py-1 bg-accent text-bg text-xs rounded disabled:opacity-50"
          >
            Send
          </button>
          <button
            onClick={stop}
            className="px-3 py-1 bg-red-700 text-white text-xs rounded disabled:opacity-50"
            disabled={!streaming}
          >
            Stop
          </button>
        </div>
      </div>
    </div>
  );
}
