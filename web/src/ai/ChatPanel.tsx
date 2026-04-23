// web/src/ai/ChatPanel.tsx
import { useState, useEffect } from "react";
import { api } from "../api/client";
import { useGraph } from "../store/graph";
import BubbleLog from "./BubbleLog";
import ProviderSelect from "./providerSelect";
import type { ChatProvider } from "../api/types";

const PROVIDER_KEY = "synapse-web.chatProvider";
const MODEL_KEY = "synapse-web.chatModel";

const readLS = (key: string, fallback: string): string => {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
};
const writeLS = (key: string, value: string) => {
  try { localStorage.setItem(key, value); } catch { /* ignore */ }
};

export default function ChatPanel() {
  const bubbles = useGraph((s) => s.chatBubbles);
  const [text, setText] = useState("");
  const [providers, setProviders] = useState<ChatProvider[]>([]);
  // Persist provider/model so switching the Properties/Chat tab (which
  // remounts this panel) doesn't silently reset the selection back to the
  // Ollama default — that's how turns end up misrouted to a local model.
  const [provider, setProviderState] = useState(() => readLS(PROVIDER_KEY, "Ollama"));
  const [model, setModelState] = useState(() => readLS(MODEL_KEY, "gemma3:12b"));
  const setProvider = (p: string) => { setProviderState(p); writeLS(PROVIDER_KEY, p); };
  const setModel = (m: string) => { setModelState(m); writeLS(MODEL_KEY, m); };

  // `streaming` is driven by store state, not a local flag. The /api/chat/turn
  // POST returns almost immediately (it kicks off a daemon thread server-side),
  // so a local `busy` flag would clear before any token arrived — making the
  // Stop button useless. Watch the last assistant bubble instead.
  const streaming = bubbles.some((b) => b.role === "assistant" && b.streaming);

  const refreshProviders = () =>
    api.listProviders().then((r) => setProviders(r.providers)).catch(() => {});
  useEffect(() => { refreshProviders(); }, []);

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
        onKeySaved={refreshProviders}
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
