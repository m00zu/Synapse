import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ChatProvider } from "../api/types";

interface Props {
  providers: ChatProvider[];
  provider: string;
  model: string;
  onProvider: (p: string) => void;
  onModel: (m: string) => void;
  onKeySaved?: () => void;
}

export default function ProviderSelect({
  providers, provider, model, onProvider, onModel, onKeySaved,
}: Props) {
  const [models, setModels] = useState<string[]>([]);
  const [keyInput, setKeyInput] = useState("");
  const [keyEditing, setKeyEditing] = useState(false);

  const loadModels = (p: string) =>
    api.listModels(p)
      .then((r) => {
        setModels(r.models);
        // If the current selection isn't in the list, pick the first entry so
        // the turn request sends a valid model name — "gemma3:12b" is the
        // Ollama default but doesn't exist on Claude/OpenAI/etc.
        if (r.models.length && !r.models.includes(model)) {
          onModel(r.models[0]);
        }
      })
      .catch(() => setModels([]));

  useEffect(() => { loadModels(provider); }, [provider]);

  const providerInfo = providers.find((p) => p.name === provider);
  const needsKey = providerInfo && !providerInfo.has_key &&
                   !["Ollama", "Ollama Cloud"].includes(provider);

  const saveKey = async () => {
    await api.saveProviderKey(provider, keyInput);
    setKeyInput(""); setKeyEditing(false);
    // Refresh models now that the key is stored — most providers reject
    // /models without a valid key, so the list was empty before save.
    await loadModels(provider);
    onKeySaved?.();
  };

  return (
    <div className="border-b border-border px-2 py-2 flex flex-col gap-1 text-xs">
      <div className="flex gap-1">
        <select
          value={provider}
          onChange={(e) => onProvider(e.target.value)}
          className="flex-1 bg-bg border border-border rounded px-2 py-1 text-fg"
        >
          {providers.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}{p.has_key || ["Ollama", "Ollama Cloud"].includes(p.name) ? "" : " 🔑"}
            </option>
          ))}
        </select>
        <select
          value={model}
          onChange={(e) => onModel(e.target.value)}
          className="flex-1 bg-bg border border-border rounded px-2 py-1 text-fg"
        >
          {models.length === 0 ? (
            <option value={model}>{model}</option>
          ) : (
            models.map((m) => (<option key={m} value={m}>{m}</option>))
          )}
        </select>
      </div>
      {(needsKey || keyEditing) && (
        <div className="flex gap-1">
          <input
            type="password"
            value={keyInput}
            onChange={(e) => setKeyInput(e.target.value)}
            placeholder={`${provider} API key`}
            className="flex-1 bg-bg border border-border rounded px-2 py-1 text-fg font-mono"
          />
          <button
            onClick={saveKey}
            className="px-2 py-1 bg-accent text-bg rounded"
          >
            Save
          </button>
        </div>
      )}
      {providerInfo && !needsKey && !keyEditing && (
        <button
          onClick={() => setKeyEditing(true)}
          className="self-start text-[10px] text-fg/40 hover:text-fg/70"
        >
          Update key
        </button>
      )}
    </div>
  );
}
