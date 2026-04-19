import { useEffect, useState } from "react";
import { api } from "../../api/client";

interface BrowseEntry { name: string; is_dir: boolean; path: string }

export default function ServerBrowseDialog({
  initialPath, onSelect, onClose,
}: {
  initialPath: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}) {
  const [root, setRoot] = useState(initialPath || "/");
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.browseDir(root)
      .then((r) => { setRoot(r.root); setEntries(r.entries); setError(null); })
      .catch((e) => { setError(String(e)); setEntries([]); });
  }, [root]);

  const goUp = () => {
    const parent = root.split("/").slice(0, -1).join("/") || "/";
    setRoot(parent);
  };

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="w-[540px] max-h-[70vh] bg-bg2 border border-border rounded p-3 flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs text-fg/60 font-mono truncate">{root}</div>
          <button onClick={onClose}
                  className="text-xs text-fg/60 hover:text-fg">✕</button>
        </div>
        {error ? (
          <div className="text-xs text-red-400 p-2">{error}</div>
        ) : (
          <ul className="overflow-y-auto max-h-[50vh] text-sm">
            <li className="px-2 py-1 hover:bg-bg cursor-pointer"
                onClick={goUp}>📁 ..</li>
            {entries.map((e) => (
              <li key={e.path}
                  className="px-2 py-1 hover:bg-bg cursor-pointer flex justify-between"
                  onClick={() => {
                    if (e.is_dir) setRoot(e.path);
                    else { onSelect(e.path); onClose(); }
                  }}>
                <span>{e.is_dir ? "📁" : "📄"} {e.name}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
