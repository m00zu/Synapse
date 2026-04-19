import { useEffect, useState } from "react";
import { api } from "../../api/client";

interface BrowseEntry { name: string; is_dir: boolean; path: string }

const LAST_DIR_KEY = "synapse-web.lastBrowseDir";

export default function ServerBrowseDialog({
  initialPath, onSelect, onClose,
}: {
  initialPath: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}) {
  // Resolve the starting directory in priority order:
  //   1. initialPath passed in (usually the parent of the current field value)
  //   2. last-browsed dir remembered in localStorage
  //   3. empty string → server resolves to $HOME / --allow-path
  // The server validates and 403s anything outside the allowed root, so a
  // stale localStorage entry from a previous `synapse serve --allow-path`
  // session can't escape the current root.
  const [root, setRoot] = useState(() => {
    if (initialPath) return initialPath;
    try {
      return localStorage.getItem(LAST_DIR_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [allowedRoot, setAllowedRoot] = useState<string>("");
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.browseDir(root)
      .then((r) => {
        setRoot(r.root);
        setAllowedRoot(r.allowed_root);
        setEntries(r.entries);
        setError(null);
        // Persist the actual resolved path — not the user's input — so the
        // next Browse opens right where they left off.
        try { localStorage.setItem(LAST_DIR_KEY, r.root); } catch { /* ignore */ }
      })
      .catch((e) => {
        setError(String(e));
        setEntries([]);
        // If the saved dir no longer exists or is outside the allowed root,
        // wipe it so the next open starts from $HOME instead of failing.
        try { localStorage.removeItem(LAST_DIR_KEY); } catch { /* ignore */ }
      });
  }, [root]);

  const goUp = () => {
    // Don't go above the allowed root — the server would 403 anyway.
    if (allowedRoot && root === allowedRoot) return;
    const parent = root.split("/").slice(0, -1).join("/") || "/";
    // If the computed parent would escape the allowed root, clamp to root.
    if (allowedRoot && !parent.startsWith(allowedRoot)) {
      setRoot(allowedRoot);
    } else {
      setRoot(parent);
    }
  };

  const atRoot = !!allowedRoot && root === allowedRoot;

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
          <div className="text-xs text-fg/60 font-mono truncate">{root || "(resolving…)"}</div>
          <button onClick={onClose}
                  className="text-xs text-fg/60 hover:text-fg">✕</button>
        </div>
        {error ? (
          <div className="text-xs text-red-400 p-2">{error}</div>
        ) : (
          <ul className="overflow-y-auto max-h-[50vh] text-sm">
            {!atRoot && (
              <li className="px-2 py-1 hover:bg-bg cursor-pointer"
                  onClick={goUp}>📁 ..</li>
            )}
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
