import { useMemo, useState } from "react";
import { useGraph } from "../../store/graph";

// Display order for known categories; anything else falls to "Other" at the end.
const CATEGORY_ORDER = [
  "I/O", "Image", "Table", "Stats", "Plot",
  "Display", "Utility", "Collection", "Plugins", "Other",
];

export default function NodePalette() {
  const catalog = useGraph((s) => s.catalog);
  const categories = useGraph((s) => s.categories);
  const [filter, setFilter] = useState("");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const grouped = useMemo(() => {
    if (!catalog) return null;
    const groups: Record<string, string[]> = {};
    for (const type of Object.keys(catalog)) {
      const cat = categories?.[type]?.category ?? "Other";
      (groups[cat] ??= []).push(type);
    }
    for (const key of Object.keys(groups)) groups[key].sort();
    return groups;
  }, [catalog, categories]);

  if (!catalog || !grouped) {
    return <div className="p-2 text-xs text-fg/50">Loading nodes…</div>;
  }

  const normalizedFilter = filter.trim().toLowerCase();
  const matches = (t: string) =>
    !normalizedFilter || t.toLowerCase().includes(normalizedFilter);

  const sortedCategories = Object.keys(grouped).sort((a, b) => {
    const ia = CATEGORY_ORDER.indexOf(a);
    const ib = CATEGORY_ORDER.indexOf(b);
    const oa = ia === -1 ? CATEGORY_ORDER.length : ia;
    const ob = ib === -1 ? CATEGORY_ORDER.length : ib;
    return oa - ob || a.localeCompare(b);
  });

  const totalCount = Object.values(grouped).reduce((s, arr) => s + arr.length, 0);

  return (
    <aside className="w-56 border-r border-border overflow-y-auto shrink-0 flex flex-col">
      <div className="sticky top-0 bg-bg border-b border-border px-2 pt-2 pb-2 z-10">
        <h2 className="text-xs uppercase tracking-wide text-fg/60 mb-1 px-1">
          Nodes ({totalCount})
        </h2>
        <input
          type="text"
          placeholder="Filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="w-full bg-bg2 border border-border rounded px-2 py-1 text-xs text-fg focus:outline-none focus:border-accent"
        />
      </div>
      <div className="flex-1">
        {sortedCategories.map((cat) => {
          const types = grouped[cat].filter(matches);
          if (types.length === 0) return null;
          const isCollapsed = collapsed[cat];
          return (
            <div key={cat} className="border-b border-border/50 last:border-b-0">
              <button
                type="button"
                onClick={() => setCollapsed({ ...collapsed, [cat]: !isCollapsed })}
                className="w-full flex items-center justify-between px-3 py-1.5 text-xs uppercase tracking-wide text-fg/70 hover:bg-bg2"
              >
                <span>{isCollapsed ? "▸" : "▾"} {cat}</span>
                <span className="text-fg/40">{types.length}</span>
              </button>
              {!isCollapsed && (
                <ul>
                  {types.map((t) => (
                    <li
                      key={t}
                      draggable
                      onDragStart={(e) =>
                        e.dataTransfer.setData("application/x-synapse-node", t)
                      }
                      className="px-4 py-1 text-sm hover:bg-bg2 cursor-grab select-none text-fg/90"
                    >
                      {t}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
