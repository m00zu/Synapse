import { useMemo, useState } from "react";
import { useGraph } from "../../store/graph";

// Display order for known top-level categories. Anything else falls to
// "Other" at the end, sorted alphabetically.
const CATEGORY_ORDER = [
  "I/O", "Image", "Table", "Stats", "Plot",
  "Display", "Utility", "Collection", "Plugins", "Other",
];

// Bucket name used for classes whose identifier has no sub-segment — so they
// render as a flat list under the top-level category header instead of a
// nested collapsible group.
const ROOT_BUCKET = "__root__";

/** Prettify a sub-category segment for display.
 * 'image_process' → 'Image Process'; 'filter' → 'Filter'; etc. */
function prettify(s: string): string {
  if (!s) return s;
  return s
    .split(".")
    .map((part) =>
      part
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
    )
    .join(" · ");
}

export default function NodePalette() {
  const catalog = useGraph((s) => s.catalog);
  const categories = useGraph((s) => s.categories);
  const [filter, setFilter] = useState("");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  // Shape: grouped[category][subcategory] = [type, type, ...]
  // subcategory is ROOT_BUCKET for classes that live directly under category.
  const grouped = useMemo(() => {
    if (!catalog) return null;
    const out: Record<string, Record<string, string[]>> = {};
    for (const type of Object.keys(catalog)) {
      const info = categories?.[type];
      const cat = info?.category ?? "Other";
      const sub = info?.subcategory ? info.subcategory : ROOT_BUCKET;
      (out[cat] ??= {});
      (out[cat][sub] ??= []).push(type);
    }
    const displayName = (t: string) => categories?.[t]?.display_name ?? t;
    for (const cat of Object.keys(out)) {
      for (const sub of Object.keys(out[cat])) {
        out[cat][sub].sort((a, b) =>
          displayName(a).localeCompare(displayName(b))
        );
      }
    }
    return out;
  }, [catalog, categories]);

  if (!catalog || !grouped) {
    return <div className="p-2 text-xs text-fg/50">Loading nodes…</div>;
  }

  const normalizedFilter = filter.trim().toLowerCase();
  const displayName = (t: string) => categories?.[t]?.display_name ?? t;
  const matches = (t: string) => {
    if (!normalizedFilter) return true;
    return (
      displayName(t).toLowerCase().includes(normalizedFilter) ||
      t.toLowerCase().includes(normalizedFilter)
    );
  };

  const sortedCategories = Object.keys(grouped).sort((a, b) => {
    const ia = CATEGORY_ORDER.indexOf(a);
    const ib = CATEGORY_ORDER.indexOf(b);
    const oa = ia === -1 ? CATEGORY_ORDER.length : ia;
    const ob = ib === -1 ? CATEGORY_ORDER.length : ib;
    return oa - ob || a.localeCompare(b);
  });

  const totalCount = Object.values(grouped)
    .flatMap((subs) => Object.values(subs).flat())
    .length;

  return (
    <aside className="w-60 border-r border-border overflow-y-auto shrink-0 flex flex-col">
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
          const subs = grouped[cat];
          // Flatten for counting after filter.
          const visibleSubs = Object.entries(subs)
            .map(([sub, types]) => [sub, types.filter(matches)] as const)
            .filter(([, types]) => types.length > 0);
          if (visibleSubs.length === 0) return null;
          const catKey = `cat:${cat}`;
          const catCollapsed = collapsed[catKey];
          const catCount = visibleSubs.reduce((n, [, t]) => n + t.length, 0);
          return (
            <div key={cat} className="border-b border-border/50 last:border-b-0">
              <button
                type="button"
                onClick={() => setCollapsed({ ...collapsed, [catKey]: !catCollapsed })}
                className="w-full flex items-center justify-between px-3 py-1.5 text-xs uppercase tracking-wide text-fg/70 hover:bg-bg2 font-semibold"
              >
                <span>{catCollapsed ? "▸" : "▾"} {cat}</span>
                <span className="text-fg/40">{catCount}</span>
              </button>
              {!catCollapsed && visibleSubs.map(([sub, types]) => {
                // ROOT_BUCKET = flat nodes directly under this category.
                if (sub === ROOT_BUCKET) {
                  return (
                    <ul key={sub}>
                      {types.map((t) => (
                        <li
                          key={t}
                          draggable
                          onDragStart={(e) =>
                            e.dataTransfer.setData(
                              "application/x-synapse-node", t,
                            )
                          }
                          className="px-4 py-1 text-sm hover:bg-bg2 cursor-grab select-none text-fg/90"
                          title={t}
                        >
                          {displayName(t)}
                        </li>
                      ))}
                    </ul>
                  );
                }
                const subKey = `sub:${cat}:${sub}`;
                // Subgroups default to collapsed when there's a filter
                // to keep the view compact. If there's no filter, honor
                // the user's toggled state; default collapsed.
                const subCollapsed = normalizedFilter
                  ? false
                  : (collapsed[subKey] ?? false);
                return (
                  <div key={sub} className="border-t border-border/30">
                    <button
                      type="button"
                      onClick={() => setCollapsed({ ...collapsed, [subKey]: !subCollapsed })}
                      className="w-full flex items-center justify-between px-5 py-1 text-[11px] text-fg/60 hover:bg-bg2"
                    >
                      <span>{subCollapsed ? "▸" : "▾"} {prettify(sub)}</span>
                      <span className="text-fg/40">{types.length}</span>
                    </button>
                    {!subCollapsed && (
                      <ul>
                        {types.map((t) => (
                          <li
                            key={t}
                            draggable
                            onDragStart={(e) =>
                              e.dataTransfer.setData(
                                "application/x-synapse-node", t,
                              )
                            }
                            className="pl-7 pr-3 py-1 text-sm hover:bg-bg2 cursor-grab select-none text-fg/90"
                            title={t}
                          >
                            {displayName(t)}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
