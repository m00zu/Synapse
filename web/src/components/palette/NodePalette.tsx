import { useGraph } from "../../store/graph";

export default function NodePalette() {
  const catalog = useGraph((s) => s.catalog);
  if (!catalog) {
    return <div className="p-2 text-xs text-fg/50">Loading nodes…</div>;
  }
  const types = Object.keys(catalog).sort();
  return (
    <aside className="w-56 border-r border-border overflow-y-auto shrink-0">
      <h2 className="px-3 py-2 text-xs uppercase tracking-wide text-fg/60 sticky top-0 bg-bg">
        Nodes ({types.length})
      </h2>
      <ul>
        {types.map((t) => (
          <li
            key={t}
            draggable
            onDragStart={(e) => e.dataTransfer.setData("application/x-synapse-node", t)}
            className="px-3 py-1.5 text-sm hover:bg-bg2 cursor-grab select-none"
          >
            {t}
          </li>
        ))}
      </ul>
    </aside>
  );
}
