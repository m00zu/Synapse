import { useGraph } from "../../store/graph";
import Renderer, { type WidgetContext } from "../widgets/Renderer";

export default function PropertiesPanel() {
  const selectedId = useGraph((s) => s.selectedId);
  const nodes = useGraph((s) => s.nodes);
  const catalog = useGraph((s) => s.catalog);
  const patchProp = useGraph((s) => s.patchProp);
  const removeNode = useGraph((s) => s.removeNode);

  const node = nodes.find((n) => n.id === selectedId);
  if (!node || !catalog) {
    return (
      <aside className="w-80 border-l border-border p-3 text-xs text-fg/50 shrink-0">
        Select a node to edit its properties.
      </aside>
    );
  }

  const specs = catalog[node.type] ?? [];
  const ctx: WidgetContext = {
    nodeId: node.id,
    propValue: (p) => node.props[p],
    onChange: (p, v) => {
      patchProp(node.id, p, v).catch((err) =>
        console.error("patchProp failed:", err)
      );
    },
  };

  return (
    <aside className="w-80 border-l border-border overflow-y-auto shrink-0">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border sticky top-0 bg-bg">
        <div className="text-xs uppercase tracking-wide text-fg/60">{node.type}</div>
        <button
          onClick={() => {
            removeNode(node.id).catch((err) => console.error("removeNode failed:", err));
          }}
          className="text-xs text-red-400 hover:text-red-300"
          title="Delete node"
        >
          Delete
        </button>
      </div>
      <div className="flex flex-col gap-3 p-3">
        {specs.length === 0
          ? <div className="text-xs text-fg/50 italic">(no widgets for this node)</div>
          : specs.map((s, i) => (<Renderer key={i} spec={s} ctx={ctx} />))}
      </div>
    </aside>
  );
}
