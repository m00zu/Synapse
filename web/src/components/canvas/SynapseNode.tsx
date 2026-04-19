import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useGraph } from "../../store/graph";
import Renderer, { type WidgetContext } from "../widgets/Renderer";

export default function SynapseNode({ id, data, selected }: NodeProps) {
  const status = useGraph((s) => s.runStatus[id]);
  const catalog = useGraph((s) => s.catalog);
  const nodes = useGraph((s) => s.nodes);
  const categories = useGraph((s) => s.categories);
  const patchProp = useGraph((s) => s.patchProp);

  const type = (data as { type: string }).type;
  const display = categories?.[type]?.display_name ?? type;
  const node = nodes.find((n) => n.id === id);
  const specs = catalog?.[type] ?? [];

  const colorClass =
    status === "running" ? "border-amber-400 animate-pulse" :
    status === "ok"      ? "border-green-500" :
    status === "error"   ? "border-red-500" :
    selected             ? "border-accent" :
                           "border-border";

  const ctx: WidgetContext | null = node
    ? {
        nodeId: node.id,
        propValue: (p) => node.props[p],
        onChange: (p, v) => {
          patchProp(node.id, p, v).catch((e) =>
            console.error(`patchProp(${node.id}, ${p}) failed:`, e),
          );
        },
      }
    : null;

  return (
    <div className={`rounded border ${colorClass} bg-bg2 text-xs min-w-[220px]`}>
      <div className="px-3 py-2 border-b border-border/50 text-fg font-medium">
        {display}
      </div>
      {ctx && specs.length > 0 && (
        // `nodrag nowheel` lets clicks + scrolls inside form controls reach
        // the input itself rather than xyflow's drag/zoom layer.
        <div className="nodrag nowheel flex flex-col gap-2 p-3">
          {specs.map((s, i) => (
            <Renderer key={i} spec={s} ctx={ctx} />
          ))}
        </div>
      )}
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
