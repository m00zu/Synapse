import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useGraph } from "../../store/graph";
import Renderer, { type WidgetContext } from "../widgets/Renderer";
import { portColor } from "./portColors";

export default function SynapseNode({ id, data, selected }: NodeProps) {
  const status = useGraph((s) => s.runStatus[id]);
  const catalog = useGraph((s) => s.catalog);
  const nodes = useGraph((s) => s.nodes);
  const categories = useGraph((s) => s.categories);
  const patchProp = useGraph((s) => s.patchProp);

  const type = (data as { type: string }).type;
  const info = categories?.[type];
  const display = info?.display_name ?? type;
  const inputs = info?.inputs ?? [];
  const outputs = info?.outputs ?? [];
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

  // Match xyflow's default handle size (10px) visually via a 10x10 circle.
  const handleStyle = (color: string) => ({
    background: color,
    width: 10,
    height: 10,
    border: "1.5px solid rgba(0,0,0,0.5)",
  });

  return (
    <div className={`rounded border ${colorClass} bg-bg2 text-xs min-w-[240px]`}>
      <div className="px-3 py-2 border-b border-border/50 text-fg font-medium">
        {display}
      </div>

      {/* Ports: two columns, inputs left + outputs right. Each row has its
          own Handle anchored to the row's edge so multi-port nodes
          (SplitRGB, outlier detection, etc.) get one visible port per row. */}
      {(inputs.length > 0 || outputs.length > 0) && (
        <div className="flex py-1">
          <div className="flex-1 flex flex-col">
            {inputs.map((p) => (
              <div key={p.name} className="relative pl-4 pr-2 py-1 text-fg/80">
                <Handle
                  id={p.name}
                  type="target"
                  position={Position.Left}
                  style={handleStyle(portColor(p.type))}
                />
                <span>{p.name}</span>
              </div>
            ))}
          </div>
          <div className="flex-1 flex flex-col items-end">
            {outputs.map((p) => (
              <div key={p.name} className="relative pr-4 pl-2 py-1 text-fg/80">
                <span>{p.name}</span>
                <Handle
                  id={p.name}
                  type="source"
                  position={Position.Right}
                  style={handleStyle(portColor(p.type))}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {ctx && specs.length > 0 && (
        // `nodrag nowheel` lets clicks + scrolls inside form controls reach
        // the input itself rather than xyflow's drag/zoom layer.
        <div className="nodrag nowheel flex flex-col gap-2 p-3 border-t border-border/50">
          {specs.map((s, i) => (
            <Renderer key={i} spec={s} ctx={ctx} />
          ))}
        </div>
      )}
    </div>
  );
}
