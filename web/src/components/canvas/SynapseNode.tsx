import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useGraph } from "../../store/graph";

export default function SynapseNode({ id, data, selected }: NodeProps) {
  const status = useGraph((s) => s.runStatus[id]);
  const colorClass =
    status === "running" ? "border-amber-400 animate-pulse" :
    status === "ok"      ? "border-green-500" :
    status === "error"   ? "border-red-500" :
    selected             ? "border-accent" :
                           "border-border";
  return (
    <div className={`rounded border ${colorClass} bg-bg2 px-3 py-2 text-xs min-w-[120px]`}>
      <div className="text-fg font-medium">{(data as { type: string }).type}</div>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
