import { Handle, Position, type NodeProps } from "@xyflow/react";

export default function SynapseNode({ data, selected }: NodeProps) {
  const borderColor = selected ? "border-accent" : "border-border";
  return (
    <div className={`rounded border ${borderColor} bg-bg2 px-3 py-2 text-xs min-w-[120px]`}>
      <div className="text-fg font-medium">{(data as { type: string }).type}</div>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
