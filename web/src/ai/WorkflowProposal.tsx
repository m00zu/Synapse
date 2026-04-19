import type { BubbleState } from "./bubbleState";
import { useGraph } from "../store/graph";

export default function WorkflowProposal({ bubble }: { bubble: BubbleState }) {
  const wf = bubble.workflow!;
  const applyLocal = useGraph((s) => s.applyChatWorkflow);
  const discard = useGraph((s) => s.discardChatWorkflow);

  if (wf.state === "applied") {
    return <div className="text-green-400 text-xs">✓ Applied</div>;
  }
  if (wf.state === "discarded") {
    return <div className="text-fg/50 text-xs italic">Discarded</div>;
  }

  return (
    <div className="mt-1 border border-border/50 rounded p-2 text-xs bg-bg/50">
      <div className="text-fg/70 mb-2">
        Proposed: {wf.node_count} nodes, {wf.edge_count} edges
        <div className="text-fg/50 text-[10px] mt-1">
          {wf.preview_types.slice(0, 6).join(" → ")}
          {wf.preview_types.length > 6 ? " …" : ""}
        </div>
      </div>
      <div className="flex gap-2">
        <button
          onClick={() => applyLocal(bubble.bubble_id)}
          className="px-3 py-1 bg-green-700 text-white rounded"
        >
          Apply
        </button>
        <button
          onClick={() => discard(bubble.bubble_id)}
          className="px-3 py-1 bg-bg2 border border-border rounded"
        >
          Discard
        </button>
      </div>
    </div>
  );
}
