// web/src/ai/BubbleLog.tsx
import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { BubbleState } from "./bubbleState";
import ToolChip from "./ToolChip";
import WorkflowProposal from "./WorkflowProposal";

export default function BubbleLog({ bubbles }: { bubbles: BubbleState[] }) {
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    // Stay pinned to bottom when new tokens land.
    el.scrollTop = el.scrollHeight;
  }, [bubbles]);

  return (
    <div ref={scroller} className="flex-1 overflow-y-auto p-2 flex flex-col gap-2">
      {bubbles.map((b) => (
        <BubbleView key={b.bubble_id} bubble={b} />
      ))}
    </div>
  );
}

function BubbleView({ bubble }: { bubble: BubbleState }) {
  if (bubble.role === "user") {
    return (
      <div className="self-end max-w-[85%] bg-accent text-bg rounded px-3 py-1.5 text-sm whitespace-pre-wrap">
        {bubble.text}
      </div>
    );
  }
  if (bubble.role === "error") {
    return (
      <div className="self-start max-w-[85%] bg-red-900/60 text-red-100 rounded border border-red-500 px-3 py-1.5 text-xs whitespace-pre-wrap">
        {bubble.text || "error"}
      </div>
    );
  }
  // assistant
  return (
    <div className="self-start max-w-[90%] bg-bg2 border border-border rounded px-3 py-2 text-xs flex flex-col gap-2">
      {bubble.chips.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {bubble.chips.map((c) => (
            <ToolChip key={c.chip_id} bubble={bubble} chip={c} />
          ))}
        </div>
      )}
      <div className="text-fg/90">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {bubble.text || (bubble.streaming ? "" : "(no reply)")}
        </ReactMarkdown>
        {bubble.streaming && (
          <span className="inline-block w-1.5 h-3 bg-accent align-middle animate-pulse ml-1" />
        )}
      </div>
      {bubble.workflow && <WorkflowProposal bubble={bubble} />}
    </div>
  );
}
