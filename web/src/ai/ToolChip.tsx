import { useGraph } from "../store/graph";
import type { BubbleState, ToolChip as Chip } from "./bubbleState";

export default function ToolChip(
  { bubble, chip }: { bubble: BubbleState; chip: Chip }
) {
  const toggle = useGraph((s) => s.toggleChipExpanded);
  const expanded = bubble.expanded_chips.has(chip.chip_id);
  const glyph =
    chip.status === "running" ? "⋯" :
    chip.status === "ok"      ? "✓" :
                                 "⚠";
  return (
    <div>
      <button
        onClick={() => toggle(bubble.bubble_id, chip.chip_id)}
        className="px-2 py-0.5 text-[11px] bg-bg border border-border rounded hover:bg-bg2 flex items-center gap-1"
      >
        <span className="font-mono text-fg/60">🔧 {chip.name}</span>
        <span>→</span>
        <span>{glyph}</span>
        <span className="text-fg/60 truncate max-w-[140px]">
          {chip.result_summary || chip.input_preview}
        </span>
      </button>
      {expanded && (
        <pre className="mt-1 p-2 bg-bg border border-border/50 rounded text-[10px] whitespace-pre-wrap max-h-48 overflow-auto">
          <b>Input:</b> {JSON.stringify(chip.full_input, null, 2)}
          {chip.full_result && <>
            {"\n\n"}
            <b>Result:</b> {JSON.stringify(chip.full_result, null, 2)}
          </>}
        </pre>
      )}
    </div>
  );
}
