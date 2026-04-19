// web/src/ai/bubbleState.ts
export type ChipStatus = "running" | "ok" | "error";

export interface ToolChip {
  chip_id: string;
  name: string;
  input_preview: string;
  status: ChipStatus;
  result_summary: string;
  full_input: Record<string, unknown>;
  full_result: Record<string, unknown> | null;
}

export interface WorkflowProposal {
  node_count: number;
  edge_count: number;
  preview_types: string[];
  state: "pending" | "applied" | "discarded";
  workflow: Record<string, unknown>; // raw JSON for applying
}

export type BubbleRole = "user" | "assistant" | "system" | "error";

export interface BubbleState {
  bubble_id: string;
  role: BubbleRole;
  text: string;
  chips: ToolChip[];
  expanded_chips: Set<string>;
  workflow: WorkflowProposal | null;
  streaming: boolean;
}

/** Shorten dict JSON for chip preview labels (~80 chars). */
export function shortJson(obj: unknown, maxLen = 80): string {
  const s = JSON.stringify(obj);
  return s.length <= maxLen ? s : s.slice(0, maxLen - 1) + "…";
}
