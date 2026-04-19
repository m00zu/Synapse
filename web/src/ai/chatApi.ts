// web/src/ai/chatApi.ts
import type { WsEvent } from "../api/types";
import type { BubbleState, ToolChip, WorkflowProposal } from "./bubbleState";
import { shortJson } from "./bubbleState";

/** Apply one WS event to an array of bubbles. Returns a new array
 * (immutable-style) for zustand friendliness. */
export function applyChatEvent(
  bubbles: BubbleState[],
  ev: WsEvent,
): BubbleState[] {
  if (!("bubble_id" in ev)) return bubbles;
  const id = (ev as { bubble_id: string }).bubble_id;

  if (ev.kind === "chat_turn_started") {
    // Append the user bubble + a fresh assistant bubble we'll stream into.
    return [
      ...bubbles,
      { bubble_id: `${id}-user`, role: "user", text: ev.user_text,
        chips: [], expanded_chips: new Set(), workflow: null, streaming: false },
      { bubble_id: id, role: "assistant", text: "",
        chips: [], expanded_chips: new Set(), workflow: null, streaming: true },
    ];
  }

  const idx = bubbles.findIndex((b) => b.bubble_id === id);
  if (idx < 0) return bubbles;
  const bubble = bubbles[idx];
  let next: BubbleState = bubble;

  switch (ev.kind) {
    case "chat_token":
      next = { ...bubble, text: bubble.text + ev.text };
      break;
    case "chat_tool_start": {
      const chip: ToolChip = {
        chip_id: ev.chip_id, name: ev.name,
        input_preview: shortJson(ev.input),
        status: "running", result_summary: "",
        full_input: ev.input, full_result: null,
      };
      next = { ...bubble, chips: [...bubble.chips, chip] };
      break;
    }
    case "chat_tool_finish": {
      next = {
        ...bubble,
        chips: bubble.chips.map((c) =>
          c.chip_id === ev.chip_id
            ? {
                ...c,
                status: ev.status,
                full_result: ev.result,
                result_summary:
                  ev.status === "error" && typeof ev.result === "object" &&
                  ev.result && "error" in ev.result
                    ? String((ev.result as { error: unknown }).error).slice(0, 80)
                    : Object.keys(ev.result).slice(0, 2).join(", "),
              }
            : c,
        ),
      };
      break;
    }
    case "chat_workflow_preview": {
      const r = ev.result as Record<string, unknown>;
      const proposal: WorkflowProposal = {
        node_count: Number(r.node_count ?? 0),
        edge_count: Number(r.edge_count ?? 0),
        preview_types: Array.isArray(r.preview_types)
          ? (r.preview_types as string[]) : [],
        state: r.canvas_was_empty ? "applied" : "pending",
        workflow: (r.workflow as Record<string, unknown>) ?? {},
      };
      next = { ...bubble, workflow: proposal };
      break;
    }
    case "chat_error":
      next = { ...bubble, role: "error", text: bubble.text + `\nerror: ${ev.error}`,
               streaming: false };
      break;
    case "chat_turn_cancelled":
    case "chat_turn_done":
      next = { ...bubble, streaming: false };
      break;
    default:
      return bubbles;
  }

  const updated = bubbles.slice();
  updated[idx] = next;
  return updated;
}
