// web/src/ai/__tests__/bubbleState.test.ts
import { describe, it, expect } from "vitest";
import { shortJson } from "../bubbleState";
import { applyChatEvent } from "../chatApi";
import type { BubbleState } from "../bubbleState";

describe("shortJson", () => {
  it("returns short input unchanged", () => {
    expect(shortJson({ a: 1 })).toBe('{"a":1}');
  });
  it("truncates long input", () => {
    const s = shortJson({ x: "a".repeat(200) });
    expect(s.endsWith("…")).toBe(true);
    expect(s.length).toBe(80);
  });
});

describe("applyChatEvent", () => {
  const base: BubbleState[] = [];

  it("chat_turn_started appends user + empty assistant bubbles", () => {
    const after = applyChatEvent(base, {
      kind: "chat_turn_started", bubble_id: "b1",
      turn_id: "t1", user_text: "hi",
    });
    expect(after).toHaveLength(2);
    expect(after[0].role).toBe("user");
    expect(after[0].text).toBe("hi");
    expect(after[1].role).toBe("assistant");
    expect(after[1].streaming).toBe(true);
  });

  it("chat_token appends to the assistant bubble's text", () => {
    let s = applyChatEvent(base, {
      kind: "chat_turn_started", bubble_id: "b1", turn_id: "t1", user_text: "hi",
    });
    s = applyChatEvent(s, { kind: "chat_token", bubble_id: "b1", text: "Hel" });
    s = applyChatEvent(s, { kind: "chat_token", bubble_id: "b1", text: "lo" });
    expect(s[1].text).toBe("Hello");
  });

  it("chat_tool_start/finish flips chip status", () => {
    let s = applyChatEvent(base, {
      kind: "chat_turn_started", bubble_id: "b1", turn_id: "t1", user_text: "q",
    });
    s = applyChatEvent(s, {
      kind: "chat_tool_start", bubble_id: "b1", chip_id: "c1",
      name: "inspect_canvas", input: {},
    });
    expect(s[1].chips[0].status).toBe("running");
    s = applyChatEvent(s, {
      kind: "chat_tool_finish", bubble_id: "b1", chip_id: "c1",
      status: "ok", result: { nodes: 3 },
    });
    expect(s[1].chips[0].status).toBe("ok");
    expect(s[1].chips[0].result_summary).toBe("nodes");
  });

  it("chat_turn_done clears streaming", () => {
    let s = applyChatEvent(base, {
      kind: "chat_turn_started", bubble_id: "b1", turn_id: "t1", user_text: "q",
    });
    s = applyChatEvent(s, { kind: "chat_turn_done", bubble_id: "b1" });
    expect(s[1].streaming).toBe(false);
  });
});
