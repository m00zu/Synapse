import { describe, it, expect, beforeEach, vi } from "vitest";
import { useToasts, pushError } from "../toasts";

describe("toasts store", () => {
  beforeEach(() => {
    useToasts.setState({ toasts: [] });
    vi.useFakeTimers();
  });

  it("push adds a toast with a generated id", () => {
    useToasts.getState().push({ kind: "error", text: "boom" });
    const toasts = useToasts.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].text).toBe("boom");
    expect(toasts[0].kind).toBe("error");
    expect(toasts[0].id).toBeTruthy();
  });

  it("pushError is a shortcut for an error-kind toast", () => {
    pushError("API down");
    expect(useToasts.getState().toasts[0]).toMatchObject({ kind: "error", text: "API down" });
  });

  it("dismiss removes the toast by id", () => {
    useToasts.getState().push({ kind: "info", text: "hi" });
    const id = useToasts.getState().toasts[0].id;
    useToasts.getState().dismiss(id);
    expect(useToasts.getState().toasts).toEqual([]);
  });

  it("toasts auto-dismiss after 5s", () => {
    useToasts.getState().push({ kind: "info", text: "auto" });
    expect(useToasts.getState().toasts).toHaveLength(1);
    vi.advanceTimersByTime(5100);
    expect(useToasts.getState().toasts).toEqual([]);
  });
});
