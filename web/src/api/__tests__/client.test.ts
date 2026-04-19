import { describe, it, expect, beforeAll, afterEach, afterAll } from "vitest";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { api } from "../client";

const server = setupServer(
  http.get("/api/nodes", () =>
    HttpResponse.json({
      GaussianBlurNode: [{ kind: "NumberField", prop: "sigma", label: "Sigma", min: 0, max: 20, step: 0.1, decimals: 1, default: 1, tab: "" }],
    })
  ),
  http.post("/api/graph/nodes", async ({ request }) => {
    const body = (await request.json()) as { type: string };
    return HttpResponse.json({ id: `n:${body.type}` }, { status: 201 });
  }),
  http.patch("/api/graph/nodes/:id/props", () =>
    HttpResponse.json({ ok: true })
  ),
  http.post("/api/exec/run", () =>
    HttpResponse.json({ run_id: "abc" }, { status: 202 })
  ),
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("api client", () => {
  it("getCatalog returns WidgetCatalog", async () => {
    const cat = await api.getCatalog();
    expect(cat.GaussianBlurNode[0].kind).toBe("NumberField");
  });

  it("addNode POSTs and returns the id", async () => {
    const res = await api.addNode({ type: "GaussianBlurNode" });
    expect(res.id).toBe("n:GaussianBlurNode");
  });

  it("patchProps PATCHes the right URL", async () => {
    const res = await api.patchProps("n1", { sigma: 2.5 });
    expect(res.ok).toBe(true);
  });

  it("runGraph returns run_id", async () => {
    const res = await api.runGraph();
    expect(res.run_id).toBe("abc");
  });
});
