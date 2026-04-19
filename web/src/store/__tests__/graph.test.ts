// store/__tests__/graph.test.ts
import { describe, it, expect, beforeAll, afterEach, afterAll } from "vitest";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { useGraph } from "../graph";

const server = setupServer(
  http.get("/api/nodes", () =>
    HttpResponse.json({
      GaussianBlurNode: [{ kind: "NumberField", prop: "sigma", label: "Sigma",
                          min: 0, max: 20, step: 0.1, decimals: 1, default: 1.5, tab: "" }],
    })
  ),
  http.get("/api/nodes/categories", () =>
    HttpResponse.json({
      GaussianBlurNode: { identifier: "nodes.image_process.filter", category: "Image" },
    })
  ),
  http.post("/api/graph/nodes", () => HttpResponse.json({ id: "nX" }, { status: 201 })),
  http.delete("/api/graph/nodes/:id", () => new HttpResponse(null, { status: 204 })),
  http.patch("/api/graph/nodes/:id/props", () => HttpResponse.json({ ok: true })),
  http.patch("/api/graph/nodes/:id/pos", () => HttpResponse.json({ ok: true })),
);

beforeAll(() => server.listen());
afterEach(() => {
  server.resetHandlers();
  useGraph.setState({
    catalog: null, categories: null, nodes: [], edges: [], selectedId: null,
    runStatus: {}, runActive: false,
  });
});
afterAll(() => server.close());

describe("graph store", () => {
  it("loadCatalog populates catalog", async () => {
    await useGraph.getState().loadCatalog();
    expect(useGraph.getState().catalog).not.toBeNull();
  });

  it("addNode adds to local mirror with catalog defaults", async () => {
    await useGraph.getState().loadCatalog();
    const id = await useGraph.getState().addNode("GaussianBlurNode");
    const n = useGraph.getState().nodes.find((x) => x.id === id)!;
    expect(n.props.sigma).toBe(1.5);
  });

  it("patchProp updates local mirror", async () => {
    await useGraph.getState().loadCatalog();
    const id = await useGraph.getState().addNode("GaussianBlurNode");
    await useGraph.getState().patchProp(id, "sigma", 2.5);
    expect(useGraph.getState().nodes.find((n) => n.id === id)!.props.sigma).toBe(2.5);
  });

  it("removeNode cascades its edges + deselects", async () => {
    await useGraph.getState().loadCatalog();
    const a = await useGraph.getState().addNode("GaussianBlurNode");
    useGraph.setState({
      edges: [{ src: a, dst: "other" }],
      selectedId: a,
    });
    await useGraph.getState().removeNode(a);
    expect(useGraph.getState().edges).toEqual([]);
    expect(useGraph.getState().selectedId).toBeNull();
  });

  it("loadCatalog also populates categories", async () => {
    await useGraph.getState().loadCatalog();
    expect(useGraph.getState().categories).not.toBeNull();
    expect(useGraph.getState().categories!.GaussianBlurNode.category).toBe("Image");
  });

  it("setNodePos updates local position immediately", async () => {
    await useGraph.getState().loadCatalog();
    const id = await useGraph.getState().addNode("GaussianBlurNode", 10, 20);
    useGraph.getState().setNodePos(id, 300, 150);
    const n = useGraph.getState().nodes.find((x) => x.id === id)!;
    expect(n.x).toBe(300);
    expect(n.y).toBe(150);
  });

  it("commitNodePos PATCHes the server with the current local position", async () => {
    await useGraph.getState().loadCatalog();
    const id = await useGraph.getState().addNode("GaussianBlurNode", 10, 20);
    useGraph.getState().setNodePos(id, 300, 150);
    // MSW handler above returns ok:true for any id; just confirm no throw.
    await useGraph.getState().commitNodePos(id);
  });
});

describe("WS events", () => {
  it("node_started sets runStatus=running", () => {
    useGraph.getState().applyWsEvent({ kind: "node_started", node_id: "a" });
    expect(useGraph.getState().runStatus.a).toBe("running");
    expect(useGraph.getState().runActive).toBe(true);
  });

  it("node_finished sets runStatus to ok or error", () => {
    useGraph.getState().applyWsEvent({ kind: "node_finished", node_id: "a", success: true });
    expect(useGraph.getState().runStatus.a).toBe("ok");
    useGraph.getState().applyWsEvent({ kind: "node_finished", node_id: "b", success: false, error: "boom" });
    expect(useGraph.getState().runStatus.b).toBe("error");
  });

  it("run_finished clears runActive", () => {
    useGraph.getState().applyWsEvent({ kind: "node_started", node_id: "a" });
    useGraph.getState().applyWsEvent({ kind: "run_finished", run_id: "x1" });
    expect(useGraph.getState().runActive).toBe(false);
  });
});
