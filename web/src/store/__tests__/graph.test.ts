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
  http.post("/api/graph/nodes", () => HttpResponse.json({ id: "nX" }, { status: 201 })),
  http.delete("/api/graph/nodes/:id", () => new HttpResponse(null, { status: 204 })),
  http.patch("/api/graph/nodes/:id/props", () => HttpResponse.json({ ok: true })),
);

beforeAll(() => server.listen());
afterEach(() => {
  server.resetHandlers();
  useGraph.setState({ catalog: null, nodes: [], edges: [], selectedId: null });
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
});
