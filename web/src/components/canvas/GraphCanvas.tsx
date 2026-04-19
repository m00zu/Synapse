import {
  ReactFlow, Background, Controls,
  type Connection, type NodeMouseHandler,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback } from "react";
import { useGraph } from "../../store/graph";
import SynapseNode from "./SynapseNode";

const nodeTypes = { synapse: SynapseNode };

export default function GraphCanvas() {
  const nodes = useGraph((s) => s.nodes);
  const edges = useGraph((s) => s.edges);
  const addNode = useGraph((s) => s.addNode);
  const addEdge = useGraph((s) => s.addEdge);
  const select = useGraph((s) => s.select);
  const selectedId = useGraph((s) => s.selectedId);

  const rfNodes = nodes.map((n) => ({
    id: n.id,
    position: { x: n.x, y: n.y },
    type: "synapse" as const,
    data: { type: n.type },
    selected: n.id === selectedId,
  }));
  const rfEdges = edges.map((e, i) => ({
    id: `e${i}:${e.src}-${e.dst}`,
    source: e.src,
    target: e.dst,
  }));

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    const type = event.dataTransfer.getData("application/x-synapse-node");
    if (!type) return;
    const bounds = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    addNode(type, x, y).catch((err) => console.error("addNode failed:", err));
  }, [addNode]);

  const onConnect = useCallback((p: Connection) => {
    if (!p.source || !p.target) return;
    addEdge({ src: p.source, dst: p.target }).catch((err) =>
      console.error("addEdge failed:", err)
    );
  }, [addEdge]);

  const onNodeClick: NodeMouseHandler = useCallback((_, node) => {
    select(node.id);
  }, [select]);

  return (
    <div
      className="flex-1 h-full"
      onDrop={onDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        onPaneClick={() => select(null)}
        onConnect={onConnect}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}
