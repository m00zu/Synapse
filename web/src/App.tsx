import { useEffect } from "react";
import { useGraph } from "./store/graph";
import NodePalette from "./components/palette/NodePalette";
import GraphCanvas from "./components/canvas/GraphCanvas";
import PropertiesPanel from "./components/properties/PropertiesPanel";
import Toolbar from "./components/toolbar/Toolbar";

export default function App() {
  const loadCatalog = useGraph((s) => s.loadCatalog);
  useEffect(() => { loadCatalog(); }, [loadCatalog]);

  return (
    <div className="flex flex-col h-screen">
      <Toolbar />
      <div className="flex flex-1 overflow-hidden">
        <NodePalette />
        <GraphCanvas />
        <PropertiesPanel />
      </div>
    </div>
  );
}
