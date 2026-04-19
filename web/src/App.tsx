import { useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useGraph } from "./store/graph";
import { useWsEvents } from "./hooks/useWsEvents";
import NodePalette from "./components/palette/NodePalette";
import GraphCanvas from "./components/canvas/GraphCanvas";
import PropertiesPanel from "./components/properties/PropertiesPanel";
import Toolbar from "./components/toolbar/Toolbar";
import Toasts from "./components/toasts/Toasts";
import ErrorBoundary from "./components/ErrorBoundary";

export default function App() {
  const loadCatalog = useGraph((s) => s.loadCatalog);
  useEffect(() => { loadCatalog(); }, [loadCatalog]);
  useWsEvents();

  return (
    <ErrorBoundary>
      {/* ReactFlowProvider here (not just around GraphCanvas) lets useReactFlow
          work inside GraphCanvas — which we need for screenToFlowPosition
          so palette-drops land in flow coordinates (not viewport pixels). */}
      <ReactFlowProvider>
        <div className="flex flex-col h-screen">
          <Toolbar />
          <div className="flex flex-1 overflow-hidden">
            <NodePalette />
            <GraphCanvas />
            <PropertiesPanel />
          </div>
          <Toasts />
        </div>
      </ReactFlowProvider>
    </ErrorBoundary>
  );
}
