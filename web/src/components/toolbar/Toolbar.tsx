import { useState } from "react";
import { api } from "../../api/client";
import { useGraph } from "../../store/graph";

export default function Toolbar() {
  const nodeCount = useGraph((s) => s.nodes.length);
  const [status, setStatus] = useState<string>("");

  const onRun = async () => {
    try {
      setStatus("running…");
      const { run_id } = await api.runGraph();
      setStatus(`run ${run_id} started`);
    } catch (e) {
      setStatus(`error: ${e}`);
    }
  };

  const onStop = async () => {
    try {
      await api.stopGraph();
      setStatus("stop requested");
    } catch (e) {
      setStatus(`error: ${e}`);
    }
  };

  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
      <h1 className="text-sm font-semibold text-accent mr-auto">Synapse</h1>
      <div className="text-xs text-fg/50">{nodeCount} node(s)</div>
      <button
        onClick={onRun}
        className="px-3 py-1 bg-green-700 hover:bg-green-600 text-white text-xs rounded"
      >
        ▶ Run
      </button>
      <button
        onClick={onStop}
        className="px-3 py-1 bg-red-700 hover:bg-red-600 text-white text-xs rounded"
      >
        ⏹ Stop
      </button>
      <div className="text-xs text-fg/60 min-w-[120px]">{status}</div>
    </div>
  );
}
