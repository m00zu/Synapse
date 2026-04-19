import { useMemo } from "react";
import { usePreviewVersion } from "../../hooks/usePreviewRefresh";
import { useGraph } from "../../store/graph";

export default function FigurePreview({ nodeId, port }: { nodeId: string; port: string }) {
  const version = usePreviewVersion(nodeId, port);
  const runActive = useGraph((s) => s.runActive);
  // Cache-bust via ?v= so the browser re-fetches after each node_finished.
  const url = useMemo(
    () =>
      `/api/files/preview/${encodeURIComponent(nodeId)}/${encodeURIComponent(port)}?v=${version}`,
    [nodeId, port, version]
  );
  if (version === 0 && !runActive) {
    return (
      <div className="text-xs text-fg/40 italic p-2">No preview yet — click Run.</div>
    );
  }
  return (
    <img
      src={url}
      alt={`${nodeId} / ${port}`}
      className="max-w-full max-h-64 rounded border border-border"
      onError={(e) => {
        (e.currentTarget as HTMLImageElement).style.display = "none";
      }}
    />
  );
}
