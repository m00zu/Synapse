import { useGraph } from "../store/graph";

export function usePreviewVersion(nodeId: string, port: string): number {
  return useGraph((s) => s.previewVersions[`${nodeId}:${port}`] ?? 0);
}
