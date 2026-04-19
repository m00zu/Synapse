import type { ButtonSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";

export default function Button(
  { spec }: { spec: ButtonSpec; ctx: WidgetContext }
) {
  // Phase 1c: actions aren't wired to a backend endpoint yet; log the click
  // so the UI is responsive. Phase 1d will add /api/graph/nodes/{id}/action/{action}.
  return (
    <button
      className="px-2 py-1 bg-bg2 border border-border rounded hover:bg-bg text-xs"
      onClick={() => console.log(`[action] ${spec.action}`)}
    >
      {spec.label}
    </button>
  );
}
