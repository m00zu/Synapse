import type { PreviewSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";

export default function Preview({ spec }: { spec: PreviewSpec; ctx: WidgetContext }) {
  return (
    <div className="text-xs text-fg/50 italic border border-dashed border-border rounded p-2">
      Preview ({spec.preview_kind}) — Phase 1d
    </div>
  );
}
