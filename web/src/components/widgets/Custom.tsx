import type { CustomSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";

export default function Custom({ spec }: { spec: CustomSpec; ctx: WidgetContext }) {
  return (
    <div className="text-xs text-fg/50 italic border border-dashed border-border rounded p-2">
      Custom component "{spec.component_id}" — Phase 1d
    </div>
  );
}
