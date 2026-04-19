import type { FilePathSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";

// Task 6 upgrades this with upload + server-browse dialog buttons.
export default function FilePath(
  { spec, ctx }: { spec: FilePathSpec; ctx: WidgetContext }
) {
  const raw = ctx.propValue(spec.prop);
  const value = (typeof raw === "string" ? raw : spec.default);
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-fg/70">{spec.label}</span>
      <input
        type="text"
        className="bg-bg border border-border rounded px-2 py-1 text-fg font-mono"
        value={value}
        onChange={(e) => ctx.onChange(spec.prop, e.target.value)}
      />
    </label>
  );
}
