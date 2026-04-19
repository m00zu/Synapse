import type { ProgressSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";

export default function Progress(
  { spec, ctx }: { spec: ProgressSpec; ctx: WidgetContext }
) {
  const raw = ctx.propValue(spec.prop);
  const value = typeof raw === "number" ? Math.max(0, Math.min(1, raw)) : 0;
  return (
    <div className="flex flex-col gap-1 text-xs">
      {spec.label ? <span className="text-fg/70">{spec.label}</span> : null}
      <progress
        className="w-full h-1.5"
        value={value}
        max={1}
      />
    </div>
  );
}
