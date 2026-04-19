import type { CheckBoxSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";

export default function CheckBox(
  { spec, ctx }: { spec: CheckBoxSpec; ctx: WidgetContext }
) {
  const raw = ctx.propValue(spec.prop);
  const value = (typeof raw === "boolean" ? raw : spec.default);
  return (
    <label className="flex items-center gap-2 text-xs text-fg/90">
      <input
        type="checkbox"
        className="accent-accent"
        checked={value}
        onChange={(e) => ctx.onChange(spec.prop, e.target.checked)}
      />
      <span>{spec.label}</span>
    </label>
  );
}
