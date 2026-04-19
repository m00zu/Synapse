import type { NumberFieldSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";

export default function NumberField(
  { spec, ctx }: { spec: NumberFieldSpec; ctx: WidgetContext }
) {
  const raw = ctx.propValue(spec.prop);
  const value = (typeof raw === "number" ? raw : spec.default);
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-fg/70">{spec.label}</span>
      <input
        type="number"
        className="bg-bg border border-border rounded px-2 py-1 text-fg"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        value={value}
        onChange={(e) => ctx.onChange(spec.prop, Number(e.target.value))}
      />
    </label>
  );
}
