import type { ComboBoxSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";

export default function ComboBox(
  { spec, ctx }: { spec: ComboBoxSpec; ctx: WidgetContext }
) {
  const raw = ctx.propValue(spec.prop);
  const value = (typeof raw === "string" ? raw : spec.default) ?? "";
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-fg/70">{spec.label}</span>
      <select
        className="bg-bg border border-border rounded px-2 py-1 text-fg"
        value={value}
        onChange={(e) => ctx.onChange(spec.prop, e.target.value)}
      >
        {spec.options.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    </label>
  );
}
