import type { CustomSpec } from "../../api/types";
import type { WidgetContext } from "../widgets/Renderer";

const CHANNELS = [1, 2, 3, 4];  // 0 = pad/black, 1-4 = channels

export default function ChannelSelector(
  { spec, ctx }: { spec: CustomSpec; ctx: WidgetContext }
) {
  const prop = (spec.props.prop as string) || "channels";
  const label = (spec.props.label as string) || "Channels";
  const raw = ctx.propValue(prop);
  const value = (typeof raw === "string" ? raw : (spec.props.default as string) || "1,2,3");
  const selected = value.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));

  const toggle = (ch: number) => {
    const idx = selected.indexOf(ch);
    const next = idx === -1 ? [...selected, ch] : selected.filter((x) => x !== ch);
    // Cap at 3 channels (R, G, B slots).
    const capped = next.slice(-3);
    ctx.onChange(prop, capped.join(","));
  };

  return (
    <div className="flex flex-col gap-1 text-xs">
      <span className="text-fg/70">{label}</span>
      <div className="flex gap-1">
        {CHANNELS.map((ch) => {
          const active = selected.includes(ch);
          return (
            <button
              key={ch}
              type="button"
              onClick={() => toggle(ch)}
              className={`px-2 py-1 rounded border text-xs ${
                active ? "bg-accent text-bg border-accent" : "bg-bg2 text-fg border-border hover:bg-bg"
              }`}
            >
              {ch}
            </button>
          );
        })}
      </div>
      <div className="text-fg/50 font-mono">{selected.join(",") || "(none)"}</div>
    </div>
  );
}
