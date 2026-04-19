import type { HorizontalLayoutSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";
import Renderer from "./Renderer";

export default function HorizontalLayout(
  { spec, ctx }: { spec: HorizontalLayoutSpec; ctx: WidgetContext }
) {
  return (
    <div className="flex flex-row gap-2">
      {spec.children.map((child, i) => (
        <Renderer key={i} spec={child} ctx={ctx} />
      ))}
    </div>
  );
}
