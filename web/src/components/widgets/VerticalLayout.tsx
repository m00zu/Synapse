import type { VerticalLayoutSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";
import Renderer from "./Renderer";

export default function VerticalLayout(
  { spec, ctx }: { spec: VerticalLayoutSpec; ctx: WidgetContext }
) {
  return (
    <div className="flex flex-col gap-2">
      {spec.children.map((child, i) => (
        <Renderer key={i} spec={child} ctx={ctx} />
      ))}
    </div>
  );
}
