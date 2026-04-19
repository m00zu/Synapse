import type { CustomSpec } from "../../api/types";
import type { WidgetContext } from "../widgets/Renderer";
import TablePreview from "../preview/TablePreview";

export default function DataTableCell(
  { spec: _spec, ctx }: { spec: CustomSpec; ctx: WidgetContext }
) {
  // DataTableCellNode displays its table input. The generic Preview widget
  // at the output port serves the same purpose; this just wraps it wider.
  return (
    <div className="min-w-[360px]">
      <TablePreview nodeId={ctx.nodeId} port="out" />
    </div>
  );
}
