// data_table_cell.tsx (Task 4 stub; Task 6 replaces)
import type { CustomSpec } from "../../api/types";
import type { WidgetContext } from "../widgets/Renderer";
export default function DataTableCell(
  { spec: _spec, ctx: _ctx }: { spec: CustomSpec; ctx: WidgetContext }
) {
  return <div className="text-xs text-fg/50 italic p-2">Data table cell (Task 6)</div>;
}
