import type { CustomSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";
import ChannelSelector from "../custom_nodes/channel_selector";
import PythonScriptEditor from "../custom_nodes/python_script_editor";
import DataTableCell from "../custom_nodes/data_table_cell";

type CustomComponent = React.ComponentType<{ spec: CustomSpec; ctx: WidgetContext }>;

const REGISTRY: Record<string, CustomComponent> = {
  channel_selector: ChannelSelector,
  python_script_editor: PythonScriptEditor,
  data_table_cell: DataTableCell,
};

export default function Custom({ spec, ctx }: { spec: CustomSpec; ctx: WidgetContext }) {
  const Component = REGISTRY[spec.component_id];
  if (Component) return <Component spec={spec} ctx={ctx} />;
  return (
    <div className="text-xs text-fg/50 italic border border-dashed border-border rounded p-2">
      Unknown custom component "{spec.component_id}"
    </div>
  );
}
