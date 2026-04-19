import type { WidgetSpec } from "../../api/types";
import NumberField from "./NumberField";
import ComboBox from "./ComboBox";
import CheckBox from "./CheckBox";
import TextField from "./TextField";
import FilePath from "./FilePath";
import Button from "./Button";
import Progress from "./Progress";
import Preview from "./Preview";
import CustomW from "./Custom";
import VerticalLayout from "./VerticalLayout";
import HorizontalLayout from "./HorizontalLayout";

export interface WidgetContext {
  nodeId: string;
  propValue: (prop: string) => unknown;
  onChange: (prop: string, value: unknown) => void;
}

export default function Renderer({ spec, ctx }: { spec: WidgetSpec; ctx: WidgetContext }) {
  switch (spec.kind) {
    case "NumberField": return <NumberField spec={spec} ctx={ctx} />;
    case "ComboBox": return <ComboBox spec={spec} ctx={ctx} />;
    case "CheckBox": return <CheckBox spec={spec} ctx={ctx} />;
    case "TextField": return <TextField spec={spec} ctx={ctx} />;
    case "FilePath": return <FilePath spec={spec} ctx={ctx} />;
    case "Button": return <Button spec={spec} ctx={ctx} />;
    case "Progress": return <Progress spec={spec} ctx={ctx} />;
    case "Preview": return <Preview spec={spec} ctx={ctx} />;
    case "Custom": return <CustomW spec={spec} ctx={ctx} />;
    case "VerticalLayout": return <VerticalLayout spec={spec} ctx={ctx} />;
    case "HorizontalLayout": return <HorizontalLayout spec={spec} ctx={ctx} />;
  }
}
