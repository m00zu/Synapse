// TS mirror of synapse/widgets/spec.py. Keep field names in exact sync —
// the /api/nodes endpoint emits the dataclass .asdict() form and the
// renderer assumes these keys.

export type WidgetKind =
  | "ComboBox" | "NumberField" | "CheckBox" | "TextField" | "FilePath"
  | "Button" | "Progress" | "Preview" | "Custom"
  | "VerticalLayout" | "HorizontalLayout";

export interface WidgetBase {
  kind: WidgetKind;
  tab?: string;
}

export interface ComboBoxSpec extends WidgetBase {
  kind: "ComboBox";
  prop: string;
  label: string;
  options: string[];
  default: string | null;
}

export interface NumberFieldSpec extends WidgetBase {
  kind: "NumberField";
  prop: string;
  label: string;
  min: number;
  max: number;
  step: number;
  decimals: number;
  default: number;
}

export interface CheckBoxSpec extends WidgetBase {
  kind: "CheckBox";
  prop: string;
  label: string;
  default: boolean;
}

export interface TextFieldSpec extends WidgetBase {
  kind: "TextField";
  prop: string;
  label: string;
  default: string;
  placeholder: string;
}

export interface FilePathSpec extends WidgetBase {
  kind: "FilePath";
  prop: string;
  label: string;
  mode: "server-browse" | "upload" | "either";
  file_filter: string;
  default: string;
}

export interface ButtonSpec extends WidgetBase {
  kind: "Button";
  action: string;
  label: string;
}

export interface ProgressSpec extends WidgetBase {
  kind: "Progress";
  prop: string;
  label: string;
}

export interface PreviewSpec extends WidgetBase {
  kind: "Preview";
  preview_kind: "image" | "table" | "figure";
  source: string;
}

export interface CustomSpec extends WidgetBase {
  kind: "Custom";
  component_id: string;
  props: Record<string, unknown>;
}

export interface VerticalLayoutSpec extends WidgetBase {
  kind: "VerticalLayout";
  children: WidgetSpec[];
}

export interface HorizontalLayoutSpec extends WidgetBase {
  kind: "HorizontalLayout";
  children: WidgetSpec[];
}

export type WidgetSpec =
  | ComboBoxSpec | NumberFieldSpec | CheckBoxSpec | TextFieldSpec | FilePathSpec
  | ButtonSpec | ProgressSpec | PreviewSpec | CustomSpec
  | VerticalLayoutSpec | HorizontalLayoutSpec;

// Server → client event envelope (from /api/ws).
export type WsEvent =
  | { kind: "node_started"; node_id: string }
  | { kind: "node_progress"; node_id: string; value: number }
  | { kind: "node_finished"; node_id: string; success: boolean; error?: string }
  | { kind: "run_finished"; run_id: string }
  | { kind: "preview_available"; node_id: string; port: string;
      preview_kind: "image" | "table" | "figure" };

// /api/nodes response shape.
export type WidgetCatalog = Record<string, WidgetSpec[]>;

// /api/nodes/categories response shape.
export interface PortInfo {
  name: string;
  type: string; // e.g. "image", "table", "mask", "any"
}

export interface NodeCategoryInfo {
  identifier: string;
  category: string;
  /** Remainder of the identifier after the category prefix, e.g.
   * 'nodes.image_process.filter' → 'filter'. Empty when the identifier
   * maps directly to a category with no sub-namespace. */
  subcategory: string;
  /** Human-readable name from the node class's NODE_NAME attribute.
   * Falls back to the class name if not set. */
  display_name: string;
  inputs: PortInfo[];
  outputs: PortInfo[];
}
export type NodeCategories = Record<string, NodeCategoryInfo>;
