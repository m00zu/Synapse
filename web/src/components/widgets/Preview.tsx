import type { PreviewSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";
import ImagePreview from "../preview/ImagePreview";
import TablePreview from "../preview/TablePreview";
import FigurePreview from "../preview/FigurePreview";

export default function Preview({ spec, ctx }: { spec: PreviewSpec; ctx: WidgetContext }) {
  // spec.source is "output:<port>". Parse the port name out.
  const port = spec.source.startsWith("output:")
    ? spec.source.slice("output:".length)
    : "out";
  switch (spec.preview_kind) {
    case "image":  return <ImagePreview  nodeId={ctx.nodeId} port={port} />;
    case "table":  return <TablePreview  nodeId={ctx.nodeId} port={port} />;
    case "figure": return <FigurePreview nodeId={ctx.nodeId} port={port} />;
  }
}
