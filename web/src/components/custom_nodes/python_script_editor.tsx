import Editor, { loader } from "@monaco-editor/react";
import type { CustomSpec } from "../../api/types";
import type { WidgetContext } from "../widgets/Renderer";

// Point Monaco at the locally-vendored vs/ copy (copied into dist/ by
// vite.config.ts's copyMonacoPlugin). This lets `synapse serve` work
// offline — the CDN default would hang forever on a laptop without net.
loader.config({ paths: { vs: "/monaco/vs" } });

export default function PythonScriptEditor(
  { spec, ctx }: { spec: CustomSpec; ctx: WidgetContext }
) {
  const prop = (spec.props.prop as string) || "code";
  const raw = ctx.propValue(prop);
  const value = (typeof raw === "string" ? raw : "");
  return (
    <div className="flex flex-col gap-1 text-xs">
      <span className="text-fg/70">Python code</span>
      <div className="h-64 border border-border rounded overflow-hidden">
        <Editor
          defaultLanguage="python"
          value={value}
          theme="vs-dark"
          options={{
            minimap: { enabled: false },
            fontSize: 12,
            scrollBeyondLastLine: false,
            automaticLayout: true,
          }}
          onChange={(v) => ctx.onChange(prop, v ?? "")}
        />
      </div>
    </div>
  );
}
