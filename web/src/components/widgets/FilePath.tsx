import { useState } from "react";
import type { FilePathSpec } from "../../api/types";
import type { WidgetContext } from "./Renderer";
import { api } from "../../api/client";
import ServerBrowseDialog from "../files/ServerBrowseDialog";

export default function FilePath(
  { spec, ctx }: { spec: FilePathSpec; ctx: WidgetContext }
) {
  const raw = ctx.propValue(spec.prop);
  const value = (typeof raw === "string" ? raw : spec.default);
  const [browsing, setBrowsing] = useState(false);

  const onUpload = async (file: File) => {
    try {
      const res = await api.uploadFile(file);
      ctx.onChange(spec.prop, res.server_path);
    } catch (e) {
      console.error("upload failed:", e);
    }
  };

  return (
    <div className="flex flex-col gap-1 text-xs">
      <span className="text-fg/70">{spec.label}</span>
      <div className="flex gap-1">
        <input
          type="text"
          className="flex-1 bg-bg border border-border rounded px-2 py-1 text-fg font-mono min-w-0"
          value={value}
          onChange={(e) => ctx.onChange(spec.prop, e.target.value)}
        />
        <button
          className="px-2 py-1 border border-border rounded hover:bg-bg2 shrink-0"
          onClick={() => setBrowsing(true)}
          type="button"
        >
          Browse
        </button>
        <label className="px-2 py-1 border border-border rounded hover:bg-bg2 cursor-pointer shrink-0">
          Upload
          <input
            type="file"
            hidden
            onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
          />
        </label>
      </div>
      {browsing && (
        <ServerBrowseDialog
          // Empty string → server resolves to $HOME (or --allow-path).
          initialPath={value as string}
          onSelect={(p) => ctx.onChange(spec.prop, p)}
          onClose={() => setBrowsing(false)}
        />
      )}
    </div>
  );
}
