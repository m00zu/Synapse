import { useEffect, useState } from "react";
import { usePreviewVersion } from "../../hooks/usePreviewRefresh";

interface TableBlob {
  columns: string[];
  rows: unknown[][];
  total_rows: number;
}

export default function TablePreview({ nodeId, port }: { nodeId: string; port: string }) {
  const version = usePreviewVersion(nodeId, port);
  const [blob, setBlob] = useState<TableBlob | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (version === 0) return;
    let cancelled = false;
    fetch(
      `/api/files/preview/${encodeURIComponent(nodeId)}/${encodeURIComponent(port)}?v=${version}`
    )
      .then((r) =>
        r.ok ? r.json() : Promise.reject(new Error(String(r.status)))
      )
      .then((b: TableBlob) => {
        if (!cancelled) {
          setBlob(b);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [nodeId, port, version]);

  if (version === 0)
    return <div className="text-xs text-fg/40 italic p-2">No preview yet.</div>;
  if (error)
    return <div className="text-xs text-red-400 p-2">{error}</div>;
  if (!blob)
    return <div className="text-xs text-fg/40 p-2">Loading…</div>;

  return (
    <div className="max-h-64 overflow-auto text-xs border border-border rounded">
      <table className="w-full border-collapse">
        <thead className="sticky top-0 bg-bg2">
          <tr>
            {blob.columns.map((c) => (
              <th key={c} className="text-left px-2 py-1 border-b border-border">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {blob.rows.map((row, i) => (
            <tr key={i} className="border-b border-border/30 hover:bg-bg">
              {row.map((v, j) => (
                <td key={j} className="px-2 py-1 font-mono whitespace-nowrap">
                  {v == null ? (
                    <span className="text-fg/30">null</span>
                  ) : (
                    String(v)
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {blob.total_rows > blob.rows.length && (
        <div className="px-2 py-1 text-fg/50">
          Showing {blob.rows.length} / {blob.total_rows} rows
        </div>
      )}
    </div>
  );
}
