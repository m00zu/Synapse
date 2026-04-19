import { useToasts } from "../../store/toasts";

export default function Toasts() {
  const { toasts, dismiss } = useToasts();
  return (
    <div className="fixed top-3 right-3 flex flex-col gap-2 z-50 max-w-sm">
      {toasts.map((t) => (
        <div
          key={t.id}
          role="alert"
          onClick={() => dismiss(t.id)}
          className={`px-3 py-2 rounded border cursor-pointer text-sm shadow-lg ${
            t.kind === "error"
              ? "bg-red-900/80 border-red-500 text-red-100"
              : "bg-bg2 border-border text-fg"
          }`}
        >
          {t.text}
        </div>
      ))}
    </div>
  );
}
