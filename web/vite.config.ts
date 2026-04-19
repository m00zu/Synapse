import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import fs from "fs";

/** Copy monaco-editor's vs/ runtime into the build output so the Python
 * editor works offline. Without this, @monaco-editor/react would fetch
 * vs/ from jsdelivr's CDN at runtime — great for online demos, broken
 * for laptop-only `synapse serve` use. */
function copyMonacoPlugin(): Plugin {
  return {
    name: "copy-monaco",
    apply: "build",
    writeBundle() {
      const src = path.resolve(__dirname, "node_modules/monaco-editor/min/vs");
      const dst = path.resolve(__dirname, "../synapse/web/dist/monaco/vs");
      if (!fs.existsSync(src)) return;
      fs.cpSync(src, dst, { recursive: true });
    },
  };
}

export default defineConfig({
  plugins: [react(), copyMonacoPlugin()],
  build: {
    // Built artifacts ship with the Python wheel at synapse/web/dist/
    outDir: path.resolve(__dirname, "../synapse/web/dist"),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true, // so /api/ws (WebSocket) proxies correctly
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
