import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0d1117",
        bg2: "#161b22",
        fg: "#c9d1d9",
        accent: "#58a6ff",
        border: "#30363d",
      },
    },
  },
  plugins: [],
} satisfies Config;
