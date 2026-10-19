import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// base "./" lets the built dashboard run from any folder (offline demo mode).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  server: { port: 5173 },
  build: {
    rollupOptions: {
      output: { manualChunks: { react: ["react", "react-dom", "react-router-dom"], charts: ["recharts"] } },
    },
  },
});
