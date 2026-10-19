import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dashboard calls the API through "/api" on its own origin, and Vite forwards it to FastAPI.
// Same origin => no CORS in the browser at all, and 127.0.0.1 avoids the Windows localhost/::1 mismatch.
// Override the target with API_TARGET, or bypass the proxy entirely with VITE_API_URL (see src/lib/api.ts).
const target = process.env.API_TARGET ?? "http://127.0.0.1:8000";
const proxy = {
  "/api": {
    target,
    changeOrigin: true,
    ws: true, // WebSocket /api/stream/... as well
    rewrite: (path: string) => path.replace(/^\/api/, ""),
  },
};

// base "./" lets the built dashboard run from any folder (offline demo mode).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  server: { host: "127.0.0.1", port: 5173, proxy },
  preview: { host: "127.0.0.1", port: 4173, proxy },
  build: {
    rollupOptions: {
      output: { manualChunks: { react: ["react", "react-dom", "react-router-dom"], charts: ["recharts"] } },
    },
  },
});
