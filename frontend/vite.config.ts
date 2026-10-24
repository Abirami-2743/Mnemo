import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs on :8000; in production FastAPI serves the built dashboard itself.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": { target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000", changeOrigin: true } } },
});
