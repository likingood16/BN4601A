import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// npm run dev: screens on :5173, the Python back end (server.py) on :8000
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
