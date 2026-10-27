import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Port 3000 matches the default PARLORA_CORS_ORIGINS and stays clear of Hub's
// Vite port (5173); strictPort fails instead of silently picking another one.
export default defineConfig({
  plugins: [react()],
  server: { port: 3000, strictPort: true },
  preview: { port: 3000, strictPort: true },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["src/test/setup.ts"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/*.test.{ts,tsx}", "src/test/**", "src/main.tsx", "src/**/*.d.ts"],
      thresholds: { lines: 85, branches: 85, functions: 85, statements: 85 },
      reporter: ["text", "json-summary"],
    },
  },
});
