import { defineConfig } from '@playwright/test';

// E2E_PORT: separate dev servers for parallel checkouts (worktrees); the server is reused per port.
const port = Number(process.env.E2E_PORT ?? 5179);

export default defineConfig({
  testDir: 'e2e',
  // Default run: the two end-to-end flows (single page: scan, check, next scan, collection; batch:
  // scan pages, reload, review at the end). Others run by name.
  testMatch: process.env.E2E_ALL ? '**/*.spec.ts' : ['**/check-flow.spec.ts', '**/batch-flow.spec.ts'],
  timeout: 90000,
  use: { baseURL: `https://localhost:${port}`, ignoreHTTPSErrors: true, viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, channel: 'chrome' },
  webServer: { command: `npx vite --port ${port} --strictPort`, url: `https://localhost:${port}`, ignoreHTTPSErrors: true, reuseExistingServer: true, timeout: 60000 },
});
