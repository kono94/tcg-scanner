import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  // Default run: the one end-to-end flow (scan, check, next scan, collection). Others run by name.
  testMatch: process.env.E2E_ALL ? '**/*.spec.ts' : '**/check-flow.spec.ts',
  timeout: 90000,
  use: { baseURL: 'https://localhost:5179', ignoreHTTPSErrors: true, viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, channel: 'chrome' },
  webServer: { command: 'npx vite --port 5179 --strictPort', url: 'https://localhost:5179', ignoreHTTPSErrors: true, reuseExistingServer: true, timeout: 60000 },
});
