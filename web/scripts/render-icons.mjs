// Rasterize public/icons/icon.svg into the app icons (home screen, install, favicon) with Playwright's Chromium.
// Run after changing the SVG: node scripts/render-icons.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const out = join(dirname(fileURLToPath(import.meta.url)), '..', 'public/icons');
const tab = readFileSync(join(out, 'icon.svg'), 'utf8');
// Home-screen icons are square and full-bleed: iOS and Android round the corners themselves.
const full = tab.replace(' rx="112"', '');
// Maskable: Android may crop to a circle; keep the motif inside the central 80% safe zone.
const maskable = full.replace(/(<rect width="512" height="512" fill="[^"]+"\/>)([\s\S]*)<\/svg>/, '$1<g transform="translate(256 256) scale(.8) translate(-256 -256)">$2</g></svg>');
const jobs = [[full, 180, 'apple-touch-icon.png'], [full, 192, 'icon-192.png'], [full, 512, 'icon-512.png'], [maskable, 512, 'icon-maskable-512.png'], [tab, 32, 'favicon-32.png']];

const browser = await chromium.launch();
const page = await browser.newPage();
for (const [svg, size, name] of jobs) {
  await page.setViewportSize({ width: size, height: size });
  await page.setContent(`<style>html,body{margin:0;background:transparent}svg{display:block;width:${size}px;height:${size}px}</style>${svg}`);
  await page.locator('svg').screenshot({ path: join(out, name), omitBackground: true });
}
await browser.close();
console.log(`wrote ${jobs.length} icons to public/icons`);
