// Main-thread responsiveness after a scan, with the CPU throttled to approximate a phone.
import { expect, test } from '@playwright/test';

test('review opens quickly after a scan (CPU throttled 6x)', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tcg-test-run', '1'));
  await page.addInitScript(() => localStorage.setItem('tcg-scanner-settings-v1', JSON.stringify({ mode: 'manual' })));
  await page.goto('/?mock=4QxpRG4E6d0-312-324.webm&wasm');
  await expect(page.getByRole('button', { name: /Use camera/ })).toHaveCount(0, { timeout: 120000 });
  await page.waitForTimeout(800);
  await page.getByRole('button', { name: 'Scan now' }).click();
  await expect(page.locator('.title')).toHaveText(/Review page/, { timeout: 120000 });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 6 });
  await page.waitForTimeout(1500);
  // Long tasks over 2 s of idle review.
  const busy = await page.evaluate(() => new Promise<number>((resolve) => {
    let total = 0;
    const obs = new PerformanceObserver((l) => { for (const e of l.getEntries()) total += e.duration; });
    obs.observe({ type: 'longtask', buffered: false });
    setTimeout(() => { obs.disconnect(); resolve(total); }, 2000);
  }));
  const t0 = Date.now();
  await page.getByRole('button', { name: 'All cards on this page' }).click();
  await expect(page.locator('.review')).toBeVisible();
  const open = Date.now() - t0;
  const t1 = Date.now();
  await page.locator('.tile').first().click();
  await expect(page.locator('.viewer')).toBeVisible();
  const viewer = Date.now() - t1;
  console.log(`idle main-thread long tasks ${Math.round(busy)} ms per 2 s | open review ${open} ms | open card ${viewer} ms`);
});
