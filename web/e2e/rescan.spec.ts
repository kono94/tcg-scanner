// Repeated capture cycles: the worker reuses its canvases/buffers; results must stay correct each time.
import { expect, test } from '@playwright/test';

test('scan, scan again, repeatedly', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tcg-test-run', '1'));
  await page.addInitScript(() => localStorage.setItem('tcg-scanner-settings-v1', JSON.stringify({ mode: 'manual' })));
  await page.goto('/?mock=' + (process.env.MOCK ?? '4QxpRG4E6d0-312-324.webm') + (process.env.WASM ? '&wasm' : ''));
  await expect(page.getByRole('button', { name: /Use camera/ })).toHaveCount(0, { timeout: 120000 });
  const counts: string[] = [];
  for (let i = 0; i < 6; i++) {
    await page.waitForTimeout(400);
    await page.getByRole('button', { name: 'Scan now' }).click();
    await expect(page.locator('.title')).toHaveText(/Review page|No cards found/, { timeout: 60000 });
    counts.push((await page.locator('.sub').textContent()) ?? '');
    await page.getByRole('button', { name: 'Scan again' }).first().click();
  }
  const heap = await page.evaluate(() => (performance as any).memory?.usedJSHeapSize / 1e6);
  console.log('cycles:', counts.join(' | '), '| main heap MB', heap?.toFixed(0));
  expect(counts.filter((c) => /ready|attention/.test(c)).length).toBeGreaterThanOrEqual(4);
});
