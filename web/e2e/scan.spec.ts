// End-to-end: test-video camera mock -> capture -> detection -> per-card recognition -> review.
import { expect, test } from '@playwright/test';

const shots = process.env.SHOTS_DIR;

test('scans a page from the test-video camera', async ({ page }) => {
  const logs: string[] = [];
  page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
  await page.addInitScript(() => localStorage.setItem('tcg-test-run', '1'));
  await page.goto('/?mock=' + (process.env.MOCK ?? 'synthetic_replay.mp4'));
  await expect(page.locator('.title')).not.toHaveText('Loading models', { timeout: 120000 });
  if (shots) await page.screenshot({ path: `${shots}/1-live.png` });
  // Auto-capture once steady, or capture manually.
  const scan = page.getByRole('button', { name: 'Scan now' });
  await page.waitForTimeout(2500);
  if ((await scan.count()) && (await scan.isEnabled())) await scan.click();
  await expect(page.locator('.chip')).toContainText('Captured', { timeout: 30000 });
  if (shots) {
    await page.waitForTimeout(250);
    await page.screenshot({ path: `${shots}/2-processing.png` });
  }
  await expect(page.locator('.title')).toHaveText(/Review page|No cards found/, { timeout: 120000 });
  // A test video can be between scenes; capture again a few times until something is detected.
  for (let attempt = 0; attempt < 6 && (await page.locator('.title').textContent()) === 'No cards found'; attempt++) {
    await page.getByRole('button', { name: 'Scan again' }).first().click();
    await page.waitForTimeout(450);
    await page.getByRole('button', { name: 'Scan now' }).click();
    await expect(page.locator('.title')).toHaveText(/Review page|No cards found/, { timeout: 120000 });
  }
  await page.waitForTimeout(800);
  if (shots) await page.screenshot({ path: `${shots}/3-settled.png` });
  const found = await page.locator('.strip .slot').count();
  console.log('cards found:', found, '| sub:', await page.locator('.sub').textContent());
  if (found) {
    await page.locator('.strip .slot').first().click();
    await expect(page.locator('.viewer')).toBeVisible();
    await page.waitForTimeout(600);
    if (shots) await page.screenshot({ path: `${shots}/4-viewer.png` });
    // Compare with the first candidate, then step to the next card.
    const cand = page.locator('.cand').first();
    if (await cand.count()) {
      await cand.click();
      await page.waitForTimeout(500);
      if (shots) await page.screenshot({ path: `${shots}/5-compare.png` });
      await page.getByRole('button', { name: 'Not this' }).click();
    }
    await page.getByRole('button', { name: 'In photo' }).click();
    await page.waitForTimeout(300);
    if (shots) await page.screenshot({ path: `${shots}/6-in-photo.png` });
    await page.getByRole('button', { name: 'Next card' }).first().click();
    await expect(page.locator('.vtop .t')).toContainText('CARD 2');
    await page.getByRole('button', { name: 'Close', exact: true }).click();
    await page.getByRole('button', { name: 'Show captured image full screen' }).click();
    await page.waitForTimeout(300);
    if (shots) await page.screenshot({ path: `${shots}/7-expanded.png` });
    await page.getByRole('button', { name: 'Shrink captured image' }).click();
    await page.getByRole('button', { name: 'All cards on this page' }).click();
    if (shots) { await page.waitForTimeout(400); await page.screenshot({ path: `${shots}/8-review.png` }); }
  }
  console.log(logs.filter((l) => /error|warn/i.test(l)).slice(0, 20).join('\n'));
});
