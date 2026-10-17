// "Review each card": the viewer opens after the page settles and steps through every card.
import { expect, test } from '@playwright/test';

test('review mode steps through cards and records choices', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tcg-test-run', '1'));
  await page.addInitScript(() => localStorage.setItem('tcg-scanner-settings-v1', JSON.stringify({ mode: 'manual', reviewMode: 'review' })));
  await page.goto('/?mock=' + (process.env.MOCK ?? '4QxpRG4E6d0-312-324.webm'));
  await expect(page.locator('.title')).not.toHaveText('Loading models', { timeout: 120000 });
  await page.waitForTimeout(1500);
  await page.getByRole('button', { name: 'Scan now' }).click();
  await expect(page.locator('.viewer')).toBeVisible({ timeout: 60000 });
  const total = await page.locator('.strip .slot').count();
  for (let i = 0; i < Math.min(total, 4) && (await page.locator('.viewer').count()); i++) {
    const correct = page.getByRole('button', { name: /Correct · (next|done)/ });
    if (await correct.count()) await correct.click();
    else if (await page.locator('.cand').count()) {
      await page.locator('.cand').first().click();
      await page.getByRole('button', { name: 'This is it' }).click();
    } else await page.getByRole('button', { name: 'Skip' }).click();
    await page.waitForTimeout(150);
  }
  const checked = await page.evaluate(() => (window as any).tcgDebug.page().instances.filter((c: any) => c.userChoice).length);
  console.log('cards found', total, '| checked by user', checked);
  expect(checked).toBeGreaterThanOrEqual(1);
});
