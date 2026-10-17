// Batch workflow: scan page after page, reload mid-way (queue survives), finish, review, Collection.
import { expect, test } from '@playwright/test';

test('batch: scan pages, reload, review at the end', async ({ page }) => {
  const shots = process.env.SHOTS_DIR;
  const mock = '/?mock=' + (process.env.MOCK ?? '4QxpRG4E6d0-312-324.webm');
  await page.addInitScript(() => {
    localStorage.setItem('tcg-test-run', '1');
    if (!localStorage.getItem('tcg-scanner-settings-v1')) localStorage.setItem('tcg-scanner-settings-v1', JSON.stringify({ mode: 'manual', workflow: 'batch' }));
  });
  await page.goto(mock);
  const scan = page.getByRole('button', { name: 'Scan now' });
  await expect(scan).toBeEnabled({ timeout: 120000 });
  await page.waitForTimeout(800);
  const bar = page.locator('.batchbar');
  for (const n of [1, 2]) {
    await scan.click();
    await expect(page.locator('.toast.show')).toContainText(new RegExp(`Page ${n} saved · \\d+ cards?|No cards found`), { timeout: 60000 });
    // Straight back to the live preview: no captured still, Scan now available again.
    await expect(page.locator('.cam.captured')).toHaveCount(0);
    if (n === 1 && shots) await page.screenshot({ path: `${shots}/b1-after-capture.png` });
    await expect(scan).toBeEnabled({ timeout: 60000 });
    await page.waitForTimeout(1200); // the test video moves on
  }
  await expect(bar).toContainText('Batch · 2 pages');
  // Reload while cards may still wait: pages and the queue survive, checking resumes.
  await page.reload();
  await expect(bar).toContainText('Batch · 2 pages', { timeout: 120000 });
  await page.getByRole('button', { name: 'Finish & review' }).click();
  await expect(page.getByRole('heading', { name: 'Batch review' })).toBeVisible();
  await expect(page.getByText(/still being checked/)).toHaveCount(0, { timeout: 120000 });
  if (shots) await page.screenshot({ path: `${shots}/b2-review.png` });
  // Cards that need the user: answer one, the next opens by itself.
  const need = page.locator('.batchrev .clist .crow2');
  const count = await need.count();
  if (count) {
    await need.first().click();
    await expect(page.getByText(/Closest matches|cannot be checked again/)).toBeVisible({ timeout: 60000 });
    if (shots) await page.screenshot({ path: `${shots}/b3-card.png` });
    await page.getByRole('button', { name: 'Not in the catalog' }).click();
    if (count > 1) await expect(page.locator('.savedcard')).toBeVisible();
    else await expect(page.getByRole('heading', { name: 'Batch review' })).toBeVisible();
    console.log('needed the user:', count);
  }
  if (await page.locator('.savedcard').count()) await page.locator('.savedcard .back').click();
  await expect(page.getByRole('heading', { name: 'Batch review' })).toBeVisible();
  await page.getByRole('button', { name: 'Done' }).click();
  // Both pages are in the Collection; the batch is closed.
  await expect(page.locator('.prow')).toHaveCount(2);
  if (shots) await page.screenshot({ path: `${shots}/b4-collection.png` });
});
