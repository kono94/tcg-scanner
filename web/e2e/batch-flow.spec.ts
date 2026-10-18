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
  // A saved page is confirmed on the frozen photo (title); other outcomes in the toast.
  const feedback = () => page.evaluate(() => [document.querySelector('.top .title')?.textContent ?? '', document.querySelector('.toast.show')?.textContent ?? ''].join(' | '));
  for (const n of [1, 2]) {
    await scan.click();
    await expect.poll(feedback, { timeout: 60000, intervals: [100] }).toMatch(new RegExp(`Page ${n} saved|No cards found|Possible repeat of page 1`));
    // The test video can still show the first page's cards: a manual capture of it is held, and
    // must be kept explicitly before scanning continues (it is never saved or discarded implicitly).
    if (/Possible repeat/.test(await feedback())) {
      await expect(scan).toBeDisabled();
      await page.getByRole('button', { name: 'Keep anyway', exact: true }).click();
      await expect(page.locator('.duplicatebar')).toBeHidden();
    }
    // Straight back to the live preview: no captured still, Scan now available again.
    await expect(page.locator('.cam.captured')).toHaveCount(0);
    if (n === 1 && shots) await page.screenshot({ path: `${shots}/b1-after-capture.png` });
    await expect(scan).toBeEnabled({ timeout: 60000 });
    await page.waitForTimeout(1200); // the test video moves on
  }
  await expect(bar).toContainText('2 pages');
  // Reload while cards may still wait: pages and the queue survive, checking resumes.
  await page.reload();
  await expect(bar).toContainText('2 pages', { timeout: 120000 });
  await page.getByRole('button', { name: 'Review the scanned pages' }).click();
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
