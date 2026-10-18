// The main loop: scan -> check the cards that need the user (pick or skip) -> next scan.
import { expect, test } from '@playwright/test';

test('scan, check unknown cards, next scan', async ({ page }) => {
  const shots = process.env.SHOTS_DIR;
  await page.addInitScript(() => localStorage.setItem('tcg-test-run', '1'));
  await page.addInitScript(() => localStorage.setItem('tcg-scanner-settings-v1', JSON.stringify({ mode: 'manual' })));
  await page.goto('/?mock=' + (process.env.MOCK ?? '4QxpRG4E6d0-312-324.webm'));
  await expect(page.getByRole('button', { name: /Use camera/ })).toHaveCount(0, { timeout: 120000 });
  await page.waitForTimeout(800);
  await page.getByRole('button', { name: 'Scan now' }).click();
  const check = page.getByRole('button', { name: /^Check \d+ cards?$/ });
  await expect(check.or(page.getByRole('button', { name: 'Next scan' }))).toBeVisible({ timeout: 120000 });
  if (shots) await page.screenshot({ path: `${shots}/c1-settled.png` });
  let steps = 0;
  if (await check.count()) {
    const total = Number((await check.textContent())!.match(/\d+/)![0]);
    await check.click();
    for (; steps < total + 2 && (await page.locator('.viewer').count()); steps++) {
      await expect(page.locator('.vtop .t')).toContainText('CHECK');
      if (steps === 0 && shots) await page.waitForTimeout(400), await page.screenshot({ path: `${shots}/c2-check.png` });
      // Rotate: pick the first candidate, "none of these", skip.
      if (steps % 3 === 0 && (await page.locator('.cand').count())) {
        await page.locator('.cand').first().click();
        await page.getByRole('button', { name: 'This is it' }).click();
      } else if (steps % 3 === 1 && (await page.getByRole('button', { name: 'None of these' }).count())) {
        await page.getByRole('button', { name: 'None of these' }).click();
      } else {
        const skip = page.getByRole('button', { name: 'Decide later', exact: true });
        if (await skip.count()) await skip.click();
        else await page.getByRole('button', { name: /Correct · (next|done)/ }).click();
      }
    }
    await expect(page.locator('.viewer')).toHaveCount(0);
    console.log('checked', steps, 'of', total);
  }
  if (shots) await page.screenshot({ path: `${shots}/c3-done.png` });
  await page.getByRole('button', { name: 'Next scan' }).click();
  await expect(page.locator('.chip')).toContainText('Page 2', { timeout: 60000 });
  await expect(page.getByRole('button', { name: 'Scan now' })).toBeVisible();
  // The saved page shows up in the Collection tab.
  await page.locator('.homebtn').click();
  if (shots) { await page.waitForTimeout(500); await page.screenshot({ path: `${shots}/c3b-home.png` }); }
  await page.locator('.hbtn', { hasText: 'Collection' }).click();
  await expect(page.locator('.prow')).toHaveCount(1);
  if (shots) await page.screenshot({ path: `${shots}/c4-collection.png` });
  await page.locator('.prow').first().click();
  await expect(page.locator('.crow2').first()).toBeVisible();
  if (shots) await page.screenshot({ path: `${shots}/c5-page.png` });
  // A saved card can be checked again: search the gallery again, search the catalog, pick a card.
  const rows = await page.locator('.crow2').count();
  await page.locator('.crow2').first().click();
  await expect(page.getByText(/Closest matches|cannot be checked again/)).toBeVisible({ timeout: 60000 });
  await page.getByRole('searchbox').fill('Nami');
  await expect(page.locator('.cand').filter({ hasText: /\d/ }).first()).toBeVisible();
  if (shots) await page.screenshot({ path: `${shots}/c6-card.png` });
  await page.locator('.cand').last().click();
  await page.getByRole('button', { name: 'Use this card' }).click();
  await expect(page.locator('.savedcard .k').filter({ hasText: 'checked by you' })).toBeVisible();
  await page.locator('.back').click();
  // Swipe left, then Delete.
  await page.locator('.crow2').first().scrollIntoViewIfNeeded();
  const box = (await page.locator('.crow2').first().boundingBox())!;
  await page.mouse.move(box.x + box.width - 20, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 140, box.y + box.height / 2, { steps: 8 });
  await page.mouse.up();
  await page.locator('.swipe.open .swipedel').click();
  await expect(page.locator('.crow2')).toHaveCount(rows - 1);
  await page.locator('.homebtn').click();
  await page.locator('.hbtn.primary').click();
  await expect(page.getByRole('button', { name: 'Scan now' })).toBeVisible();
});
