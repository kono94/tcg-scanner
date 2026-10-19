// Collection export and import: a saved binder survives clearing the browser through an exported file.
import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

const thumb = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=';
const seeded = {
  sessions: { activeId: null, sessions: [{ id: 's-e2e', name: 'Export Binder', startedAt: '2026-10-19T10:00:00Z', nextCapture: 2, named: true }] },
  pages: [{ key: 'page-e2e-1', sessionId: 's-e2e', pageNumber: 1, revision: 1, savedAt: '2026-10-19T10:00:00Z', partial: false, cards: [
    { number: 1, identity: null, name: null, cardId: null, state: 'needs-attention', reason: 'match-unclear', languageAssumed: false, checkedByUser: false, value: null, thumb, hasCrop: true },
    { number: 2, identity: null, name: null, cardId: null, state: 'queued', languageAssumed: false, checkedByUser: false, value: null, thumb, hasCrop: true, queued: 1 },
  ] }],
};

test('export a binder, clear the browser, import it again', async ({ page }) => {
  const shots = process.env.SHOTS_DIR;
  await page.addInitScript((data) => {
    localStorage.setItem('tcg-test-run', '1');
    // Desktop download path (phones get the share sheet).
    Object.defineProperty(navigator, 'canShare', { value: undefined });
    if (!sessionStorage.getItem('seeded')) {
      sessionStorage.setItem('seeded', '1');
      localStorage.setItem('tcg-scanner-sessions-v2', JSON.stringify(data.sessions));
      localStorage.setItem('tcg-scanner-collection-v1', JSON.stringify(data.pages));
    }
  }, seeded);
  await page.goto('/');
  // Home shows the app version (release tag or git describe) and the bundle.
  await expect(page.getByText(/^Version \S+ · bundle \w{8}$/)).toBeVisible({ timeout: 60000 });
  if (shots) await page.screenshot({ path: `${shots}/t0-home.png` });
  await page.locator('.hbtn', { hasText: 'Collection' }).click();
  await expect(page.locator('.sn', { hasText: 'Export Binder' })).toBeVisible();
  if (shots) await page.screenshot({ path: `${shots}/t1-collection.png` });

  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export the collection as a file' }).click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(/^tcg-collection-\d{4}-\d{2}-\d{2}\.json$/);
  const exported = JSON.parse(await readFile((await file.path())!, 'utf8'));
  expect(exported).toMatchObject({ format: 'tcg-scanner-collection', version: 1, sessions: [{ name: 'Export Binder' }] });
  expect(exported.pages[0].cards).toHaveLength(2);

  // A new, empty browser: only Import is offered.
  await page.evaluate(() => { localStorage.removeItem('tcg-scanner-sessions-v2'); localStorage.removeItem('tcg-scanner-collection-v1'); });
  await page.reload();
  await page.locator('.hbtn', { hasText: 'Collection' }).click();
  await expect(page.getByText('Scanned pages appear here')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Export the collection as a file' })).toHaveCount(0);
  if (shots) await page.screenshot({ path: `${shots}/t2-empty.png` });
  const chooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Import a collection file' }).click();
  await (await chooser).setFiles((await file.path())!);
  await expect(page.getByText(/Imported 1 page · 1 new binder/)).toBeVisible();
  await expect(page.locator('.sn', { hasText: 'Export Binder' })).toBeVisible();
  await expect(page.locator('.prow')).toHaveCount(1);
  await expect(page.locator('.pthumbs img')).toHaveCount(2);

  // Importing the same file again adds nothing.
  const again = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Import a collection file' }).click();
  await (await again).setFiles((await file.path())!);
  await expect(page.getByText('Nothing new')).toBeVisible();
  await expect(page.locator('.prow')).toHaveCount(1);
  const cards = await page.evaluate(() => JSON.parse(localStorage.getItem('tcg-scanner-collection-v1')!)[0].cards);
  expect(cards.map((c: { state: string; hasCrop: boolean }) => [c.state, c.hasCrop])).toEqual([['needs-attention', false], ['needs-attention', false]]);
});
