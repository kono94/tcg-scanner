import { expect, test } from '@playwright/test';

test('batch holds its snapshot, skips repeated auto pages, holds a manual repeat for explicit keep, deletes a binder', async ({ page }, info) => {
  await page.addInitScript(() => {
    localStorage.setItem('tcg-test-run', '1');
    if (!localStorage.getItem('tcg-scanner-settings-v1')) localStorage.setItem('tcg-scanner-settings-v1', JSON.stringify({ workflow: 'batch', mode: 'auto', dwellMs: 800, batchCooldownMs: 3000 }));
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { value: async () => {
      // Static card-bearing replay with tiny sensor-like changes: a fresh frame is not a new page.
      const source = document.createElement('video');
      source.src = '/media/4QxpRG4E6d0-312-324.webm';
      source.muted = true;
      await new Promise<void>((resolve) => { source.onloadeddata = () => resolve(); source.load(); });
      await new Promise<void>((resolve) => { source.onseeked = () => resolve(); source.currentTime = 1; });
      const canvas = document.createElement('canvas');
      canvas.width = source.videoWidth; canvas.height = source.videoHeight;
      const ctx = canvas.getContext('2d')!;
      let frame = 0;
      const draw = () => {
        ctx.drawImage(source, 0, 0);
        ctx.fillStyle = ++frame % 2 ? '#aaa' : '#bbb';
        ctx.fillRect(0, 0, 2, 2);
      };
      draw();
      const interval = setInterval(draw, 100);
      const stream = canvas.captureStream(10);
      const track = stream.getVideoTracks()[0], stop = track.stop.bind(track);
      track.stop = () => { clearInterval(interval); stop(); source.removeAttribute('src'); source.load(); };
      return stream;
    } });
  });
  await page.goto('/');
  // Hold extraction so the processing display can be verified independent of inference speed.
  await page.waitForFunction(() => !!(window as unknown as { tcgDebug?: unknown }).tcgDebug);
  await page.evaluate(() => {
    const w = window as unknown as { tcgDebug: { recognizer: { extract: (...args: unknown[]) => Promise<unknown> } }; releaseExtraction: () => void };
    const original = w.tcgDebug.recognizer.extract.bind(w.tcgDebug.recognizer);
    const barrier = new Promise<void>((resolve) => { w.releaseExtraction = resolve; });
    w.tcgDebug.recognizer.extract = async (...args) => { await barrier; return original(...args); };
  });
  await page.getByRole('button', { name: 'New binder…', exact: true }).click();
  await page.getByRole('textbox', { name: 'New binder name (optional)' }).fill('Duplicate test binder');
  await page.getByRole('button', { name: 'Start', exact: true }).click();
  // Batch auto scanning waits for an explicit Start under the live preview.
  await expect(page.getByRole('heading', { name: 'Ready for page 1' })).toBeVisible({ timeout: 120000 });
  await page.waitForTimeout(2000);
  expect(await page.locator('.cam.captured').count()).toBe(0);
  await page.getByRole('button', { name: 'Start scanning', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Saving page 1' })).toBeVisible({ timeout: 120000 });
  await expect(page.locator('.cam.captured')).toHaveCount(1);
  const still = page.locator('canvas.still');
  await expect.poll(() => still.evaluate((c: HTMLCanvasElement) => {
    const rgba = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
    return rgba.some((v, i) => i % 4 !== 3 && v > 20);
  })).toBe(true);
  await expect.poll(() => page.locator('.cam video').evaluate((v: HTMLVideoElement) => v.paused && !(v.srcObject as MediaStream).getVideoTracks()[0].enabled)).toBe(true);
  await page.screenshot({ path: info.outputPath('batch-frozen.png'), animations: 'disabled' });
  await page.evaluate(() => (window as unknown as { releaseExtraction: () => void }).releaseExtraction());
  // Found cards are outlined on the frozen photo, which then confirms the save before going live.
  await expect(page.locator('.st-found .br').first()).toBeAttached({ timeout: 60000 });
  await expect(page.getByRole('heading', { name: 'Page 1 saved' })).toBeVisible({ timeout: 60000 });
  // Then the page-turn window, counted down over the live preview.
  await expect(page.getByRole('heading', { name: 'Turn the page' })).toBeVisible({ timeout: 10000 });
  await expect(page.locator('.cuebox.turn')).toContainText(/Turn the page\s*\d s/);
  const count = () => page.evaluate(() => JSON.parse(localStorage.getItem('tcg-scanner-collection-v1') ?? '[]').length as number);
  // Deliberate observation window spans the auto cooldown and several dwell periods.
  await page.waitForTimeout(6500);
  expect(await count()).toBe(1);
  // Pause scanning: an explicit paused state, resumed with Continue scanning.
  await page.getByRole('button', { name: 'Pause scanning', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Scanning paused' })).toBeVisible();
  await expect(page.locator('.cuebox.paused')).toBeVisible();
  await page.getByRole('button', { name: 'Continue scanning', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Pause scanning', exact: true })).toBeVisible();
  const scan = page.getByRole('button', { name: 'Scan now', exact: true });
  await expect(scan).toBeEnabled();
  await scan.click();
  await expect(page.locator('.duplicatebar')).toContainText('Possible repeat of page 1', { timeout: 60000 });
  expect(await count()).toBe(1);
  // Persisted suspect survives reload; a manual capture alone did not override it.
  await page.locator('.homebtn').click();
  await page.getByRole('button', { name: /Settings.*Language/ }).click();
  await page.getByRole('combobox', { name: 'Trigger', exact: true }).selectOption('manual');
  await page.reload();
  await page.getByRole('button', { name: /Continue Duplicate test binder/ }).click();
  await expect(page.getByRole('button', { name: 'Keep anyway', exact: true })).toBeEnabled({ timeout: 120000 });
  await page.getByRole('button', { name: 'Keep anyway', exact: true }).click();
  await expect.poll(count).toBe(2);
  await expect(page.locator('.duplicatebar')).toBeHidden();
  // Deleting the active collection also retires queued work and removes its IndexedDB crops.
  await page.locator('.homebtn').click();
  await page.locator('.hbtn', { hasText: 'Collection' }).click();
  await expect(page.locator('.prow')).toHaveCount(2);
  await page.getByRole('button', { name: 'Delete Duplicate test binder', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Delete binder', exact: true }).click();
  await expect(page.locator('.prow')).toHaveCount(0);
  expect(await count()).toBe(0);
  await expect.poll(() => page.evaluate(async () => {
    const db = await new Promise<IDBDatabase>((resolve) => { const req = indexedDB.open('tcg-scanner-crops'); req.onsuccess = () => resolve(req.result); });
    try { return await new Promise<number>((resolve) => { const req = db.transaction('crops').objectStore('crops').count(); req.onsuccess = () => resolve(req.result); }); }
    finally { db.close(); }
  })).toBe(0);
  await page.reload();
  expect(await count()).toBe(0);
});
