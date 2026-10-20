import { expect, test } from '@playwright/test';

test('camera is held inside the scanner, released on leaving it, and reacquired on return', async ({ page }, info) => {
  await page.addInitScript(() => {
    localStorage.setItem('tcg-test-run', '1');
    localStorage.setItem('tcg-scanner-settings-v1', JSON.stringify({ mode: 'manual' }));
    let calls = 0;
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { value: async () => {
      calls++;
      const canvas = document.createElement('canvas');
      canvas.width = 640; canvas.height = 480;
      const g = canvas.getContext('2d')!;
      g.fillStyle = '#777';
      g.fillRect(0, 0, 640, 480);
      // Keep drawing so the stream delivers frames.
      setInterval(() => g.fillRect(0, 0, 1, 1), 100);
      return canvas.captureStream(10);
    } });
    Object.defineProperty(window, 'cameraRequests', { get: () => calls });
  });
  await page.setViewportSize({ width: 320, height: 740 });
  await page.goto('/');
  await page.getByRole('button', { name: 'New binder…', exact: true }).click();
  await page.getByRole('textbox', { name: 'New binder name (optional)' }).fill('Red Binder');
  await page.getByRole('button', { name: 'Start', exact: true }).click();
  const video = page.locator('.cam video');
  const live = () => video.evaluate((v: HTMLVideoElement) => {
    const t = (v.srcObject as MediaStream | null)?.getVideoTracks()[0];
    return !v.paused && t?.readyState === 'live' && t.enabled;
  });
  const requests = () => page.evaluate(() => (window as unknown as { cameraRequests: number }).cameraRequests);
  await expect.poll(live).toBe(true);
  let track = await video.evaluateHandle((v: HTMLVideoElement) => (v.srcObject as MediaStream).getVideoTracks()[0]);
  // Leaving the scanner releases the camera (the camera light goes off).
  await page.locator('.homebtn').click();
  await expect.poll(() => track.evaluate((t: MediaStreamTrack) => t.readyState)).toBe('ended');
  await expect.poll(() => video.evaluate((v: HTMLVideoElement) => v.paused)).toBe(true);
  await page.getByRole('button', { name: /Settings.*Language/ }).click();
  await page.locator('.homebtn').click();
  await page.screenshot({ path: info.outputPath('home-320.png'), animations: 'disabled' });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', { name: /Continue Red Binder/ }).click();
  await expect.poll(live).toBe(true);
  expect(await requests()).toBe(2);

  // Inside the scanner the camera stays held: a capture under review only pauses it, Scan again resumes it.
  await expect(page.getByRole('button', { name: 'Scan now' })).toBeEnabled({ timeout: 60000 });
  track = await video.evaluateHandle((v: HTMLVideoElement) => (v.srcObject as MediaStream).getVideoTracks()[0]);
  await page.getByRole('button', { name: 'Scan now' }).click();
  await expect(page.locator('.bot .primary', { hasText: 'Scan again' })).toBeVisible({ timeout: 60000 });
  expect(await track.evaluate((t: MediaStreamTrack) => [t.readyState, t.enabled])).toEqual(['live', false]);
  await page.locator('.bot .primary', { hasText: 'Scan again' }).click();
  await expect.poll(live).toBe(true);
  expect(await track.evaluate((t: MediaStreamTrack) => t.readyState)).toBe('live');
  expect(await requests()).toBe(2);

  track = await video.evaluateHandle((v: HTMLVideoElement) => (v.srcObject as MediaStream).getVideoTracks()[0]);
  await page.locator('.homebtn').click();
  await page.getByRole('button', { name: 'Finish Red Binder', exact: true }).click();
  await expect.poll(() => track.evaluate((t: MediaStreamTrack) => t.readyState)).toBe('ended');
  expect(await video.evaluate((v: HTMLVideoElement) => v.srcObject)).toBeNull();
});

test('named binders keep pages across workflows, reload, finish and resume', async ({ page }, info) => {
  await page.addInitScript(() => {
    localStorage.setItem('tcg-test-run', '1');
    if (!localStorage.getItem('tcg-scanner-settings-v1')) localStorage.setItem('tcg-scanner-settings-v1', JSON.stringify({ mode: 'manual' }));
  });
  await page.goto('/?mock=' + (process.env.MOCK ?? '4QxpRG4E6d0-312-324.webm'));
  const scan = page.getByRole('button', { name: 'Scan now', exact: true });
  await expect(scan).toBeEnabled({ timeout: 120000 });
  await page.locator('.homebtn').click();
  await page.getByRole('button', { name: 'New binder…', exact: true }).click();
  await page.getByRole('textbox', { name: 'New binder name (optional)' }).fill('Red Binder');
  await page.getByRole('button', { name: 'Start', exact: true }).click();
  await expect(scan).toBeEnabled();
  await scan.click();
  await expect(page.getByRole('button', { name: /^Check \d+ cards?$|^Next scan$/ })).toBeVisible({ timeout: 120000 });
  // A new session preserves the unresolved foreground capture first.
  await page.locator('.homebtn').click();
  await page.getByRole('button', { name: 'New binder…', exact: true }).click();
  await page.getByRole('textbox', { name: 'New binder name (optional)' }).fill('Blue Binder');
  await page.getByRole('button', { name: 'Start', exact: true }).click();
  await expect(page.locator('.chip')).toContainText('Blue Binder · Page 1');
  await page.locator('.homebtn').click();
  await page.getByRole('button', { name: 'Finish Blue Binder', exact: true }).click();
  await page.locator('.hbtn', { hasText: 'Collection' }).click();
  await expect(page.locator('.prow')).toHaveCount(1);
  await page.getByRole('button', { name: 'Continue Red Binder', exact: true }).click();
  await expect(page.locator('.chip')).toContainText('Red Binder · Page 2');
  await page.locator('.homebtn').click();
  await page.getByRole('button', { name: 'Batch review', exact: true }).click();
  await page.getByRole('button', { name: /Continue Red Binder/ }).click();
  await expect(scan).toBeEnabled();
  await scan.click();
  await expect(page.getByRole('heading', { name: 'Page 2 saved' })).toBeVisible({ timeout: 60000 });
  await page.reload();
  await expect(scan).toBeEnabled({ timeout: 120000 });
  await expect(page.locator('.chip')).toContainText('Red Binder · Page 3');
  await page.locator('.homebtn').click();
  await page.getByRole('button', { name: 'Finish Red Binder', exact: true }).click();
  await page.locator('.hbtn', { hasText: 'Collection' }).click();
  await expect(page.locator('.prow')).toHaveCount(2);
  await expect(page.locator('.shead .sn').filter({ hasText: 'Red Binder' })).toHaveCount(1);
  // In-app dialog (not the browser prompt): prefilled, Cancel keeps the name.
  await page.getByRole('button', { name: 'Rename Red Binder', exact: true }).click();
  const rename = page.getByRole('dialog', { name: 'Rename binder' });
  await expect(rename.getByRole('textbox', { name: 'Binder name' })).toHaveValue('Red Binder');
  await rename.getByRole('button', { name: 'Cancel' }).click();
  await expect(rename).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Continue Red Binder', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Rename Red Binder', exact: true }).click();
  await rename.getByRole('textbox', { name: 'Binder name' }).fill('  Trade Binder  ');
  await rename.getByRole('button', { name: 'Rename', exact: true }).click();
  await expect(page.locator('.shead .sn').filter({ hasText: 'Trade Binder' })).toHaveCount(1);
  await expect(page.locator('.prow')).toHaveCount(2);
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('tcg-scanner-sessions-v2')!).sessions.some((s: { name: string }) => s.name === 'Trade Binder'))).toBe(true);
  await page.screenshot({ path: info.outputPath('collection.png'), animations: 'disabled' });
  await page.getByRole('button', { name: 'Continue Trade Binder', exact: true }).click();
  await expect(page.locator('.chip')).toContainText('Trade Binder · Page 3');
});
