// Records the real per-card timeline and screenshots during processing (diagnostic, not an assertion of speed).
import { expect, test } from '@playwright/test';

test('timeline of one page scan', async ({ page }) => {
  const shots = process.env.SHOTS_DIR;
  // Manual trigger so the test chooses the capture moment.
  await page.addInitScript(() => localStorage.setItem('tcg-test-run', '1'));
  await page.addInitScript(() => localStorage.setItem('tcg-scanner-settings-v1', JSON.stringify({ mode: 'manual' })));
  await page.goto('/?mock=synthetic_replay.mp4');
  await expect(page.locator('.title')).not.toHaveText('Loading models', { timeout: 120000 });
  const runtime = await page.evaluate(() => (window as any).tcgDebug.runtime());
  for (let attempt = 0; attempt < 10; attempt++) {
    await page.waitForTimeout(300 + attempt * 170);
    await page.getByRole('button', { name: 'Scan now' }).click();
    const t0 = Date.now();
    const frames: string[] = [];
    let n = 0;
    while (Date.now() - t0 < 3000) {
      const st = await page.evaluate(() => (window as any).tcgDebug.page().status);
      if (shots && n < 12) { await page.screenshot({ path: `${shots}/t-${String(n).padStart(2, '0')}.png` }); frames.push(`${Date.now() - t0}ms ${st}`); n++; }
      if (st === 'settled' || st === 'failed') break;
    }
    const pg = await page.evaluate(() => {
      const p = (window as any).tcgDebug.page();
      return { n: p.instances.length, detectMs: p.detectMs, cards: p.instances.map((i: any) => ({ s: i.state, start: i.startedAt - p.capturedAt, end: i.settledAt - p.capturedAt, ms: i.ms })) };
    });
    if (pg.n === 0) { await page.getByRole('button', { name: 'Scan again' }).first().click(); continue; }
    console.log('runtime', JSON.stringify(runtime));
    console.log('detect', pg.detectMs?.toFixed(0), 'ms;', pg.n, 'cards');
    for (const c of pg.cards) console.log(`  ${c.s.padEnd(16)} start ${c.start.toFixed(0)} end ${c.end.toFixed(0)} | rectify ${c.ms?.rectify.toFixed(0)} embed ${c.ms?.embed.toFixed(0)} search ${c.ms?.search.toFixed(0)}`);
    console.log(frames.join(' | '));
    return;
  }
});
