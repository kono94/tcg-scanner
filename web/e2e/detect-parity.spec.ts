// The in-browser detection path (canvas letterbox + WebGPU/WASM + TS decode) vs Ultralytics on one frame.
import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';

test('browser detection matches Ultralytics on a frame', async ({ page }) => {
  test.skip(!process.env.PY_DETECTIONS, 'Set PY_DETECTIONS to an Ultralytics result JSON for public/media/debug-frame.png');
  const expected = JSON.parse(readFileSync(process.env.PY_DETECTIONS!, 'utf8')) as { xywhr: number[][]; conf: number[] };
  await page.addInitScript(() => localStorage.setItem('tcg-test-run', '1'));
  await page.goto('/');
  await page.waitForFunction(() => 'tcgDebug' in window);
  const got = await page.evaluate(async () => {
    const d = (window as any).tcgDebug;
    await d.ready();
    return d.detectUrl('media/debug-frame.png');
  });
  console.log('python', expected.xywhr.length, 'browser', got.boxes.length);
  for (const [cx, cy, w, h, r] of expected.xywhr) {
    const best = got.boxes.reduce((a: any, b: any) => (Math.hypot(b.cx - cx, b.cy - cy) < Math.hypot(a.cx - cx, a.cy - cy) ? b : a));
    console.log(`py ${cx},${cy} ${w}x${h} r${r} | js ${best.cx.toFixed(1)},${best.cy.toFixed(1)} ${best.w.toFixed(1)}x${best.h.toFixed(1)} r${best.angle.toFixed(2)}`);
    expect(Math.hypot(best.cx - cx, best.cy - cy)).toBeLessThan(12); // Ultralytics predict uses a rectangular letterbox; the app uses a square one.
  }
});
