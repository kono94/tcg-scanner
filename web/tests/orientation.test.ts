import { describe, expect, it } from 'vitest';

import { boxCorners, portraitQuad } from '../src/vision/quad';


describe('box orientation', () => {
  it('keeps a near-square loose box upright instead of sideways', () => {
    // Benn Beckman case: card nearly upright, box almost square and tilted ~15 degrees.
    const box = { cx: 300, cy: 300, w: 320, h: 300, angle: (15 * Math.PI) / 180, score: 1 };
    const q = portraitQuad(boxCorners(box));
    const tilt = Math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]) * 180 / Math.PI;
    expect(Math.abs(tilt)).toBeLessThan(46);
    expect(q[0][1]).toBeLessThan(q[3][1]);
  });
});
