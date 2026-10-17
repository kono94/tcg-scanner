import { describe, expect, it } from 'vitest';
import { encoderTensor, warpCard } from '../src/vision/preprocess';
import type { Quad } from '../src/vision/image';
import { f32, hasFixtures, json, rgb, type EncoderFixture } from './fixtures';

const fixtures = hasFixtures ? json<{ fixtures: EncoderFixture[] }>('encoder.json').fixtures : [];

function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

describe.skipIf(!hasFixtures)('tcg-full-card-v1 parity with Python/OpenCV', () => {
  it('warps full-bounds sources exactly like cv2.warpPerspective', () => {
    for (const f of fixtures) {
      const [w, h] = f.source_size;
      const out = warpCard(rgb(f.source_rgb, w, h), f.corners as unknown as Quad);
      expect(maxDiff(out.data, rgb(f.canonical_rgb, 224, 312).data)).toBe(0);
    }
  });

  it('warps a tilted perspective quad like cv2.warpPerspective', () => {
    const f = json<{ source_rgb: string; source_size: [number, number]; corners: [number, number][]; expected_rgb: string }>('warp.json');
    const out = warpCard(rgb(f.source_rgb, ...f.source_size), f.corners as unknown as Quad);
    expect(maxDiff(out.data, rgb(f.expected_rgb, 224, 312).data)).toBeLessThanOrEqual(1);
  });

  it('builds encoder tensors exactly like prepare_encoder_input', () => {
    for (const f of fixtures) {
      expect(maxDiff(encoderTensor(rgb(f.canonical_rgb, 224, 312)), f32(f.input))).toBe(0);
    }
  });
});
