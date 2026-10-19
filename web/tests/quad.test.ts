import { describe, expect, it } from 'vitest';
import { boxCorners, letterboxFor, portraitQuad, readingOrder } from '../src/vision/quad';
import { decodePose } from '../src/vision/pose';
import { f32, hasFixtures, json, type DetectorFixture } from './fixtures';

const fixtures = hasFixtures ? json<{ fixtures: DetectorFixture[] }>('detector.json').fixtures : [];

describe.skipIf(!hasFixtures)('detector decoding parity with Ultralytics', () => {
  it('matches the letterbox transform', () => {
    for (const f of fixtures) {
      const lb = letterboxFor(...f.letterbox.source_size, f.input_size ?? 320);
      expect([lb.padX, lb.padY]).toEqual([f.letterbox.pad_x, f.letterbox.pad_y]);
      expect(lb.scale).toBeCloseTo(f.letterbox.scale, 10);
    }
  });

  it('decodes the same boxes, scores and order', () => {
    for (const f of fixtures) {
      const anchors = f.output_shape[2];
      const lb = letterboxFor(...f.letterbox.source_size, f.input_size ?? 640);
      const dets = decodePose(f32(f.output), anchors, lb, 0.25, 0.7, 300, false);
      expect(dets.length).toBe(f.ultralytics_conf.length);
      dets.forEach((d, i) => {
        expect(d.score).toBeCloseTo(f.ultralytics_conf[i], 4);
        d.corners.forEach((p, k) => p.forEach((v, axis) => expect(Math.abs(v - f.ultralytics_corners![i][k][axis])).toBeLessThan(0.05)));
      });
    }
  });

  it('keeps a variable-length instance list (zero, nine, fourteen)', () => {
    const counts = fixtures.map((f) => {
      const lb = letterboxFor(...f.letterbox.source_size, f.input_size ?? 640);
      return decodePose(f32(f.output), f.output_shape[2], lb, 0.25, 0.7, 300, false).length;
    });
    expect(counts).toContain(0);
    expect(Math.max(...counts)).toBeGreaterThan(9);
  });
});

describe('quad ordering', () => {
  it('orders a rotated card as portrait TL, TR, BR, BL', () => {
    // Landscape-encoded rotated box of a portrait card rotated 10 degrees.
    const box = { cx: 100, cy: 200, w: 88, h: 63, angle: Math.PI / 2 + 0.17, score: 1 };
    const [tl, tr, br, bl] = portraitQuad(boxCorners(box));
    expect(tl[1]).toBeLessThan(bl[1]);
    expect(tr[1]).toBeLessThan(br[1]);
    expect(Math.hypot(tr[0] - tl[0], tr[1] - tl[1])).toBeCloseTo(63, 5);
    expect(Math.hypot(bl[0] - tl[0], bl[1] - tl[1])).toBeCloseTo(88, 5);
    expect(tr[0]).toBeGreaterThan(tl[0]);
  });

  it('numbers cards in reading order', () => {
    const at = (cx: number, cy: number) => ({ box: { cx, cy, w: 88, h: 63, angle: 0, score: 1 } });
    const items = [at(300, 105), at(100, 98), at(200, 300), at(100, 302), at(200, 100)];
    expect(readingOrder(items).map((i) => [i.box.cx, i.box.cy])).toEqual([[100, 98], [200, 100], [300, 105], [100, 302], [200, 300]]);
  });
});
