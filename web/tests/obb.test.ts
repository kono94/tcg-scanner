import { describe, expect, it } from 'vitest';
import { boxCorners, decodeObb, letterboxFor, portraitQuad, readingOrder } from '../src/vision/obb';
import { f32, hasFixtures, json, type DetectorFixture } from './fixtures';

const fixtures = hasFixtures ? json<{ fixtures: DetectorFixture[] }>('detector.json').fixtures : [];

describe.skipIf(!hasFixtures)('OBB decoding parity with Ultralytics', () => {
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
      const boxes = decodeObb(f32(f.output), anchors, letterboxFor(...f.letterbox.source_size, f.input_size ?? 320));
      expect(boxes.length).toBe(f.ultralytics_conf.length);
      boxes.forEach((b, i) => {
        const [cx, cy, w, h, r] = f.ultralytics_obb_xywhr[i];
        expect(b.score).toBeCloseTo(f.ultralytics_conf[i], 4);
        expect(Math.abs(b.cx - cx)).toBeLessThan(0.01);
        expect(Math.abs(b.cy - cy)).toBeLessThan(0.01);
        expect(Math.abs(b.w - w)).toBeLessThan(0.01);
        expect(Math.abs(b.h - h)).toBeLessThan(0.01);
        expect(Math.abs(b.angle - r)).toBeLessThan(0.002);
      });
    }
  });

  it('keeps a variable-length instance list (zero, nine, fourteen)', () => {
    const counts = fixtures.map((f) => decodeObb(f32(f.output), f.output_shape[2], letterboxFor(...f.letterbox.source_size, f.input_size ?? 320)).length);
    expect(counts).toContain(0);
    expect(Math.max(...counts)).toBeGreaterThan(9);
  });
});

describe('quad ordering', () => {
  it('orders a rotated card as portrait TL, TR, BR, BL', () => {
    // Landscape-encoded OBB of a portrait card rotated 10 degrees.
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
