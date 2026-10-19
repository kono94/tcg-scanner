import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { letterboxFor } from '../src/vision/quad';
import { boxOfQuad, decodePose, plausibleCard } from '../src/vision/pose';
import { EXPORT_DIR } from './fixtures';

const dir = join(EXPORT_DIR, '../pose-fixture');
const has = existsSync(join(dir, 'pose.json'));

describe.skipIf(!has)('corner (pose) decoding parity with Ultralytics', () => {
  it('decodes the same boxes, scores and corners', () => {
    const f = JSON.parse(readFileSync(join(dir, 'pose.json'), 'utf8'));
    const raw = readFileSync(join(dir, 'output.f32'));
    const out = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    const dets = decodePose(out, f.output_shape[2], letterboxFor(f.source_size[0], f.source_size[1], f.input_size), 0.25, 0.7, 300, false);
    expect(dets.length).toBe(f.scores.length);
    dets.forEach((d, i) => {
      expect(d.score).toBeCloseTo(f.scores[i], 4);
      d.corners.forEach((p, k) => {
        expect(Math.abs(p[0] - f.corners[i][k][0])).toBeLessThan(0.05);
        expect(Math.abs(p[1] - f.corners[i][k][1])).toBeLessThan(0.05);
      });
    });
  });
});

describe('quad summary', () => {
  it('gives the long side, centre and portrait angle of a corner quad', () => {
    const b = boxOfQuad([[0, 0], [63, 0], [63, 88], [0, 88]], 0.9);
    expect([b.cx, b.cy, b.w, b.h]).toEqual([31.5, 44, 88, 63]);
    expect(b.angle).toBeCloseTo(Math.PI / 2, 6);
  });
});

describe('card-shape filter', () => {
  it('keeps perspective-foreshortened cards with square or nearly square outlines through decoding', () => {
    for (const height of [63, 66]) {
      const corners = [[0, 0], [63, 0], [63, height], [0, height]];
      const out = new Float32Array([31.5, height / 2, 63, height, 0.9, ...corners.flatMap(([x, y]) => [x, y, 1])]);
      const dets = decodePose(out, 1, letterboxFor(100, 100, 100));
      expect(dets).toHaveLength(1);
      expect(dets[0].corners).toEqual(corners);
    }
  });

  it('rejects crossed or stretched corner sets', async () => {
    const det = (corners: [number, number][]) => ({ score: 1, x0: 0, y0: 0, x1: 63, y1: 88, corners: corners as any, visibility: [1, 1, 1, 1] as [number, number, number, number] });
    expect(plausibleCard(det([[0, 0], [63, 0], [63, 88], [0, 88]]))).toBe(true);
    expect(plausibleCard(det([[0, 0], [63, 88], [63, 0], [0, 88]]))).toBe(false); // crossed
    expect(plausibleCard(det([[0, 0], [63, 0], [60, 3], [0, 88]]))).toBe(false); // collapsed corner
    expect(plausibleCard(det([[0, 0], [20, 0], [20, 88], [0, 88]]))).toBe(false); // stretched
  });
});
