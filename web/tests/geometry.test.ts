import { describe, expect, it } from 'vitest';
import { containFit, mapQuad, quadMatrix3d } from '../src/ui/geometry';
import type { Quad } from '../src/vision/image';

describe('view geometry', () => {
  it('maps capture pixels into an object-fit: contain box', () => {
    const fit = containFit(3840, 2160, 360, 400);
    expect(fit.scale).toBeCloseTo(360 / 3840);
    expect(fit.ox).toBe(0);
    expect(fit.oy).toBeCloseTo((400 - 2160 * fit.scale) / 2);
    const q = mapQuad([[0, 0], [3840, 0], [3840, 2160], [0, 2160]], fit);
    expect(q[2][0]).toBeCloseTo(360);
  });

  it('builds a matrix3d that sends element corners onto the quad', () => {
    const quad: Quad = [[10, 20], [110, 30], [100, 160], [5, 150]];
    const m = quadMatrix3d(224, 312, quad).slice(9, -1).split(',').map(Number);
    // Column-major 4x4; apply to (x, y, 0, 1).
    const apply = (x: number, y: number) => {
      const X = m[0] * x + m[4] * y + m[12], Y = m[1] * x + m[5] * y + m[13], W = m[3] * x + m[7] * y + m[15];
      return [X / W, Y / W];
    };
    [[0, 0], [224, 0], [224, 312], [0, 312]].forEach(([x, y], i) => {
      const [X, Y] = apply(x, y);
      expect(X).toBeCloseTo(quad[i][0], 3);
      expect(Y).toBeCloseTo(quad[i][1], 3);
    });
  });
});

describe('visible capture region', () => {
  it('captures the centre crop an object-fit: cover box shows', async () => {
    const { visibleRegion } = await import('../src/ui/geometry');
    // Portrait 1440x2560 camera in a 3:4 box: full width, cropped top and bottom.
    expect(visibleRegion({ videoWidth: 1440, videoHeight: 2560 }, { width: 375, height: 500 })).toEqual([0, 320, 1440, 1920]);
    // Landscape video in a box of the same aspect: everything.
    expect(visibleRegion({ videoWidth: 3840, videoHeight: 2160 }, { width: 360, height: 202.5 })).toEqual([0, 0, 3840, 2160]);
  });
});

describe('hidden capture border', () => {
  it('shows the captured photo with the preview zoom, border outside the box', async () => {
    const { zoomedFit, PREVIEW_ZOOM, CAPTURE_MARGIN } = await import('../src/ui/geometry');
    const f = zoomedFit(1440, 1920, 375, 500, PREVIEW_ZOOM);
    // The inner 90% of the capture spans the box exactly.
    expect(1440 * CAPTURE_MARGIN * f.scale + f.ox).toBeCloseTo(0, 6);
    expect(1440 * (1 - CAPTURE_MARGIN) * f.scale + f.ox).toBeCloseTo(375, 6);
    expect(1920 * CAPTURE_MARGIN * f.scale + f.oy).toBeCloseTo(0, 6);
  });
});
