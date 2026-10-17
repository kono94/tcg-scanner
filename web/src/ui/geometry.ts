// View-space geometry: capture source pixels -> view pixels, and quads -> CSS matrix3d.
import type { Point, Quad } from '../vision/image';
import { invert3, perspectiveTransform } from '../vision/preprocess';

/** object-fit: contain mapping of a source image into a box. */
export interface ContainFit {
  scale: number;
  ox: number;
  oy: number;
}

export function containFit(sw: number, sh: number, bw: number, bh: number): ContainFit {
  const scale = Math.min(bw / sw, bh / sh);
  return { scale, ox: (bw - sw * scale) / 2, oy: (bh - sh * scale) / 2 };
}

/**
 * Hidden capture border: the preview is zoomed so this fraction of the capture is outside the box
 * on every side. Cards that look complete at the preview's edge are then complete in the photo.
 */
export const CAPTURE_MARGIN = 0.05;
export const PREVIEW_ZOOM = 1 / (1 - 2 * CAPTURE_MARGIN);

/** Like containFit, then zoomed about the centre (the captured photo shown with the preview's zoom). */
export function zoomedFit(sw: number, sh: number, bw: number, bh: number, zoom: number): ContainFit {
  const f = containFit(sw, sh, bw, bh);
  const scale = f.scale * zoom;
  return { scale, ox: (bw - sw * scale) / 2, oy: (bh - sh * scale) / 2 };
}

export function mapQuad(q: Quad, fit: ContainFit, dx = 0, dy = 0): Quad {
  return q.map(([x, y]) => [x * fit.scale + fit.ox + dx, y * fit.scale + fit.oy + dy] as Point) as unknown as Quad;
}

export function rectQuad(r: { left: number; top: number; width: number; height: number }): Quad {
  return [[r.left, r.top], [r.left + r.width, r.top], [r.left + r.width, r.top + r.height], [r.left, r.top + r.height]];
}

/** CSS matrix3d that maps a w x h element (origin 0 0) onto the quad. */
export function quadMatrix3d(w: number, h: number, q: Quad): string {
  const H = perspectiveTransform([[0, 0], [w, 0], [w, h], [0, h]], q);
  // Guard against degenerate quads mid-flight.
  try { invert3(H); } catch { return 'scale(0)'; }
  const [a, b, c, d, e, f, g, i, j] = H;
  return `matrix3d(${[a, d, 0, g, b, e, 0, i, 0, 0, 1, 0, c, f, 0, j].map((v) => +v.toFixed(8)).join(',')})`;
}

export function centroid(q: Quad): Point {
  return [(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4];
}

/** Corner-bracket path: short strokes along both edges at each corner. */
export function bracketPath(q: Quad, length = 12): string {
  const parts: string[] = [];
  for (let i = 0; i < 4; i++) {
    const p = q[i], prev = q[(i + 3) % 4], next = q[(i + 1) % 4];
    const seg = (to: Point) => {
      const dx = to[0] - p[0], dy = to[1] - p[1], len = Math.hypot(dx, dy) || 1;
      const l = Math.min(length, len * 0.3);
      return [p[0] + (dx / len) * l, p[1] + (dy / len) * l];
    };
    const a = seg(prev), b = seg(next);
    parts.push(`M${a[0].toFixed(1)},${a[1].toFixed(1)}L${p[0].toFixed(1)},${p[1].toFixed(1)}L${b[0].toFixed(1)},${b[1].toFixed(1)}`);
  }
  return parts.join('');
}

/** Source-pixel rectangle of the video that an object-fit: cover box of size `box` shows. */
export function visibleRegion(video: { videoWidth: number; videoHeight: number }, box: { width: number; height: number }): [number, number, number, number] {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!box.width || !box.height) return [0, 0, vw, vh];
  const boxAspect = box.height / box.width;
  if (vh / vw > boxAspect) {
    const sh = Math.round(vw * boxAspect);
    return [0, Math.round((vh - sh) / 2), vw, sh];
  }
  const sw = Math.round(vh / boxAspect);
  return [Math.round((vw - sw) / 2), 0, sw, vh];
}
