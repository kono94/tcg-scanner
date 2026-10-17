// YOLO OBB output decoding, a port of the Ultralytics rotated NMS/post-processing used by the
// exported detector. Coordinates: detector input (letterboxed square) -> capture source pixels.
import type { Point, Quad } from './image';

/** Rotated box in capture source pixels: centre, long side, short side, angle in [0, pi). */
export interface OrientedBox {
  cx: number;
  cy: number;
  w: number;
  h: number;
  angle: number;
  score: number;
}

/** Letterbox from source pixels into the square detector input. */
export interface Letterbox {
  size: number;
  scale: number;
  padX: number;
  padY: number;
}

export function letterboxFor(sourceWidth: number, sourceHeight: number, size: number): Letterbox {
  const scale = Math.min(size / sourceWidth, size / sourceHeight);
  // Ultralytics scale_boxes pads with round((size - dim * gain) / 2 - 0.1).
  const padX = Math.round((size - sourceWidth * scale) / 2 - 0.1) || 0;
  const padY = Math.round((size - sourceHeight * scale) / 2 - 0.1) || 0;
  return { size, scale, padX, padY };
}

function covariance(w: number, h: number, r: number): [number, number, number] {
  const a = (w * w) / 12, b = (h * h) / 12;
  const c = Math.cos(r), s = Math.sin(r);
  return [a * c * c + b * s * s, a * s * s + b * c * c, (a - b) * c * s];
}

/** Probabilistic IoU between two rotated boxes (Ultralytics batch_probiou). */
export function probiou(p: OrientedBox, q: OrientedBox, eps = 1e-7): number {
  const [a1, b1, c1] = covariance(p.w, p.h, p.angle);
  const [a2, b2, c2] = covariance(q.w, q.h, q.angle);
  const A = a1 + a2, B = b1 + b2, C = c1 + c2;
  const den = A * B - C * C + eps;
  const t1 = ((A * (p.cy - q.cy) ** 2 + B * (p.cx - q.cx) ** 2) / den) * 0.25;
  const t2 = ((C * (q.cx - p.cx) * (p.cy - q.cy)) / den) * 0.5;
  const t3 = Math.log((A * B - C * C) / (4 * Math.sqrt(Math.max(0, a1 * b1 - c1 * c1) * Math.max(0, a2 * b2 - c2 * c2)) + eps) + eps) * 0.5;
  const bd = Math.min(100, Math.max(eps, t1 + t2 + t3));
  return 1 - Math.sqrt(1 - Math.exp(-bd) + eps);
}

/**
 * Decode a (1, 4 + classes + 1, N) OBB tensor for a single-class detector.
 * Returns boxes in capture source pixels, highest score first.
 */
export function decodeObb(output: Float32Array, anchors: number, lb: Letterbox, conf = 0.25, iou = 0.7, maxDet = 300): OrientedBox[] {
  const rows = output.length / anchors;
  if (rows !== 6) throw new Error(`Expected single-class OBB output with 6 rows, got ${rows}`);
  const candidates: OrientedBox[] = [];
  for (let i = 0; i < anchors; i++) {
    const score = output[4 * anchors + i];
    if (score > conf) {
      candidates.push({ cx: output[i], cy: output[anchors + i], w: output[2 * anchors + i], h: output[3 * anchors + i], angle: output[5 * anchors + i], score });
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  // Fast-NMS as in Ultralytics nms_rotated: a box survives if no higher-scored box overlaps it.
  const kept: OrientedBox[] = [];
  for (let j = 0; j < candidates.length; j++) {
    let suppressed = false;
    for (let i = 0; i < j && !suppressed; i++) suppressed = probiou(candidates[i], candidates[j]) >= iou;
    if (!suppressed) kept.push(candidates[j]);
    if (kept.length >= maxDet) break;
  }
  return kept.map((b) => {
    const longFirst = b.w > b.h;
    const angle = (((longFirst ? b.angle : b.angle + Math.PI / 2) % Math.PI) + Math.PI) % Math.PI;
    return {
      cx: (b.cx - lb.padX) / lb.scale,
      cy: (b.cy - lb.padY) / lb.scale,
      w: (longFirst ? b.w : b.h) / lb.scale,
      h: (longFirst ? b.h : b.w) / lb.scale,
      angle,
      score: b.score,
    };
  });
}

/** Rotated-rectangle corners (Ultralytics xywhr2xyxyxyxy order). */
export function boxCorners(b: OrientedBox): [Point, Point, Point, Point] {
  const c = Math.cos(b.angle), s = Math.sin(b.angle);
  const v1: Point = [(b.w / 2) * c, (b.w / 2) * s];
  const v2: Point = [(-b.h / 2) * s, (b.h / 2) * c];
  return [
    [b.cx + v1[0] + v2[0], b.cy + v1[1] + v2[1]],
    [b.cx + v1[0] - v2[0], b.cy + v1[1] - v2[1]],
    [b.cx - v1[0] - v2[0], b.cy - v1[1] - v2[1]],
    [b.cx - v1[0] + v2[0], b.cy - v1[1] + v2[1]],
  ];
}

/**
 * Order a card quad as portrait TL, TR, BR, BL in source pixels. When the quad is clearly
 * card-shaped, the short edges are top and bottom. When it is close to square (a loose box),
 * edge lengths say nothing, so the ordering that needs the least rotation from upright wins:
 * cards in binders and hands are mostly upright. Geometric order only; an upside-down card
 * stays upside down (artwork orientation is a separate question).
 */
export function portraitQuad(points: readonly Point[]): Quad {
  let p = [...points];
  // Clockwise in image coordinates (y down).
  const cross = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
  if (cross < 0) p = [p[0], p[3], p[2], p[1]];
  const len = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const e01 = len(p[0], p[1]) + len(p[2], p[3]), e12 = len(p[1], p[2]) + len(p[3], p[0]);
  const cardShaped = Math.max(e01, e12) / Math.min(e01, e12) >= 1.2;
  let best: Quad | null = null, bestTilt = Infinity;
  for (let k = 0; k < 4; k++) {
    const q = [p[k], p[(k + 1) % 4], p[(k + 2) % 4], p[(k + 3) % 4]] as unknown as Quad;
    const topShort = len(q[0], q[1]) <= len(q[1], q[2]);
    if (cardShaped && !topShort) continue;
    const tilt = Math.abs(Math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]));
    if (tilt < bestTilt) { bestTilt = tilt; best = q; }
  }
  return best!;
}

/** Stable reading order: rows by centre y (tolerance from card height), then x. */
export function readingOrder<T extends { box: OrientedBox }>(items: T[]): T[] {
  if (items.length < 2) return [...items];
  const sorted = [...items].sort((a, b) => a.box.cy - b.box.cy);
  const rowTol = 0.5 * Math.min(...items.map((i) => i.box.w));
  const rows: T[][] = [];
  for (const item of sorted) {
    const row = rows[rows.length - 1];
    if (row && Math.abs(item.box.cy - row[0].box.cy) < rowTol) row.push(item);
    else rows.push([item]);
  }
  return rows.flatMap((r) => r.sort((a, b) => a.box.cx - b.box.cx));
}
