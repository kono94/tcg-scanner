// Card quad geometry shared by detection, sessions and overlays. Coordinates: detector input
// (letterboxed square) -> capture source pixels.
import type { Point, Quad } from './image';

/** Rotated-rectangle summary of a detected card in capture source pixels (see pose.ts boxOfQuad):
 * centre, long side, short side, angle in [0, pi). Used for reading order and overlays. */
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
