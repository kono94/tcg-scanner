// Decoding of the YOLO-pose card-corner detector: axis-aligned box + 4 corner keypoints
// (TL, TR, BR, BL, upright portrait order). Port of Ultralytics non_max_suppression (greedy IoU)
// and scale_coords. Coordinates: detector input (letterboxed square) -> capture source pixels.
import type { Point, Quad } from './image';
import type { Letterbox, OrientedBox } from './quad';

export interface CornerDetection {
  score: number;
  /** Axis-aligned box in capture pixels. */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  corners: Quad;
  /** Per-corner visibility confidence. */
  visibility: [number, number, number, number];
}

function iou(a: CornerDetection, b: CornerDetection): number {
  const w = Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0));
  const h = Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
  const inter = w * h;
  return inter / ((a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter);
}

/** Decode a (1, 5 + 4*3, N) single-class pose tensor. Highest score first. */
export function decodePose(output: Float32Array, anchors: number, lb: Letterbox, conf = 0.25, iouThreshold = 0.7, maxDet = 300, shapeFilter = true): CornerDetection[] {
  const rows = output.length / anchors;
  if (rows !== 17) throw new Error(`Expected single-class 4-keypoint pose output with 17 rows, got ${rows}`);
  const cand: CornerDetection[] = [];
  const unx = (x: number) => (x - lb.padX) / lb.scale, uny = (y: number) => (y - lb.padY) / lb.scale;
  for (let i = 0; i < anchors; i++) {
    const score = output[4 * anchors + i];
    if (score <= conf) continue;
    const cx = output[i], cy = output[anchors + i], w = output[2 * anchors + i], h = output[3 * anchors + i];
    const corners: Point[] = [];
    const vis: number[] = [];
    for (let k = 0; k < 4; k++) {
      corners.push([unx(output[(5 + 3 * k) * anchors + i]), uny(output[(6 + 3 * k) * anchors + i])]);
      vis.push(output[(7 + 3 * k) * anchors + i]);
    }
    cand.push({ score, x0: unx(cx - w / 2), y0: uny(cy - h / 2), x1: unx(cx + w / 2), y1: uny(cy + h / 2),
      corners: corners as unknown as Quad, visibility: vis as [number, number, number, number] });
  }
  cand.sort((a, b) => b.score - a.score);
  // Drop invalid or excessively stretched quads that do not fill their detection box.
  if (shapeFilter) for (let i = cand.length - 1; i >= 0; i--) if (!plausibleCard(cand[i])) cand.splice(i, 1);
  // Greedy NMS (torchvision.ops.nms); IoU is scale-invariant, so source pixels are fine.
  const kept: CornerDetection[] = [];
  for (const c of cand) {
    if (kept.every((k) => iou(k, c) <= iouThreshold)) kept.push(c);
    if (kept.length >= maxDet) break;
  }
  return kept;
}

/** Rotated-box summary of a corner quad (for reading order and sizes). */
export function boxOfQuad(q: Quad, score: number): OrientedBox {
  const d = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const width = (d(q[0], q[1]) + d(q[3], q[2])) / 2, height = (d(q[0], q[3]) + d(q[1], q[2])) / 2;
  const cx = (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4, cy = (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4;
  const long = Math.max(width, height), short = Math.min(width, height);
  const angle = height >= width ? Math.atan2(q[3][1] - q[0][1], q[3][0] - q[0][0]) : Math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]);
  return { cx, cy, w: long, h: short, angle: ((angle % Math.PI) + Math.PI) % Math.PI, score };
}

/** A convex, bounded-aspect quad that fills most of its box; perspective can make cards square. */
export function plausibleCard(d: CornerDetection): boolean {
  const q = d.corners;
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
    const z = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (z === 0 || (sign && Math.sign(z) !== sign)) return false;
    sign = Math.sign(z);
  }
  const len = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const w = (len(q[0], q[1]) + len(q[3], q[2])) / 2, h = (len(q[0], q[3]) + len(q[1], q[2])) / 2;
  const ratio = Math.max(w, h) / Math.min(w, h);
  if (ratio > 2.0) return false;
  let area = 0;
  for (let i = 0; i < 4; i++) area += q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1];
  const boxArea = (d.x1 - d.x0) * (d.y1 - d.y0);
  return boxArea > 0 && Math.abs(area) / 2 > 0.45 * boxArea;
}
