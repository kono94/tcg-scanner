// TypeScript port of training/card_recognizer/canonical.py (contract tcg-full-card-v1).
// The perspective warp and INTER_AREA resize follow OpenCV's arithmetic closely so that
// query tensors match the gallery; parity is measured against Python golden fixtures.
import type { Quad, RGBImage } from './image';

export const PREPROCESSING_VERSION = 'tcg-full-card-v1';
export const CANONICAL_WIDTH = 224;
export const CANONICAL_HEIGHT = 312;
export const ENCODER_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406].map(Math.fround);
const STD = [0.229, 0.224, 0.225].map(Math.fround);

export type Matrix3 = [number, number, number, number, number, number, number, number, number];

/** OpenCV cvRound: round half to even. */
function cvRound(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function solve(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
    if (Math.abs(m[p][c]) < 1e-12) throw new Error('Degenerate homography');
    [m[c], m[p]] = [m[p], m[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = m[r][c] / m[c][c];
      for (let k = c; k <= n; k++) m[r][k] -= f * m[c][k];
    }
  }
  return m.map((row, i) => row[n] / row[i]);
}

/** Homography mapping src points to dst points (cv2.getPerspectiveTransform). */
export function perspectiveTransform(src: Quad, dst: Quad): Matrix3 {
  const a: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    a.push([x, y, 1, 0, 0, 0, -x * u, -y * u]);
    b.push(u);
    a.push([0, 0, 0, x, y, 1, -x * v, -y * v]);
    b.push(v);
  }
  const h = solve(a, b);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

export function invert3(m: Matrix3): Matrix3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-15) throw new Error('Singular transform');
  const s = 1 / det;
  return [A * s, -(b * i - c * h) * s, (b * f - c * e) * s,
    B * s, (a * i - c * g) * s, -(a * f - c * d) * s,
    C * s, -(a * h - b * g) * s, (a * e - b * d) * s];
}

// OpenCV bilinear fixed-point table: 32 sub-pixel steps, weights scaled to 2^15.
const INTER_BITS = 5, TAB = 1 << INTER_BITS, COEF_BITS = 15, COEF_SCALE = 1 << COEF_BITS;
const BILINEAR_TAB = (() => {
  const tab = new Int32Array(TAB * TAB * 4);
  for (let iy = 0; iy < TAB; iy++) {
    for (let ix = 0; ix < TAB; ix++) {
      const ay = Math.fround(iy / TAB), ax = Math.fround(ix / TAB);
      const fy = [Math.fround(1 - ay), ay], fx = [Math.fround(1 - ax), ax];
      const w = [0, 0, 0, 0];
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        w[k] = cvRound(Math.fround(fy[k >> 1] * fx[k & 1]) * COEF_SCALE);
        sum += w[k];
      }
      if (sum !== COEF_SCALE) {
        let lo = 0, hi = 0;
        for (let k = 1; k < 4; k++) {
          if (w[k] < w[lo]) lo = k;
          if (w[k] > w[hi]) hi = k;
        }
        const diff = sum - COEF_SCALE;
        if (diff < 0) w[hi] -= diff; else w[lo] -= diff;
      }
      tab.set(w, (iy * TAB + ix) * 4);
    }
  }
  return tab;
})();

/** cv2.warpPerspective(INTER_LINEAR, BORDER_CONSTANT 0) of the quad onto a width x height card. */
export function warpCard(image: RGBImage, corners: Quad, width = CANONICAL_WIDTH, height = CANONICAL_HEIGHT): RGBImage {
  const dst: Quad = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]];
  const m = invert3(perspectiveTransform(corners, dst));
  const out = new Uint8Array(width * height * 3);
  const { width: sw, height: sh, data: s } = image;
  for (let y = 0; y < height; y++) {
    const X0 = m[1] * y + m[2], Y0 = m[4] * y + m[5], W0 = m[7] * y + m[8];
    for (let x = 0; x < width; x++) {
      let W = W0 + m[6] * x;
      W = W ? TAB / W : 0;
      const X = cvRound((X0 + m[0] * x) * W), Y = cvRound((Y0 + m[3] * x) * W);
      const x0 = X >> INTER_BITS, y0 = Y >> INTER_BITS;
      const t = ((Y & (TAB - 1)) * TAB + (X & (TAB - 1))) * 4;
      const o = (y * width + x) * 3;
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let k = 0; k < 4; k++) {
          const px = x0 + (k & 1), py = y0 + (k >> 1);
          if (px >= 0 && py >= 0 && px < sw && py < sh) acc += BILINEAR_TAB[t + k] * s[(py * sw + px) * 3 + c];
        }
        out[o + c] = Math.min(255, Math.max(0, (acc + (1 << (COEF_BITS - 1))) >> COEF_BITS));
      }
    }
  }
  return { width, height, data: out };
}

interface AreaEntry { di: number; si: number; alpha: number }

function areaTable(ssize: number, dsize: number, scale: number): AreaEntry[] {
  const tab: AreaEntry[] = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale, fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) tab.push({ di: dx, si: sx1 - 1, alpha: Math.fround((sx1 - fsx1) / cellWidth) });
    for (let sx = sx1; sx < sx2; sx++) tab.push({ di: dx, si: sx, alpha: Math.fround(1 / cellWidth) });
    if (fsx2 - sx2 > 1e-3) tab.push({ di: dx, si: sx2, alpha: Math.fround(Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth) });
  }
  return tab;
}

/** cv2.resize(INTER_AREA) for shrinking by a non-integer factor, float32 accumulation. */
export function resizeArea(image: RGBImage, dw: number, dh: number): RGBImage {
  const { width: sw, height: sh, data: s } = image;
  if (dw > sw || dh > sh) throw new Error('resizeArea only shrinks');
  const xtab = areaTable(sw, dw, sw / dw), ytab = areaTable(sh, dh, sh / dh);
  const out = new Uint8Array(dw * dh * 3);
  const buf = new Float32Array(dw * 3), sum = new Float32Array(dw * 3);
  const flush = (dy: number) => {
    for (let i = 0; i < dw * 3; i++) out[dy * dw * 3 + i] = Math.min(255, Math.max(0, cvRound(sum[i])));
  };
  let prev = -1;
  for (const { di: dy, si: sy, alpha: beta } of ytab) {
    buf.fill(0);
    const row = sy * sw * 3;
    for (const { di, si, alpha } of xtab) {
      for (let c = 0; c < 3; c++) buf[di * 3 + c] = Math.fround(buf[di * 3 + c] + Math.fround(s[row + si * 3 + c] * alpha));
    }
    if (dy !== prev) {
      if (prev >= 0) flush(prev);
      for (let i = 0; i < dw * 3; i++) sum[i] = Math.fround(beta * buf[i]);
      prev = dy;
    } else {
      for (let i = 0; i < dw * 3; i++) sum[i] = Math.fround(sum[i] + Math.fround(beta * buf[i]));
    }
  }
  if (prev >= 0) flush(prev);
  return { width: dw, height: dh, data: out };
}

/** prepare_encoder_input: aspect-fit, mean-pad, normalize; returns CHW float32. */
export function encoderTensor(card: RGBImage, size = ENCODER_SIZE): Float32Array {
  const scale = Math.min(size / card.width, size / card.height);
  const rw = Math.max(1, Math.round(card.width * scale)), rh = Math.max(1, Math.round(card.height * scale));
  if (scale >= 1) throw new Error('Canonical cards are larger than the encoder input; enlarging is not part of this path');
  const resized = resizeArea(card, rw, rh);
  const x0 = (size - rw) >> 1, y0 = (size - rh) >> 1;
  const out = new Float32Array(3 * size * size);
  const pad = MEAN.map((m, c) => Math.fround(Math.fround(Math.fround(Math.fround(m * 255) / 255) - m) / STD[c]));
  for (let c = 0; c < 3; c++) out.fill(pad[c], c * size * size, (c + 1) * size * size);
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      const p = (y * rw + x) * 3;
      for (let c = 0; c < 3; c++) {
        const v = Math.fround(Math.fround(Math.fround(resized.data[p + c] / 255) - MEAN[c]) / STD[c]);
        out[c * size * size + (y + y0) * size + x + x0] = v;
      }
    }
  }
  return out;
}
