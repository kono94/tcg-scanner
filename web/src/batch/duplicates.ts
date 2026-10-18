// Cheap appearance evidence for probable duplicate captures, never card identity evidence.
import type { RGBImage } from '../vision/image';

const COLS = 6, ROWS = 8, SIZE = COLS * ROWS * 3;
const decoded = new Map<string, number[]>();

/** Block colours of a rectified card, brightness-centred to tolerate exposure changes. */
export function appearanceFingerprint(image: RGBImage): string {
  const values: number[] = [];
  for (let by = 0; by < ROWS; by++) for (let bx = 0; bx < COLS; bx++) {
    const sum = [0, 0, 0];
    let n = 0;
    // Ignore the outer 5%, where detector corner jitter is most visible.
    const x0 = Math.floor(image.width * (0.05 + 0.9 * bx / COLS));
    const x1 = Math.max(x0 + 1, Math.floor(image.width * (0.05 + 0.9 * (bx + 1) / COLS)));
    const y0 = Math.floor(image.height * (0.05 + 0.9 * by / ROWS));
    const y1 = Math.max(y0 + 1, Math.floor(image.height * (0.05 + 0.9 * (by + 1) / ROWS)));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const i = (y * image.width + x) * 3;
      for (let c = 0; c < 3; c++) sum[c] += image.data[i + c];
      n++;
    }
    values.push(...sum.map((s) => s / n));
  }
  const mean = values.reduce((a, b) => a + b, 0) / SIZE;
  return 'a1:' + values.map((v) => Math.max(0, Math.min(255, Math.round(v - mean + 128))).toString(16).padStart(2, '0')).join('');
}

function decode(value: string): number[] | null {
  if (decoded.has(value)) return decoded.get(value)!;
  if (!/^a1:[0-9a-f]+$/.test(value) || value.length !== 3 + SIZE * 2) return null;
  const result = Array.from({ length: SIZE }, (_, i) => parseInt(value.slice(3 + i * 2, 5 + i * 2), 16));
  if (decoded.size >= 256) decoded.delete(decoded.keys().next().value!);
  decoded.set(value, result);
  return result;
}

/** Threshold is a conservative starting heuristic, not calibrated recognition evidence. */
export function similarAppearance(a: string, b: string): boolean {
  const x = decode(a), y = decode(b);
  if (!x || !y) return false;
  let normal = 0, rotated = 0;
  for (let i = 0; i < SIZE; i++) {
    normal += Math.abs(x[i] - y[i]);
    const r = (COLS * ROWS - 1 - Math.floor(i / 3)) * 3 + i % 3;
    rotated += Math.abs(x[i] - y[r]);
  }
  return Math.min(normal, rotated) / SIZE <= 12;
}

/** One-to-one matching preserves repeated copies and supports any detected card count. */
export function probableDuplicate(a: readonly string[], b: readonly string[]): boolean {
  if (!a.length || !b.length || Math.min(a.length, b.length) / Math.max(a.length, b.length) < 0.85) return false;
  const matches = new Map<number, number>();
  function assign(i: number, seen: Set<number>): boolean {
    for (let j = 0; j < b.length; j++) {
      if (seen.has(j) || !similarAppearance(a[i], b[j])) continue;
      seen.add(j);
      const previous = matches.get(j);
      if (previous === undefined || assign(previous, seen)) { matches.set(j, i); return true; }
    }
    return false;
  }
  for (let i = 0; i < a.length; i++) assign(i, new Set());
  return matches.size >= Math.ceil(Math.max(a.length, b.length) * 0.85);
}
