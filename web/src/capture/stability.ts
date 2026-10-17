// Cheap scene-stability test on downscaled luma samples (docs/CAPTURE_CYCLE.md).
// Image evidence only; phone motion sensors are not required. Thresholds are uncalibrated.

export interface StabilitySample {
  /** Monotonic time in ms (performance.now). */
  t: number;
  /** Frame identity from the video (mediaTime or presented frame count); repeats mean a stale frame. */
  frameId: number;
  width: number;
  height: number;
  luma: Uint8Array;
  /** Edge strength of a higher-resolution sample (mean absolute Laplacian); optional. */
  sharpness?: number;
}

/** Mean absolute Laplacian of a luma image: drops when the frame is motion- or focus-blurred. */
export function sharpnessOf(luma: Uint8Array, width: number, height: number): number {
  let sum = 0, n = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      sum += Math.abs(4 * luma[i] - luma[i - 1] - luma[i + 1] - luma[i - width] - luma[i + width]);
      n++;
    }
  }
  return n ? sum / n : 0;
}

/** 2x2 box downscale (keeps the motion metric on its calibrated 64x48 scale). */
export function halve(luma: Uint8Array, width: number, height: number): Uint8Array {
  const w = width >> 1, h = height >> 1, out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = 2 * y * width + 2 * x;
      out[y * w + x] = (luma[i] + luma[i + 1] + luma[i + width] + luma[i + width + 1] + 2) >> 2;
    }
  }
  return out;
}

export type MotionTolerance = 'sensitive' | 'normal' | 'relaxed';

/**
 * Mean absolute luma change per sample (0-255 scale, 64x48) still counted as "steady", judged on the
 * median of the last three samples. From a hand-held iPhone over a binder page (2026-10-16): a steady
 * hand measures about 3-10 per sample, deliberate movement about 15-50. Still to be calibrated more widely.
 */
const TOLERANCE: Record<MotionTolerance, number> = { sensitive: 6, normal: 9, relaxed: 13 };

export interface StabilityState {
  steady: boolean;
  /** Continuous steady time, 0 while moving or reset. */
  steadyMs: number;
  motion: number | null;
  /** False while the frame is clearly less sharp than the best one of the last 1.5 s (blur). */
  sharp: boolean;
  reason: 'steady' | 'motion' | 'stale' | 'gap' | 'dark' | 'warming-up';
}

export class StabilityMonitor {
  private prev: StabilitySample | null = null;
  private steadySince: number | null = null;
  private lastFrameId: number | null = null;
  private lastNewFrameAt = 0;
  private recent: number[] = [];
  private sharpness: { t: number; v: number }[] = [];

  constructor(public tolerance: MotionTolerance = 'normal', private readonly maxGapMs = 400) {}

  reset(): void {
    this.prev = null;
    this.steadySince = null;
    this.lastFrameId = null;
    this.recent = [];
    this.sharpness = [];
  }

  /** Relative sharpness: at least 85% of the sharpest frame in the last 1.5 s (true when unknown). */
  private sharpNow(s: StabilitySample): boolean {
    if (s.sharpness === undefined) return true;
    this.sharpness.push({ t: s.t, v: s.sharpness });
    this.sharpness = this.sharpness.filter((x) => s.t - x.t <= 1500);
    const best = Math.max(...this.sharpness.map((x) => x.v));
    return s.sharpness >= 0.85 * best;
  }

  push(s: StabilitySample): StabilityState {
    // A repeated frame is not a new observation; too long without a new frame resets the dwell.
    if (this.lastFrameId !== null && s.frameId === this.lastFrameId) {
      if (s.t - this.lastNewFrameAt > this.maxGapMs) {
        this.reset();
        return { steady: false, steadyMs: 0, motion: null, sharp: false, reason: 'stale' };
      }
      return this.state(s.t, null, this.steadySince !== null ? 'steady' : 'motion', false);
    }
    const prev = this.prev;
    const sharp = this.sharpNow(s);
    this.lastFrameId = s.frameId;
    this.lastNewFrameAt = s.t;
    this.prev = s;
    if (!prev || prev.width !== s.width || prev.height !== s.height) {
      this.steadySince = null;
      return { steady: false, steadyMs: 0, motion: null, sharp, reason: 'warming-up' };
    }
    if (s.t - prev.t > this.maxGapMs) {
      this.steadySince = null;
      return { steady: false, steadyMs: 0, motion: null, sharp, reason: 'gap' };
    }
    let diff = 0, sum = 0;
    for (let i = 0; i < s.luma.length; i++) {
      diff += Math.abs(s.luma[i] - prev.luma[i]);
      sum += s.luma[i];
    }
    const raw = diff / s.luma.length;
    if (sum / s.luma.length < 12) {
      this.steadySince = null;
      return { steady: false, steadyMs: 0, motion: raw, sharp, reason: 'dark' };
    }
    // Hand tremor makes single frames spike (one shaky frame gives two large differences: away and
    // back). Judge the lower median of the last five, but let one large jolt (3x the limit) reset.
    this.recent.push(raw);
    if (this.recent.length > 5) this.recent.shift();
    const sorted = [...this.recent].sort((a, b) => a - b);
    const motion = sorted[(sorted.length - 1) >> 1];
    const limit = TOLERANCE[this.tolerance];
    if (motion > limit || raw > 3 * limit) {
      this.steadySince = null;
      return { steady: false, steadyMs: 0, motion, sharp, reason: 'motion' };
    }
    this.steadySince ??= prev.t;
    return this.state(s.t, motion, 'steady', sharp);
  }

  private state(t: number, motion: number | null, reason: StabilityState['reason'], sharp: boolean): StabilityState {
    const steadyMs = this.steadySince === null ? 0 : t - this.steadySince;
    return { steady: this.steadySince !== null, steadyMs, motion, sharp, reason };
  }
}

/** Downscale RGBA pixels to luma (Rec. 601) for the stability test. */
export function lumaFromRGBA(rgba: Uint8ClampedArray, width: number, height: number): Uint8Array {
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) out[i] = (rgba[i * 4] * 77 + rgba[i * 4 + 1] * 150 + rgba[i * 4 + 2] * 29) >> 8;
  return out;
}
