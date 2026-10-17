// Stable-frame capture cycle (docs/CAPTURE_CYCLE.md): at most one active batch, no queued
// timer captures, monotonic deadlines. Pure logic driven by timestamps for testability.
import type { StabilityState } from './stability';

export type TriggerMode = 'auto' | 'manual' | 'repeat';

export interface CycleSettings {
  mode: TriggerMode;
  /** Continuous steady time before an automatic capture. */
  dwellMs: number;
  /** Repeat mode: minimum time after the previous batch settles. */
  cooldownMs: number;
}

export const DEFAULT_CYCLE: CycleSettings = { mode: 'auto', dwellMs: 800, cooldownMs: 3000 };

export type CyclePhase = 'live' | 'processing' | 'review' | 'paused';

export interface CycleView {
  phase: CyclePhase;
  /** Remaining steady time before an armed capture; null when no capture is armed or the scene moves. */
  countdownMs: number | null;
  /** True when automatic capture is armed (eligible once steady). */
  armed: boolean;
}

export class CaptureCycle {
  private phase: CyclePhase = 'live';
  private beforePause: CyclePhase = 'live';
  private settledAt: number | null = null;
  private frozen = false;

  constructor(public settings: CycleSettings = { ...DEFAULT_CYCLE }) {}

  get current(): CyclePhase {
    return this.phase;
  }

  /** Evaluate one stability sample. Returns whether to capture now; never queues a later capture. */
  tick(t: number, s: StabilityState): { capture: boolean; view: CycleView } {
    const { mode, dwellMs, cooldownMs } = this.settings;
    const idle = { capture: false, view: { phase: this.phase, countdownMs: null, armed: false } };
    if (this.phase === 'live') {
      if (mode === 'manual') return idle;
      const remaining = s.steady ? Math.max(0, dwellMs - s.steadyMs) : null;
      // Steady long enough is not enough: also wait for a sharp frame (no motion/focus blur).
      return { capture: remaining === 0 && s.sharp !== false, view: { phase: 'live', countdownMs: remaining, armed: true } };
    }
    if (this.phase === 'review' && mode === 'repeat' && !this.frozen && this.settledAt !== null) {
      const cooldown = Math.max(0, cooldownMs - (t - this.settledAt));
      const remaining = s.steady ? Math.max(cooldown, dwellMs - s.steadyMs) : null;
      return { capture: remaining === 0 && s.sharp !== false, view: { phase: 'review', countdownMs: remaining, armed: true } };
    }
    return idle;
  }

  /** Manual capture bypasses the dwell, not later quality checks; refused while a batch runs. */
  canCaptureManually(): boolean {
    return this.phase === 'live' || this.phase === 'review';
  }

  beginBatch(): void {
    if (this.phase === 'processing') throw new Error('A batch is already active');
    this.phase = 'processing';
    this.settledAt = null;
  }

  settle(t: number): void {
    if (this.phase !== 'processing') return;
    this.phase = 'review';
    this.settledAt = t;
  }

  /** Detail review/editing freezes automatic replacement captures. */
  setFrozen(frozen: boolean): void {
    this.frozen = frozen;
  }

  /** Return to live acquisition (scan again or next page). */
  rearm(): void {
    if (this.phase === 'paused') this.beforePause = 'live';
    else this.phase = 'live';
    this.settledAt = null;
  }

  pause(): void {
    if (this.phase === 'paused') return;
    this.beforePause = this.phase === 'processing' ? 'live' : this.phase;
    this.phase = 'paused';
  }

  /** Resume requires fresh stability samples; the caller resets its monitor. */
  resume(t: number): void {
    if (this.phase !== 'paused') return;
    this.phase = this.beforePause;
    if (this.phase === 'review') this.settledAt = t;
  }
}
