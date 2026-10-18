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
  /** Batch mode: the last page was captured; waiting for the scene to change (page turned). */
  awaitingChange?: boolean;
  /** Batch mode: the page-turn window after a capture is still running (not a steadiness countdown). */
  coolingDown?: boolean;
  /** Remaining page-turn window while coolingDown. */
  cooldownLeftMs?: number;
}

export class CaptureCycle {
  private phase: CyclePhase = 'live';
  private beforePause: CyclePhase = 'live';
  private settledAt: number | null = null;
  private frozen = false;
  private awaitingChange = false;
  private liveNotBefore = 0;

  constructor(public settings: CycleSettings = { ...DEFAULT_CYCLE }) {}

  get current(): CyclePhase {
    return this.phase;
  }

  /**
   * Evaluate one stability sample. Returns whether to capture now; never queues a later capture.
   * `sceneChanged`: whether the scene differs from the last capture (see SceneChangeGate); only
   * consulted after requireSceneChange().
   */
  tick(t: number, s: StabilityState, sceneChanged = true): { capture: boolean; view: CycleView } {
    const { mode, dwellMs, cooldownMs } = this.settings;
    const idle = { capture: false, view: { phase: this.phase, countdownMs: null, armed: false } };
    if (this.frozen) return idle;
    if (this.phase === 'live') {
      if (mode === 'manual') return idle;
      // The page-turn window is not shown as "hold steady": the user is turning the page.
      if (t < this.liveNotBefore) return { capture: false, view: { phase: 'live', countdownMs: null, armed: false, coolingDown: true, cooldownLeftMs: this.liveNotBefore - t } };
      if (this.awaitingChange && !sceneChanged) return { capture: false, view: { phase: 'live', countdownMs: null, armed: false, awaitingChange: true } };
      // The steady dwell starts when the window ends: steadiness while turning does not count.
      const steadyMs = this.liveNotBefore ? Math.min(s.steadyMs, t - this.liveNotBefore) : s.steadyMs;
      const remaining = s.steady ? Math.max(0, dwellMs - steadyMs) : null;
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
    return !this.frozen && (this.phase === 'live' || this.phase === 'review');
  }

  beginBatch(): void {
    if (this.phase === 'processing') throw new Error('A batch is already active');
    this.phase = 'processing';
    this.awaitingChange = false;
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

  /** Batch mode: no automatic capture until the scene has changed (the next page). Manual capture stays available. */
  requireSceneChange(): void {
    this.awaitingChange = true;
  }

  /** Return to live acquisition (scan again, next page, or an explicit start); clears the page-turn gate. */
  rearm(notBefore = 0): void {
    if (this.phase === 'paused') this.beforePause = 'live';
    else this.phase = 'live';
    this.settledAt = null;
    this.awaitingChange = false;
    this.liveNotBefore = notBefore;
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
