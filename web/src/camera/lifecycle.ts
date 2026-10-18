import type { CameraHandle } from './source';
import type { CyclePhase, TriggerMode } from '../capture/cycle';

export interface CameraUse {
  hidden: boolean;
  scannerVisible: boolean;
  selected: boolean;
  phase: CyclePhase;
  mode: TriggerMode;
}

/** Camera pixels are useful only for an eligible foreground acquisition. */
export function needsCameraFrames(s: CameraUse): boolean {
  return !s.hidden && s.scannerVisible && !s.selected &&
    (s.phase === 'live' || (s.phase === 'review' && s.mode === 'repeat'));
}

/** One owned stream, coalesced opens, and no resurrection after navigation/hide. */
export class CameraLifecycle {
  current: CameraHandle | null = null;
  active = false;
  private generation = 0;
  private pending: { factory: () => Promise<CameraHandle>; promise: Promise<CameraHandle | null> } | null = null;

  open(factory: () => Promise<CameraHandle>): Promise<CameraHandle | null> {
    if (this.pending?.factory === factory) return this.pending.promise;
    this.release();
    const generation = this.generation;
    const promise = factory().then((camera) => {
      if (generation !== this.generation) { camera.stop(); return null; }
      this.current = camera;
      // Opening must not deliver frames behind a screen before the controller admits them.
      camera.setActive(false);
      return camera;
    }).finally(() => { if (generation === this.generation) this.pending = null; });
    this.pending = { factory, promise };
    return promise;
  }

  cancelOpen(): void {
    if (!this.pending) return;
    this.generation++;
    this.pending = null;
  }

  setActive(active: boolean): boolean {
    if (!this.current || this.active === active) return false;
    this.current.setActive(active);
    this.active = active;
    return true;
  }

  release(): void {
    this.generation++;
    this.pending = null;
    this.current?.stop();
    this.current = null;
    this.active = false;
  }
}
