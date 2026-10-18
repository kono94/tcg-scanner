import { describe, expect, it, vi } from 'vitest';
import { CameraLifecycle, needsCameraFrames, type CameraUse } from '../src/camera/lifecycle';
import type { CameraHandle } from '../src/camera/source';

const camera = () => ({ stream: {} as MediaStream, label: 'camera', mock: false, stop: vi.fn(), setActive: vi.fn() });
const deferred = () => {
  let resolve!: (v: CameraHandle) => void;
  const promise = new Promise<CameraHandle>((r) => { resolve = r; });
  return { promise, resolve };
};

describe('camera ownership', () => {
  it('coalesces requests and reuses a suspended stream without another camera request', async () => {
    const c = new CameraLifecycle(), pending = deferred(), handle = camera();
    const factory = vi.fn(() => pending.promise);
    const a = c.open(factory), b = c.open(factory);
    expect(a).toBe(b);
    expect(factory).toHaveBeenCalledTimes(1);
    pending.resolve(handle);
    await a;
    c.setActive(true);
    c.setActive(false);
    c.setActive(true);
    expect(c.current).toBe(handle);
    expect(factory).toHaveBeenCalledTimes(1);
    expect(handle.stop).not.toHaveBeenCalled();
  });

  it('stops a late permission result after navigation and cannot replace a newer stream', async () => {
    const c = new CameraLifecycle(), old = deferred(), stale = camera(), fresh = camera();
    const a = c.open(() => old.promise);
    c.cancelOpen();
    await c.open(async () => fresh);
    old.resolve(stale);
    expect(await a).toBeNull();
    expect(stale.stop).toHaveBeenCalledOnce();
    expect(c.current).toBe(fresh);
    c.release();
    expect(fresh.stop).toHaveBeenCalledOnce();
    expect(c.current).toBeNull();
  });

  it('does not keep retrying after permission denial', async () => {
    const c = new CameraLifecycle();
    const factory = vi.fn(async () => { throw new Error('denied'); });
    await expect(c.open(factory)).rejects.toThrow('denied');
    expect(factory).toHaveBeenCalledOnce();
    expect(c.current).toBeNull();
  });

  it('requires a visible acquiring scanner and suspends settings, review, processing and hidden states', () => {
    const live: CameraUse = { hidden: false, scannerVisible: true, selected: false, phase: 'live', mode: 'auto' };
    expect(needsCameraFrames(live)).toBe(true);
    for (const patch of [{ hidden: true }, { scannerVisible: false }, { selected: true }, { phase: 'processing' }, { phase: 'review' }, { phase: 'paused' }] as Partial<CameraUse>[]) {
      expect(needsCameraFrames({ ...live, ...patch })).toBe(false);
    }
    expect(needsCameraFrames({ ...live, phase: 'review', mode: 'repeat' })).toBe(true);
    expect(needsCameraFrames({ ...live, phase: 'review', mode: 'repeat', scannerVisible: false })).toBe(false);
  });
});
