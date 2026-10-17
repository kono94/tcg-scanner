import { describe, expect, it } from 'vitest';
import { CaptureCycle } from '../src/capture/cycle';
import { StabilityMonitor, type StabilitySample } from '../src/capture/stability';

const frame = (t: number, id: number, value = 120): StabilitySample => ({ t, frameId: id, width: 4, height: 4, luma: new Uint8Array(16).fill(value) });

describe('stability monitor', () => {
  it('accumulates steady time and resets on motion', () => {
    const m = new StabilityMonitor();
    m.push(frame(0, 1));
    expect(m.push(frame(100, 2)).steadyMs).toBe(100);
    expect(m.push(frame(200, 3)).steadyMs).toBe(200);
    expect(m.push(frame(300, 4, 160)).reason).toBe('motion');
    expect(m.push(frame(400, 5, 160)).steadyMs).toBe(100);
  });

  it('does not count a repeated frame as a new observation and resets on stale frames', () => {
    const m = new StabilityMonitor('normal', 400);
    m.push(frame(0, 1));
    m.push(frame(100, 2));
    expect(m.push(frame(200, 2)).steady).toBe(true);
    expect(m.push(frame(600, 2)).reason).toBe('stale');
    expect(m.push(frame(700, 3)).steady).toBe(false);
  });

  it('resets on a sampling gap', () => {
    const m = new StabilityMonitor('normal', 400);
    m.push(frame(0, 1));
    expect(m.push(frame(1000, 2)).reason).toBe('gap');
  });
});

describe('capture cycle', () => {
  const steady = (ms: number) => ({ steady: true, steadyMs: ms, motion: 0, sharp: true, reason: 'steady' as const });
  const moving = { steady: false, steadyMs: 0, motion: 9, sharp: true, reason: 'motion' as const };

  it('counts down only while steady and captures once at the dwell', () => {
    const c = new CaptureCycle();
    expect(c.tick(0, steady(300)).view.countdownMs).toBe(500);
    expect(c.tick(10, moving).view.countdownMs).toBeNull();
    expect(c.tick(900, steady(800)).capture).toBe(true);
    c.beginBatch();
    // No countdown or second capture while processing, even if the dwell is long expired.
    const busy = c.tick(2000, steady(5000));
    expect(busy.capture).toBe(false);
    expect(busy.view.countdownMs).toBeNull();
    expect(() => c.beginBatch()).toThrow();
  });

  it('stops for review after an auto capture', () => {
    const c = new CaptureCycle();
    c.beginBatch();
    c.settle(1000);
    expect(c.tick(10000, steady(9000)).capture).toBe(false);
    c.rearm();
    expect(c.tick(10001, steady(900)).capture).toBe(true);
  });

  it('repeat mode waits for both cooldown and dwell after settling, and freezes during detail review', () => {
    const c = new CaptureCycle({ mode: 'repeat', dwellMs: 800, cooldownMs: 3000 });
    c.beginBatch();
    c.settle(1000);
    expect(c.tick(2000, steady(2000)).view.countdownMs).toBe(2000);
    c.setFrozen(true);
    expect(c.tick(4500, steady(4000)).capture).toBe(false);
    c.setFrozen(false);
    expect(c.tick(4500, steady(4000)).capture).toBe(true);
  });

  it('manual mode never captures automatically', () => {
    const c = new CaptureCycle({ mode: 'manual', dwellMs: 800, cooldownMs: 3000 });
    expect(c.tick(5000, steady(5000)).capture).toBe(false);
    expect(c.canCaptureManually()).toBe(true);
  });

  it('pause disarms; resume returns to live', () => {
    const c = new CaptureCycle();
    c.pause();
    expect(c.tick(100, steady(5000)).capture).toBe(false);
    c.resume(200);
    expect(c.current).toBe('live');
  });
});

describe('hand tremor', () => {
  const textured = (t: number, id: number, shift: number) => ({ t, frameId: id, width: 4, height: 4,
    luma: Uint8Array.from({ length: 16 }, (_, i) => 120 + ((i + shift) % 4) * 4) });

  it('one spike does not reset the dwell, sustained motion does', () => {
    const m = new StabilityMonitor('normal');
    const lumaAt = (t: number, id: number, v: number) => ({ t, frameId: id, width: 4, height: 4, luma: new Uint8Array(16).fill(v) });
    m.push(lumaAt(0, 1, 120));
    m.push(lumaAt(100, 2, 121));
    m.push(lumaAt(200, 3, 120));
    expect(m.push(lumaAt(300, 4, 132)).steady).toBe(true); // single spike of 12 (> 9, < 27)
    expect(m.push(lumaAt(400, 5, 120)).steadyMs).toBe(400);
    m.push(lumaAt(500, 6, 140));
    m.push(lumaAt(600, 7, 120));
    expect(m.push(lumaAt(700, 8, 140)).reason).toBe('motion'); // three large changes in five samples
  });

  it('a big jolt resets at once', () => {
    const m = new StabilityMonitor('normal');
    m.push(textured(0, 1, 0));
    m.push(textured(100, 2, 0));
    expect(m.push({ t: 200, frameId: 3, width: 4, height: 4, luma: new Uint8Array(16).fill(200) }).reason).toBe('motion');
  });
});

describe('blur check', () => {
  it('waits for a sharp frame before capturing', async () => {
    const { sharpnessOf } = await import('../src/capture/stability');
    const W = 32, H = 24;
    const crisp = Uint8Array.from({ length: W * H }, (_, i) => ((i % W) >> 2) % 2 ? 200 : 40);
    const blurred = Uint8Array.from({ length: W * H }, (_, i) => 120 + Math.round(40 * Math.sin((i % W) / 3)));
    expect(sharpnessOf(crisp, W, H)).toBeGreaterThan(3 * sharpnessOf(blurred, W, H));
    const c = new CaptureCycle();
    expect(c.tick(1000, { steady: true, steadyMs: 900, motion: 1, sharp: false, reason: 'steady' }).capture).toBe(false);
    expect(c.tick(1100, { steady: true, steadyMs: 1000, motion: 1, sharp: true, reason: 'steady' }).capture).toBe(true);
  });

  it('judges sharpness against the recent best frame', () => {
    const m = new StabilityMonitor();
    const f = (t: number, id: number, sharpness: number) => ({ t, frameId: id, width: 4, height: 4, luma: new Uint8Array(16).fill(120), sharpness });
    m.push(f(0, 1, 10));
    expect(m.push(f(100, 2, 10)).sharp).toBe(true);
    expect(m.push(f(200, 3, 6)).sharp).toBe(false);
    expect(m.push(f(300, 4, 9.5)).sharp).toBe(true);
  });
});
