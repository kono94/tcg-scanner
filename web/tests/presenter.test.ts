import { describe, expect, it } from 'vitest';
import { DEFAULT_PRESENTER, Presenter, type Geometry, type PresentedCard } from '../src/ui/presenter';
import type { Quad } from '../src/vision/image';

const rect = (x: number, y: number): Quad => [[x, y], [x + 10, y], [x + 10, y + 14], [x, y + 14]];
const geo: Geometry = { photo: () => rect(0, 0), loupe: () => rect(100, 100), slot: () => rect(200, 200) };

/** Simulate a sequential worker: card i runs for durations[i] ms after the previous one. */
function simulate(durations: number[], options = DEFAULT_PRESENTER, frameMs = 16) {
  const p = new Presenter({ ...options });
  const starts: number[] = [];
  let t = 0;
  for (const d of durations) { starts.push(t); t += d; }
  const end = t + 1000;
  let maxFlights = 0, focused = new Set<string>();
  for (let now = 0; now <= end; now += frameMs) {
    const cards: PresentedCard[] = durations.map((d, i) => {
      const s = starts[i], e = s + d;
      return { id: String(i), working: now >= s && now < e, startedAt: now >= s ? s : null, settledAt: now >= e ? e : null, hasCrop: now >= s + 5 };
    });
    p.update(cards, now, geo);
    maxFlights = Math.max(maxFlights, p.flights.size);
    if (p.focus) focused.add(p.focus);
  }
  return { p, maxFlights, focused };
}

describe('presenter', () => {
  it('fast pipeline: no loupe, bounded direct flights, every card ends in its slot', () => {
    const { p, maxFlights, focused } = simulate(Array(9).fill(90));
    expect(focused.size).toBe(0);
    expect(maxFlights).toBeLessThanOrEqual(DEFAULT_PRESENTER.maxFlights);
    expect(p.landed.size).toBe(9);
    expect(p.animating).toBe(false);
  });

  it('whole page at once: flights capped, the rest land without queuing', () => {
    const { p, maxFlights } = simulate(Array(14).fill(5), DEFAULT_PRESENTER, 33);
    expect(maxFlights).toBeLessThanOrEqual(4);
    expect(p.landed.size).toBe(14);
  });

  it('mixed pipeline: only the slow card takes the loupe', () => {
    const { focused, p } = simulate([90, 90, 90, 90, 1100, 90, 90, 90, 90]);
    expect([...focused]).toEqual(['4']);
    expect(p.landed.size).toBe(9);
  });

  it('slow pipeline: every card takes the loupe', () => {
    const { focused } = simulate(Array(9).fill(800));
    expect(focused.size).toBe(9);
  });

  it('reduced motion: no flights at all', () => {
    const { maxFlights, p, focused } = simulate([90, 1100, 90], { ...DEFAULT_PRESENTER, reducedMotion: true });
    expect(maxFlights).toBe(0);
    expect(focused.size).toBe(1);
    expect(p.landed.size).toBe(3);
  });

  it('redirects a card that finishes while flying into the loupe', () => {
    const p = new Presenter();
    const card = (working: boolean, settledAt: number | null): PresentedCard[] => [{ id: 'a', working, startedAt: 0, settledAt, hasCrop: true }];
    p.update(card(true, null), 200, geo);
    expect(p.flights.get('a')?.to).toBe('loupe');
    p.update(card(false, 300), 300, geo);
    const f = p.flights.get('a')!;
    expect(f.to).toBe('slot');
    expect(f.from[0][0]).toBeGreaterThan(0);
    expect(f.from[0][0]).toBeLessThan(100);
  });
});
