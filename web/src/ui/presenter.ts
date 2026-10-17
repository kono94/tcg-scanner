// Crop-flight choreography that follows real work and never delays a result.
//
// Rules (fast and slow pipelines use the same ones):
// 1. Card state, badges and counts update the moment a result arrives; flights are decoration.
// 2. The loupe only shows a card whose work is still running `graceMs` after it started.
//    Fast cards never enter the loupe; slow ones (e.g. a hard verification) do.
// 3. A card that finished without the loupe flies straight from the photo to its slot.
// 4. A flight starts within `startWindowMs` of its trigger or not at all, at most `maxFlights`
//    at once; anything else lands immediately. No animation backlog builds up.
// 5. A card that finishes while still flying to the loupe is redirected from where it is.
// 6. Reduced motion: no flights; crops appear in place.
import type { Point, Quad } from '../vision/image';

export interface PresenterOptions {
  graceMs: number;
  flightMs: number;
  quickFlightMs: number;
  maxFlights: number;
  startWindowMs: number;
  reducedMotion: boolean;
}

export const DEFAULT_PRESENTER: PresenterOptions = { graceMs: 200, flightMs: 420, quickFlightMs: 260, maxFlights: 4, startWindowMs: 150, reducedMotion: false };

export interface PresentedCard {
  id: string;
  working: boolean;
  startedAt: number | null;
  settledAt: number | null;
  hasCrop: boolean;
}

export interface Geometry {
  photo(id: string): Quad | null;
  loupe(): Quad;
  slot(id: string): Quad | null;
}

export interface Flight {
  id: string;
  to: 'loupe' | 'slot';
  from: Quad;
  startedAt: number;
  duration: number;
}

const lerp = (a: Point, b: Point, t: number): Point => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const easeOut = (t: number) => 1 - (1 - t) ** 3;
const easeInOut = (t: number) => (t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2);

export class Presenter {
  /** Card shown in the loupe, if any. */
  focus: string | null = null;
  /** True once the focused crop has arrived in the loupe. */
  focusLanded = false;
  readonly flights = new Map<string, Flight>();
  /** Cards whose crop is shown in their filmstrip slot. */
  readonly landed = new Set<string>();
  private readonly focused = new Set<string>();

  constructor(public options: PresenterOptions = { ...DEFAULT_PRESENTER }) {}

  reset(): void {
    this.focus = null;
    this.focusLanded = false;
    this.flights.clear();
    this.landed.clear();
    this.focused.clear();
  }

  get animating(): boolean {
    return this.flights.size > 0;
  }

  /** Corners of a flight at time `now`, in the same coordinates as the geometry. */
  quadAt(flight: Flight, now: number, geo: Geometry): Quad | null {
    const target = flight.to === 'loupe' ? geo.loupe() : geo.slot(flight.id);
    if (!target) return null;
    const raw = Math.min(1, Math.max(0, (now - flight.startedAt) / flight.duration));
    const t = flight.to === 'loupe' ? easeOut(raw) : easeInOut(raw);
    return [0, 1, 2, 3].map((i) => lerp(flight.from[i], target[i], t)) as unknown as Quad;
  }

  private start(id: string, to: Flight['to'], from: Quad | null, trigger: number, now: number): boolean {
    const o = this.options;
    if (o.reducedMotion || !from || now - trigger > o.startWindowMs || this.flights.size >= o.maxFlights) return false;
    const duration = this.flights.size > 0 ? o.quickFlightMs : o.flightMs;
    this.flights.set(id, { id, to, from, startedAt: now, duration });
    return true;
  }

  update(cards: readonly PresentedCard[], now: number, geo: Geometry): void {
    const ids = new Set(cards.map((c) => c.id));
    for (const id of [...this.flights.keys()]) if (!ids.has(id)) this.flights.delete(id);
    for (const id of [...this.landed]) if (!ids.has(id)) this.landed.delete(id);
    if (this.focus && !ids.has(this.focus)) { this.focus = null; this.focusLanded = false; }

    // Finish flights.
    for (const f of [...this.flights.values()]) {
      if (now - f.startedAt < f.duration) continue;
      this.flights.delete(f.id);
      if (f.to === 'loupe') this.focusLanded = this.focus === f.id;
      else this.landed.add(f.id);
    }

    for (const c of cards) {
      // Retried cards return to the queue: forget their earlier placement.
      if (!c.hasCrop && !c.working) {
        this.landed.delete(c.id);
        this.focused.delete(c.id);
        continue;
      }
      if (c.working && c.hasCrop && c.startedAt !== null && this.focus === null && !this.focused.has(c.id)) {
        const engageAt = c.startedAt + this.options.graceMs;
        if (now >= engageAt) {
          this.focus = c.id;
          this.focused.add(c.id);
          this.focusLanded = !this.start(c.id, 'loupe', geo.photo(c.id), now, now);
        }
      }
      if (!c.working && c.settledAt !== null && !this.landed.has(c.id) && !this.flights.has(c.id) && c.hasCrop) {
        if (this.focus === c.id) {
          this.focus = null;
          this.focusLanded = false;
          if (!this.start(c.id, 'slot', geo.loupe(), now, now)) this.landed.add(c.id);
        } else if (this.focused.has(c.id)) {
          this.landed.add(c.id);
        } else if (!this.start(c.id, 'slot', geo.photo(c.id), c.settledAt, now)) {
          this.landed.add(c.id);
        }
      }
      // Redirect: finished while still flying into the loupe.
      const f = this.flights.get(c.id);
      if (f && f.to === 'loupe' && !c.working) {
        const from = this.quadAt(f, now, geo) ?? geo.loupe();
        this.flights.delete(c.id);
        this.focus = null;
        this.focusLanded = false;
        this.flights.set(c.id, { id: c.id, to: 'slot', from, startedAt: now, duration: this.options.quickFlightMs });
      }
    }
  }
}
