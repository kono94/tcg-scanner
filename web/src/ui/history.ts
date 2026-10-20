// Browser history for in-app screens, so Back/Forward (and Android's Back gesture) move between
// screens instead of leaving the app. Routes are slash paths below 'home' ('collection/page/card');
// the history mirrors the path from home to the screen shown. Opening a deeper screen pushes an
// entry; returning to an ancestor goes back to its entry; switching to another branch first goes
// back to the shared ancestor, then pushes. The URL never changes.

export interface HistoryLike {
  readonly state: unknown;
  pushState(data: unknown, unused: string): void;
  replaceState(data: unknown, unused: string): void;
  go(delta: number): void;
}

interface Entry {
  tcgRoute: true;
  /** Routes from the first entry up to and including this one. */
  path: string[];
}

const ROOT = 'home';

export function parentRoute(route: string): string | null {
  const i = route.lastIndexOf('/');
  return i >= 0 ? route.slice(0, i) : route === ROOT ? null : ROOT;
}

/** a is b or one of its ancestors. */
export function isWithin(a: string, b: string): boolean {
  for (let r: string | null = b; r !== null; r = parentRoute(r)) if (r === a) return true;
  return false;
}

function asEntry(state: unknown): Entry | null {
  const e = state as Partial<Entry> | null;
  return e?.tcgRoute === true && Array.isArray(e.path) && e.path.length > 0 && e.path.every((r) => typeof r === 'string') ? e as Entry : null;
}

export class RouteHistory {
  private path: string[];
  /** A go() is in flight: its popstate settles the path, then the shown screen is recorded. */
  private traversing = false;

  /**
   * @param current screen shown now (read again after every navigation)
   * @param navigate move the app toward a route the user went Back/Forward to (best effort)
   */
  constructor(private readonly history: HistoryLike, private readonly current: () => string, private readonly navigate: (route: string) => void) {
    // After a reload the entry keeps the path of the earlier visit; sync() walks back from it.
    this.path = asEntry(history.state)?.path ?? [];
    if (!this.path.length) {
      this.path = [current()];
      this.write('replace');
    }
  }

  /** Record the screen now shown (call after every render). */
  sync(): void {
    if (this.traversing) return;
    const route = this.current();
    const top = this.path.length - 1;
    if (route === this.path[top]) return;
    // Deepest entry on the way to this screen: push from it, after going back to it if needed.
    let base = top;
    while (base >= 0 && !isWithin(this.path[base], route)) base--;
    if (base === top) {
      this.path.push(route);
      this.write('push');
    } else if (base >= 0) {
      this.traversing = true;
      this.path = this.path.slice(0, base + 1);
      this.history.go(base - top);
    } else {
      this.path[top] = route;
      this.write('replace');
    }
  }

  /** Handle window 'popstate'. */
  onPopState(state: unknown): void {
    const entry = asEntry(state);
    const wasTraversing = this.traversing;
    this.traversing = false;
    if (!entry) {
      // An entry from before this app wrote any (e.g. a fragment link): adopt it.
      this.path = [this.current()];
      this.write('replace');
      return;
    }
    this.path = entry.path.slice();
    const target = this.path[this.path.length - 1];
    if (!wasTraversing && target !== this.current()) this.navigate(target);
    // The app may not reach the target (Forward into a closed card, a refused step): record where it is.
    this.sync();
  }

  private write(kind: 'push' | 'replace') {
    const entry: Entry = { tcgRoute: true, path: this.path.slice() };
    if (kind === 'push') this.history.pushState(entry, '');
    else this.history.replaceState(entry, '');
  }
}
