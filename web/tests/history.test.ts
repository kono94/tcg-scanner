import { describe, expect, it } from 'vitest';
import { isWithin, parentRoute, RouteHistory, type HistoryLike } from '../src/ui/history';

/** Browser session history: go() lands later, like the real popstate. */
class FakeHistory implements HistoryLike {
  entries: unknown[] = [null];
  index = 0;
  private pending: number[] = [];
  onPop: (state: unknown) => void = () => {};
  get state() { return this.entries[this.index]; }
  pushState(data: unknown) { this.entries = this.entries.slice(0, this.index + 1); this.entries.push(data); this.index++; }
  replaceState(data: unknown) { this.entries[this.index] = data; }
  go(delta: number) { this.pending.push(delta); }
  /** The user's Back/Forward button. */
  press(delta: number) { this.go(delta); this.flush(); }
  flush() {
    while (this.pending.length) {
      this.index = Math.max(0, Math.min(this.entries.length - 1, this.index + this.pending.shift()!));
      this.onPop(this.state);
    }
  }
  routes() { return this.entries.map((e) => (e as { path?: string[] } | null)?.path?.at(-1) ?? null); }
}

/** An app with a route and the parent-only "leave screen" step, like App.goTo. */
function setup(history = new FakeHistory()) {
  const app = { route: 'home' };
  const routes = new RouteHistory(history, () => app.route, (target) => {
    while (!isWithin(app.route, target)) app.route = parentRoute(app.route)!;
    if (app.route === 'home' && target !== 'home') app.route = target.split('/')[0];
    routes.sync();
  });
  history.onPop = (state) => routes.onPopState(state);
  const show = (route: string) => { app.route = route; routes.sync(); history.flush(); };
  return { app, history, routes, show };
}

describe('route helpers', () => {
  it('nests every route below home', () => {
    expect(parentRoute('collection/page/card')).toBe('collection/page');
    expect(parentRoute('collection')).toBe('home');
    expect(parentRoute('home')).toBeNull();
    expect(isWithin('home', 'scan/card')).toBe(true);
    expect(isWithin('scan', 'scan/card')).toBe(true);
    expect(isWithin('scan/card', 'scan')).toBe(false);
    expect(isWithin('collection', 'collection')).toBe(true);
    expect(isWithin('collection', 'collectionx')).toBe(false);
  });
});

describe('RouteHistory', () => {
  it('pushes deeper screens and Back steps out of them one at a time', () => {
    const { app, history, show } = setup();
    show('collection');
    show('collection/page');
    show('collection/page/card');
    expect(history.routes()).toEqual(['home', 'collection', 'collection/page', 'collection/page/card']);
    history.press(-1);
    expect(app.route).toBe('collection/page');
    history.press(-1);
    history.press(-1);
    expect(app.route).toBe('home');
    expect(history.index).toBe(0);
  });

  it('closing a screen in the app goes back in history, leaving no dead Back step', () => {
    const { history, show } = setup();
    show('collection');
    show('collection/page');
    show('collection');
    expect(history.index).toBe(1);
    show('home');
    expect(history.index).toBe(0);
  });

  it('switching tabs goes through Home, so Back from any tab returns Home', () => {
    const { app, history, show } = setup();
    show('collection');
    show('collection/page');
    show('settings');
    expect(history.routes().slice(0, history.index + 1)).toEqual(['home', 'settings']);
    history.press(-1);
    expect(app.route).toBe('home');
  });

  it('Forward reopens a tab but not a closed card', () => {
    const { app, history, show } = setup();
    show('scan');
    show('scan/card');
    history.press(-1);
    history.press(-1);
    expect(app.route).toBe('home');
    history.press(1);
    expect(app.route).toBe('scan');
    history.press(1);
    expect(app.route).toBe('scan'); // the card cannot be rebuilt: history follows the app
    expect(history.routes()[history.index]).toBe('scan');
  });

  it('screens shown while a traversal is in flight are recorded when it lands', () => {
    const { app, history, routes } = setup();
    for (const r of ['collection', 'collection/page']) { app.route = r; routes.sync(); }
    app.route = 'collection'; routes.sync(); // go(-1) pending
    app.route = 'collection/card'; routes.sync(); // ignored until popstate
    history.flush();
    expect(history.routes().slice(0, history.index + 1)).toEqual(['home', 'collection', 'collection/card']);
  });

  it('after a reload walks back to the screen the app opens on', () => {
    const history = new FakeHistory();
    setup(history).show('collection');
    setup(history).show('collection/page');
    expect(history.index).toBe(2);
    const { app, show } = setup(history); // reload: the app starts on Home
    show('home');
    expect(app.route).toBe('home');
    expect(history.index).toBe(0);
  });
});
