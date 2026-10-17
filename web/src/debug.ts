// Dev-only breadcrumbs to the dev server (/__log), so crashes on a phone leave a trail.
// An "alive" marker in localStorage (all builds) tells the next boot whether this page ended
// cleanly (pagehide) or was killed (e.g. the iOS tab killed for memory), with its last breadcrumbs.
const enabled = import.meta.env.DEV;
const started = performance.now();
const ALIVE = 'tcg-alive-v1';
const RING = 12;

interface AliveMarker {
  id: string;
  bootAt: number;
  /** Wall-clock time of the last breadcrumb. */
  lastAt: number;
  clean: boolean;
  /** How it ended, when it ended cleanly. */
  end?: string;
  crumbs: string[];
}

let marker: AliveMarker | null = null;
let lastWrite = 0;
let writeTimer: ReturnType<typeof setTimeout> | null = null;

function writeMarker(force = false) {
  if (!marker) return;
  const now = Date.now();
  // Throttled: localStorage writes are synchronous. Forced on lifecycle events.
  if (!force && now - lastWrite < 400) {
    writeTimer ??= setTimeout(() => { writeTimer = null; writeMarker(true); }, 400);
    return;
  }
  lastWrite = now;
  try { localStorage.setItem(ALIVE, JSON.stringify(marker)); } catch { /* storage unavailable */ }
}

function crumb(event: string, data: Record<string, unknown>) {
  if (!marker) return;
  const t = Math.round(performance.now() - started);
  let text = `${t} ${event}`;
  const extra = JSON.stringify(data);
  if (extra !== '{}') text += ' ' + extra.slice(0, 120);
  marker.crumbs.push(text);
  if (marker.crumbs.length > RING) marker.crumbs.shift();
  marker.lastAt = Date.now();
  writeMarker();
}

export function trace(event: string, data: Record<string, unknown> = {}): void {
  crumb(event, data);
  if (!enabled) return;
  const body = JSON.stringify({ t: Math.round(performance.now() - started), event, ...data });
  try {
    if (!navigator.sendBeacon?.('/__log', body)) void fetch('/__log', { method: 'POST', body, keepalive: true });
  } catch { /* logging must never break the app */ }
}

function markEnded(end: string) {
  if (!marker) return;
  marker.clean = true;
  marker.end = end;
  marker.lastAt = Date.now();
  writeMarker(true);
}

export function installCrashTrace(): void {
  // Previous page of this origin: did it end cleanly?
  let previous: AliveMarker | null = null;
  try { previous = JSON.parse(localStorage.getItem(ALIVE) ?? 'null') as AliveMarker | null; } catch { previous = null; }
  marker = { id: Math.random().toString(36).slice(2, 8), bootAt: Date.now(), lastAt: Date.now(), clean: false, crumbs: [] };
  const nav = performance.getEntriesByType?.('navigation')[0] as PerformanceNavigationTiming | undefined;
  const n = navigator as Navigator & { deviceMemory?: number };
  trace('boot', { ua: navigator.userAgent, deviceMemory: n.deviceMemory, cores: navigator.hardwareConcurrency,
    isolated: crossOriginIsolated, webgpu: 'gpu' in navigator, screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
    navigation: nav?.type ?? 'unknown', visible: !document.hidden });
  if (previous) {
    trace(previous.clean ? 'previous-session' : 'previous-session-unclean', {
      id: previous.id, end: previous.end ?? null, clean: previous.clean,
      ranS: Math.round((previous.lastAt - previous.bootAt) / 1000), sinceLastS: Math.round((Date.now() - previous.lastAt) / 1000),
      crumbs: previous.crumbs,
    });
  }
  writeMarker(true);
  addEventListener('error', (e) => trace('error', { message: e.message, at: `${e.filename}:${e.lineno}` }));
  addEventListener('unhandledrejection', (e) => trace('rejection', { message: String((e.reason as Error)?.message ?? e.reason) }));
  document.addEventListener('visibilitychange', () => { trace('visibility', { hidden: document.hidden }); if (document.hidden) writeMarker(true); });
  // pagehide: the page is unloading (reload, navigation) or entering the back/forward cache.
  window.addEventListener('pagehide', (e: PageTransitionEvent) => { trace('pagehide', { persisted: e.persisted }); markEnded(e.persisted ? 'bfcache' : 'pagehide'); });
  window.addEventListener('pageshow', (e: PageTransitionEvent) => {
    if (!e.persisted || !marker) return;
    marker.clean = false;
    delete marker.end;
    trace('pageshow', { persisted: true });
    writeMarker(true);
  });
  // Page Lifecycle (Chromium only; WebKit does not fire these).
  document.addEventListener('freeze', () => { trace('freeze'); writeMarker(true); });
  document.addEventListener('resume', () => trace('resume'));
  if (import.meta.hot) {
    // A dev-server full reload (a source file changed): a development artifact, not an app crash.
    import.meta.hot.on('vite:beforeFullReload', (p: { path?: string }) => { trace('hmr-full-reload', { path: p?.path ?? null }); markEnded('hmr-full-reload'); });
  }
}
