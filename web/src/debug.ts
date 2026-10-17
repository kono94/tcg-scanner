// Dev-only breadcrumbs to the dev server (/__log), so crashes on a phone leave a trail.
const enabled = import.meta.env.DEV;
const started = performance.now();

export function trace(event: string, data: Record<string, unknown> = {}): void {
  if (!enabled) return;
  const body = JSON.stringify({ t: Math.round(performance.now() - started), event, ...data });
  try {
    if (!navigator.sendBeacon?.('/__log', body)) void fetch('/__log', { method: 'POST', body, keepalive: true });
  } catch { /* logging must never break the app */ }
}

export function installCrashTrace(): void {
  if (!enabled) return;
  const nav = navigator as Navigator & { deviceMemory?: number };
  trace('boot', { ua: navigator.userAgent, deviceMemory: nav.deviceMemory, cores: navigator.hardwareConcurrency,
    isolated: crossOriginIsolated, webgpu: 'gpu' in navigator, screen: `${screen.width}x${screen.height}@${devicePixelRatio}` });
  addEventListener('error', (e) => trace('error', { message: e.message, at: `${e.filename}:${e.lineno}` }));
  addEventListener('unhandledrejection', (e) => trace('rejection', { message: String((e.reason as Error)?.message ?? e.reason) }));
  document.addEventListener('visibilitychange', () => trace('visibility', { hidden: document.hidden }));
}
