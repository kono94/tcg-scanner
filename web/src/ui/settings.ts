// Per-viewer settings in localStorage (a convenience; the app works without storage).
import { DEFAULT_CYCLE, type TriggerMode } from '../capture/cycle';
import type { MotionTolerance } from '../capture/stability';

export interface Settings {
  preferredLanguage: string;
  mode: TriggerMode;
  dwellMs: number;
  cooldownMs: number;
  /** Batch: page-turn window after each capture; the steady dwell starts when it ends. */
  batchCooldownMs: number;
  tolerance: MotionTolerance;
  theme: 'system' | 'dark' | 'light';
  preferWebGPU: boolean;
  diagnostics: boolean;
  /** 'review': step through every card after a scan and confirm or correct it. */
  reviewMode: 'trust' | 'review';
  /** Price market: Cardmarket (EUR) or TCGplayer (USD). */
  priceMarket: 'EUR' | 'USD';
  /** 'page': review each page before the next; 'batch': scan page after page, check in the background, review at the end. */
  workflow: 'page' | 'batch';
  /** Dev server only: upload every full capture with its detections for training (datasets/sources/phone/captures/). */
  saveCaptures: boolean;
  /** Capture confirmation besides the flash: a short vibration (a click sound where vibration is unsupported). */
  captureFeedback: 'vibrate' | 'off';
  /** Settings revision, for one-time default migrations. */
  rev?: number;
}

export const DEFAULT_SETTINGS: Settings = {
  preferredLanguage: 'en', mode: DEFAULT_CYCLE.mode, dwellMs: DEFAULT_CYCLE.dwellMs, cooldownMs: DEFAULT_CYCLE.cooldownMs,
  tolerance: 'normal', theme: 'system', preferWebGPU: true, diagnostics: false, reviewMode: 'trust', priceMarket: 'EUR',
  workflow: 'page', saveCaptures: false, batchCooldownMs: 5000, captureFeedback: 'vibrate', rev: 3,
};

const KEY = 'tcg-scanner-settings-v1';

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    const s = raw ? { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) } : { ...DEFAULT_SETTINGS };
    if (s.preferredLanguage !== 'en' && s.preferredLanguage !== 'de') s.preferredLanguage = 'en'; // only EN/DE supported
    if (!import.meta.env.DEV) s.saveCaptures = false; // never in production builds
    // rev 2: the batch pause no longer doubles as a page-turn guard (the scene-change gate is), so 3 s became 1.5 s.
    if ((s.rev ?? 1) < 2) { if (s.batchCooldownMs === 3000) s.batchCooldownMs = 1500; s.rev = 2; }
    // rev 3: the pause became an explicit page-turn window; 1.5 s left too little time to turn a page.
    if ((s.rev ?? 1) < 3) { if (s.batchCooldownMs < 5000) s.batchCooldownMs = DEFAULT_SETTINGS.batchCooldownMs; s.rev = 3; }
    return s;
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s: Settings): void {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* storage unavailable */ }
}
