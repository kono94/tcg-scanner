// Per-viewer settings in localStorage (a convenience; the app works without storage).
import { DEFAULT_CYCLE, type TriggerMode } from '../capture/cycle';
import type { MotionTolerance } from '../capture/stability';

export interface Settings {
  preferredLanguage: string;
  mode: TriggerMode;
  dwellMs: number;
  cooldownMs: number;
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
  /** Dev server only: upload every full capture with its detections for training (runs/captures/). */
  saveCaptures: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  preferredLanguage: 'en', mode: DEFAULT_CYCLE.mode, dwellMs: DEFAULT_CYCLE.dwellMs, cooldownMs: DEFAULT_CYCLE.cooldownMs,
  tolerance: 'normal', theme: 'system', preferWebGPU: true, diagnostics: false, reviewMode: 'trust', priceMarket: 'EUR',
  workflow: 'page', saveCaptures: false,
};

const KEY = 'tcg-scanner-settings-v1';

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    const s = raw ? { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) } : { ...DEFAULT_SETTINGS };
    if (s.preferredLanguage !== 'en' && s.preferredLanguage !== 'de') s.preferredLanguage = 'en'; // only EN/DE supported
    if (!import.meta.env.DEV) s.saveCaptures = false; // never in production builds
    return s;
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s: Settings): void {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* storage unavailable */ }
}
