// Saved pages for the Collection tab (per-viewer, in localStorage; a pilot store, not a sync service).
// Saving the same page generation again replaces it: quantities never double.
import { Catalog } from '../recognition/catalog';
import { displayCard, type MatchResult } from '../recognition/policy';
import type { PriceService } from '../prices/service';
import { deletePageCrops, encodeBitmap, putCrops } from './crops';
import type { PageState } from '../session/page';

export interface SavedValue {
  currency: string;
  preferred: number | null;
  low: number | null;
  high: number | null;
  /** Provisional: pricier variant possible or a version unpriced. */
  cue: boolean;
  /** The preferred price is converted from the other currency. */
  converted?: boolean;
}

export interface SavedCard {
  number: number;
  identity: string | null;
  name: string | null;
  cardId: string | null;
  state: string;
  languageAssumed: boolean;
  checkedByUser: boolean;
  /** Plausible catalog versions and the preferred one at scan time (absent on pages saved before 2026-10-17). */
  versions?: string[];
  preferredId?: string | null;
  /** A pricier variant was flagged at scan time (language not read). */
  checkVariant?: boolean;
  /** Value at scan time, in the market chosen then; null amounts are unknown (never zero). */
  value: SavedValue | null;
  /** Value from the current bundled snapshot in the current market; computed on load, never stored. */
  current?: SavedValue | null;
  /** Small JPEG data URL of the straightened crop. */
  thumb: string | null;
  /** Full crops are in IndexedDB (collection/crops.ts), so the card can be checked again. */
  hasCrop?: boolean;
  /** The user picked the match (or "not in the catalog") from the collection. */
  correctedAt?: string;
}

export interface SavedPage {
  key: string;
  sessionId: string;
  pageNumber: number;
  revision: number;
  savedAt: string;
  partial: boolean;
  cards: SavedCard[];
}

const PAGES = 'tcg-scanner-collection-v1';
const SESSION = 'tcg-scanner-session-v1';
const SESSION_GAP_MS = 3 * 60 * 60 * 1000;

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false; // quota or storage disabled
  }
}

/** Current session: continues while active, a new one starts after 3 hours without saving. */
export function currentSession(now = Date.now()): string {
  const s = read<{ id: string; last: number } | null>(SESSION, null);
  const id = s && now - s.last < SESSION_GAP_MS ? s.id : `s${now}`;
  write(SESSION, { id, last: now });
  return id;
}

export function loadPages(): SavedPage[] {
  return read<SavedPage[]>(PAGES, []);
}

export function deletePage(key: string): void {
  write(PAGES, loadPages().filter((p) => p.key !== key));
  void deletePageCrops(key);
}

/** Remove one card from a saved page; a page without cards is removed. Card numbers stay as scanned. */
export function deleteSavedCard(pageKey: string, number: number): void {
  const pages = loadPages();
  const page = pages.find((p) => p.key === pageKey);
  if (!page) return;
  page.cards = page.cards.filter((c) => c.number !== number);
  if (!page.cards.length) return deletePage(pageKey);
  write(PAGES, pages);
}

/** Apply a user decision to a saved card (identity, versions, flags); the value is repriced on reload. */
export function updateSavedCard(pageKey: string, number: number, patch: Partial<SavedCard>): boolean {
  const pages = loadPages();
  const card = pages.find((p) => p.key === pageKey)?.cards.find((c) => c.number === number);
  if (!card) return false;
  Object.assign(card, patch, { correctedAt: new Date().toISOString() });
  delete card.current;
  return write(PAGES, pages);
}

async function thumbnail(bitmap: ImageBitmap | null): Promise<string | null> {
  if (!bitmap) return null;
  try {
    const c = document.createElement('canvas');
    c.width = 63; c.height = 88;
    c.getContext('2d')!.drawImage(bitmap, 0, 0, 63, 88);
    return c.toDataURL('image/jpeg', 0.7);
  } catch {
    return null;
  }
}

/**
 * Save (or replace) the page of this generation. Returns false when storage is full or unavailable.
 * `displayCrop` supplies a larger crop of an instance for viewing (optional).
 */
export async function savePage(page: PageState, catalog: Catalog, displayCrop?: (id: string) => Promise<ImageBitmap | null>): Promise<boolean> {
  const cards: SavedCard[] = [];
  const key = `page-${page.generation}`;
  for (const i of page.instances) {
    let hasCrop = false;
    if (i.crop) {
      try {
        const display = await displayCrop?.(i.id).catch(() => null) ?? null;
        hasCrop = await putCrops(key, i.number, { canonical: await encodeBitmap(i.crop, 'image/png'), display: display ? await encodeBitmap(display, 'image/jpeg', 0.85) : null });
      } catch { hasCrop = false; }
    }
    const m: MatchResult | null = i.decision?.kind === 'matched' && i.state === 'confirmed' ? i.decision : null;
    const card = m ? displayCard(m, catalog) : undefined;
    const v = i.value;
    cards.push({
      number: i.number, identity: card?.id ?? null, name: card?.name ?? null, cardId: card?.cardId ?? null,
      state: i.state, languageAssumed: m?.languageAssumed ?? false, checkedByUser: !!i.userChoice?.identity,
      versions: m?.plausible.map((c) => c.id), preferredId: m ? m.preferred?.id ?? null : undefined, checkVariant: v?.cue?.kind === 'check-variant',
      value: m && v && v.currency ? { currency: v.currency, preferred: v.preferred?.amount ?? null, low: v.low, high: v.high,
        cue: !!v.cue || v.rangeIncomplete, converted: !!v.preferred?.converted } : null,
      thumb: await thumbnail(i.crop),
      hasCrop,
    });
  }
  const saved: SavedPage = {
    key, sessionId: currentSession(), pageNumber: page.pageNumber, revision: page.revision, savedAt: new Date().toISOString(),
    partial: page.instances.some((i) => i.state !== 'confirmed') || page.overLimit > 0, cards,
  };
  const pages = loadPages().filter((p) => p.key !== key);
  pages.push(saved);
  return write(PAGES, pages);
}

/**
 * Current values of saved cards: every card is priced again from the bundled snapshot in the
 * currently selected market (own quotes first, otherwise converted from the other market).
 */
export async function withCurrentPrices(pages: SavedPage[], prices: PriceService): Promise<SavedPage[]> {
  const versionsOf = (c: SavedCard) => c.versions ?? (c.identity ? [c.identity] : []);
  const ids = [...new Set(pages.flatMap((p) => p.cards.flatMap(versionsOf)))];
  const quotes = await prices.quotes(ids);
  const currency = prices.currency?.() ?? null;
  return pages.map((p) => ({
    ...p,
    cards: p.cards.map((c) => {
      if (!c.identity) return { ...c, current: null };
      const versions = versionsOf(c);
      const found = versions.map((id) => quotes.get(id) ?? null);
      const priced = found.filter((q) => q !== null);
      const preferred = quotes.get(c.preferredId === undefined ? c.identity : c.preferredId ?? '') ?? null;
      const amounts = priced.map((q) => q.amount);
      const complete = priced.length === versions.length;
      const current: SavedValue | null = priced.length || currency ? {
        currency: priced[0]?.currency ?? currency ?? 'EUR',
        preferred: preferred?.amount ?? null,
        low: amounts.length ? Math.min(...amounts) : null,
        high: complete && amounts.length ? Math.max(...amounts) : null,
        cue: (c.checkVariant ?? (c.versions ? false : !!c.value?.cue)) || !complete,
        converted: !!preferred?.converted,
      } : null;
      return { ...c, current };
    }),
  }));
}

/** Scan-time amount in another currency, via the snapshot's exchange rate. */
export function scanAmountIn(v: SavedValue | null, currency: string, usdToEur: number | null): number | null {
  if (!v || v.preferred === null) return null;
  if (v.currency === currency) return v.preferred;
  if (!usdToEur) return null;
  return v.currency === 'USD' && currency === 'EUR' ? v.preferred * usdToEur : v.currency === 'EUR' && currency === 'USD' ? v.preferred / usdToEur : null;
}

export interface Totals {
  cards: number;
  identified: number;
  confirmed: number;
  provisionalHigh: number;
  provisionalOpen: boolean;
  unpriced: number;
  currency: string;
}

/** Confirmed values, provisional upper bounds and unpriced cards are kept apart (never summed together). */
export function totals(pages: readonly SavedPage[]): Totals {
  const t: Totals = { cards: 0, identified: 0, confirmed: 0, provisionalHigh: 0, provisionalOpen: false, unpriced: 0, currency: 'USD' };
  for (const p of pages) {
    for (const c of p.cards) {
      t.cards++;
      if (!c.identity) continue;
      t.identified++;
      const v = c.current !== undefined ? c.current : c.value;
      if (!v || (v.preferred === null && v.low === null)) { t.unpriced++; continue; }
      t.currency = v.currency;
      if (!v.cue && !c.languageAssumed && v.preferred !== null) t.confirmed += v.preferred;
      else if (v.high === null) t.provisionalOpen = true;
      else t.provisionalHigh += v.high;
    }
  }
  return t;
}
