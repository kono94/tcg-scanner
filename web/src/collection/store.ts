// Saved pages for the Collection tab (per-viewer, in localStorage; a local store, not a sync service:
// collection/transfer.ts moves it between browsers as an exported file).
// Saving the same page generation again replaces it: quantities never double.
import { Catalog } from '../recognition/catalog';
import { displayCard, type MatchResult } from '../recognition/policy';
import type { ValueSummary } from '../recognition/value';
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
  /** Batch mode: why an unresolved card needs the user (AttentionReason of the session). */
  reason?: string;
  /** Batch mode: job token of a card waiting to be checked ('queued'); late results for another token are dropped. */
  queued?: number;
  /** Crop quality measured when the card was extracted (capture pixels, glare fraction). */
  quality?: { shortSide: number; glare: number };
  /** Closest catalog matches at recognition time (kept for unresolved cards), with the bundle version used. */
  hits?: { identity: string; score: number }[];
  bundle?: string;
}

export interface SavedPage {
  /** Original rectified appearances, including backs; not exact-card identities. */
  appearance?: string[];
  duplicateOverride?: { pageKey: string; at: string };
  key: string;
  sessionId: string;
  pageNumber: number;
  revision: number;
  savedAt: string;
  partial: boolean;
  cards: SavedCard[];
  /** Pages scanned in batch mode belong to one batch (review at the end). */
  batchId?: string;
  /** Games the scan was limited to (absent: all games). */
  games?: string[];
}

/** Unique per page load: page generations restart at 1 after a reload and must not overwrite older saves. */
const BOOT = Date.now().toString(36);

const PAGES = 'tcg-scanner-collection-v1';
const SESSION = 'tcg-scanner-session-v1';
const SESSIONS = 'tcg-scanner-sessions-v2';

export interface ScanSession {
  id: string;
  name: string;
  startedAt: string;
  finishedAt?: string;
  nextCapture: number;
  /** The user named it; an unnamed session that never saved a capture is not kept. */
  named?: boolean;
}

interface SessionRecords { activeId: string | null; sessions: ScanSession[] }

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

function defaultSessionName(at: string): string {
  return `Scan · ${new Date(at).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}`;
}

/** Short default for a new binder ("Binder 3"); the user can rename it. */
function nextBinderName(sessions: readonly ScanSession[]): string {
  const names = new Set(sessions.map((s) => s.name));
  let n = sessions.length + 1;
  while (names.has(`Binder ${n}`)) n++;
  return `Binder ${n}`;
}

/** Migrate grouping metadata only: original page keys, numbering, cards and crops stay intact. */
function sessionRecords(): SessionRecords {
  const stored = read<SessionRecords | null>(SESSIONS, null);
  if (stored) return stored;
  const groups = new Map<string, SavedPage[]>();
  for (const p of loadPages()) groups.set(p.sessionId, [...(groups.get(p.sessionId) ?? []), p]);
  const sessions = [...groups].map(([id, pages]) => {
    const startedAt = pages.map((p) => p.savedAt).sort()[0];
    return { id, startedAt, name: defaultSessionName(startedAt), nextCapture: Math.max(...pages.map((p) => p.pageNumber)) + 1 };
  });
  // The legacy session continues only if it was still active under its old three-hour rule.
  const legacy = read<{ id: string; last: number } | null>(SESSION, null);
  const recent = legacy && Date.now() - legacy.last < 3 * 60 * 60 * 1000;
  const records = { sessions, activeId: recent ? sessions.find((s) => s.id === legacy.id)?.id ?? null : null };
  // Do not persist an empty migration on every boot: new sessions are explicit.
  if (sessions.length) write(SESSIONS, records);
  return records;
}

export function loadScanSessions(): ScanSession[] { return sessionRecords().sessions; }

export function activeScanSession(): ScanSession | null {
  const records = sessionRecords();
  return records.sessions.find((s) => s.id === records.activeId) ?? null;
}

/** Unnamed sessions that never saved a capture would otherwise pile up as empty collections. */
function withoutUnusedActive(records: SessionRecords): SessionRecords {
  const active = records.sessions.find((s) => s.id === records.activeId);
  if (!active || active.named || active.nextCapture > 1 || loadPages().some((p) => p.sessionId === active.id)) return records;
  return { activeId: null, sessions: records.sessions.filter((s) => s !== active) };
}

export function startScanSession(name = '', now = Date.now()): ScanSession {
  const records = withoutUnusedActive(sessionRecords());
  const at = new Date(now).toISOString();
  const custom = name.trim().slice(0, 80);
  const session: ScanSession = { id: `s-${crypto.randomUUID()}`, name: custom || nextBinderName(records.sessions), startedAt: at, nextCapture: 1, ...(custom ? { named: true } : {}) };
  records.sessions.push(session);
  if (!write(SESSIONS, { ...records, activeId: session.id })) throw new Error('Could not save the session: browser storage is full or disabled.');
  return session;
}

export function resumeScanSession(id: string): boolean {
  const current = sessionRecords();
  const records = current.activeId === id ? current : withoutUnusedActive(current);
  const session = records.sessions.find((s) => s.id === id);
  if (!session) return false;
  delete session.finishedAt;
  return write(SESSIONS, { ...records, activeId: id });
}

export function renameScanSession(id: string, name: string): boolean {
  const records = sessionRecords();
  const session = records.sessions.find((s) => s.id === id);
  if (!session || !name.trim()) return false;
  session.name = name.trim().slice(0, 80);
  session.named = true;
  return write(SESSIONS, records);
}

export function finishScanSession(now = Date.now()): boolean {
  const records = withoutUnusedActive(sessionRecords());
  const session = records.sessions.find((s) => s.id === records.activeId);
  if (session) session.finishedAt = new Date(now).toISOString();
  return write(SESSIONS, { ...records, activeId: null });
}

/** Delete a named collection and its stored crops; no raw training uploads are removed. */
export async function deleteScanSession(id: string): Promise<boolean> {
  const pages = loadPages();
  const records = sessionRecords();
  if (!write(PAGES, pages.filter((p) => p.sessionId !== id))) return false;
  if (!write(SESSIONS, { activeId: records.activeId === id ? null : records.activeId, sessions: records.sessions.filter((s) => s.id !== id) })) {
    write(PAGES, pages);
    return false;
  }
  await Promise.all(pages.filter((p) => p.sessionId === id).map((p) => deletePageCrops(p.key)));
  return true;
}

/** Explicit lifetime: no date rollover or inactivity timeout. */
export function currentSession(now = Date.now()): string {
  return activeScanSession()?.id ?? startScanSession('', now).id;
}

export function nextCaptureNumber(id = activeScanSession()?.id): number {
  if (!id) return 1;
  return Math.max(loadScanSessions().find((s) => s.id === id)?.nextCapture ?? 1,
    ...loadPages().filter((p) => p.sessionId === id).map((p) => p.pageNumber + 1));
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
  return patchSavedCard(pageKey, number, { ...patch, correctedAt: new Date().toISOString() });
}

/** Machine update of a saved card (batch recognition result); keeps the page's partial flag current. */
export function patchSavedCard(pageKey: string, number: number, patch: Partial<SavedCard>): boolean {
  const pages = loadPages();
  const page = pages.find((p) => p.key === pageKey);
  const card = page?.cards.find((c) => c.number === number);
  if (!page || !card) return false;
  Object.assign(card, patch);
  for (const [k, v] of Object.entries(patch)) if (v === undefined) delete (card as unknown as Record<string, unknown>)[k];
  delete card.current;
  if (page.batchId) page.partial = page.cards.some((c) => c.state !== 'confirmed');
  return write(PAGES, pages);
}

/** Add or replace one saved page (by key). */
export function writePage(page: SavedPage): boolean {
  const pages = loadPages().filter((p) => p.key !== page.key);
  pages.push(page);
  if (!write(PAGES, pages)) return false;
  const records = sessionRecords();
  const session = records.sessions.find((s) => s.id === page.sessionId);
  if (session) { session.nextCapture = Math.max(session.nextCapture, page.pageNumber + 1); write(SESSIONS, records); }
  return true;
}

export interface ImportResult { addedPages: number; skippedPages: number; addedSessions: number }

/**
 * Merge exported binders and pages into this browser. Pages already present (same key) are skipped, so
 * importing a file twice never doubles quantities; existing binders keep their names; the active binder
 * stays as it is. Returns null when storage is full (nothing is changed then).
 */
export function importRecords(sessions: readonly ScanSession[], pages: readonly SavedPage[]): ImportResult | null {
  const current = loadPages();
  const records = sessionRecords();
  const known = new Set(current.map((p) => p.key));
  const added = pages.filter((p) => !known.has(p.key)).map((p) => ({
    ...p,
    // No queue owns these cards here and their crops stayed on the other device.
    cards: p.cards.map(({ current: _, queued: __, ...c }) => c.state === 'queued' ? { ...c, state: 'needs-attention', reason: 'error', hasCrop: false } : { ...c, hasCrop: false }),
  }));
  let addedSessions = 0;
  for (const id of new Set(added.map((p) => p.sessionId))) {
    const own = added.filter((p) => p.sessionId === id);
    const next = Math.max(...own.map((p) => p.pageNumber)) + 1;
    const existing = records.sessions.find((s) => s.id === id);
    if (existing) { existing.nextCapture = Math.max(existing.nextCapture, next); continue; }
    const imported = sessions.find((s) => s.id === id);
    const startedAt = imported?.startedAt ?? own.map((p) => p.savedAt).sort()[0];
    records.sessions.push({ ...(imported ?? { id, name: defaultSessionName(startedAt), startedAt }), nextCapture: Math.max(imported?.nextCapture ?? 1, next) });
    addedSessions++;
  }
  if (!added.length) return { addedPages: 0, skippedPages: pages.length, addedSessions: 0 };
  if (!write(PAGES, [...current, ...added])) return null;
  if (!write(SESSIONS, records)) { write(PAGES, current); return null; }
  return { addedPages: added.length, skippedPages: pages.length - added.length, addedSessions };
}

/** Saved-card fields for a match (identity, versions, scan-time value), shared by page saves and batch results. */
export function matchFields(m: MatchResult | null, v: ValueSummary | null, catalog: Catalog): Pick<SavedCard, 'identity' | 'name' | 'cardId' | 'languageAssumed' | 'versions' | 'preferredId' | 'checkVariant' | 'value'> {
  const card = m ? displayCard(m, catalog) : undefined;
  return {
    identity: card?.id ?? null, name: card?.name ?? null, cardId: card?.cardId ?? null, languageAssumed: m?.languageAssumed ?? false,
    versions: m?.plausible.map((c) => c.id), preferredId: m ? m.preferred?.id ?? null : undefined, checkVariant: v?.cue?.kind === 'check-variant',
    value: m && v && v.currency ? { currency: v.currency, preferred: v.preferred?.amount ?? null, low: v.low, high: v.high,
      cue: !!v.cue || v.rangeIncomplete, converted: !!v.preferred?.converted } : null,
  };
}

/** A saved card the user still has to look at: unresolved, or matched with a pricier variant possible. */
export function savedNeedsUser(c: SavedCard): boolean {
  return !c.checkedByUser && (c.state === 'needs-attention' || (c.state === 'confirmed' && !!c.checkVariant));
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
  const key = `page-${BOOT}-${page.generation}`;
  // Bind ownership before any async crop work; saving this generation again keeps its number.
  const previous = loadPages().find((p) => p.key === key);
  const sessionId = previous?.sessionId ?? page.scanSessionId ?? currentSession();
  const pageNumber = previous?.pageNumber ?? page.pageNumber;
  for (const i of page.instances) {
    // Card backs are not cards: they are never saved, so they never enter counts or totals.
    if (i.state === 'back') continue;
    let hasCrop = false;
    if (i.crop) {
      try {
        const display = await displayCrop?.(i.id).catch(() => null) ?? null;
        hasCrop = await putCrops(key, i.number, { canonical: await encodeBitmap(i.crop, 'image/png'), display: display ? await encodeBitmap(display, 'image/jpeg', 0.85) : null });
      } catch { hasCrop = false; }
    }
    const m: MatchResult | null = i.decision?.kind === 'matched' && i.state === 'confirmed' ? i.decision : null;
    cards.push({
      number: i.number, ...matchFields(m, i.value, catalog), state: i.state, checkedByUser: !!i.userChoice?.identity,
      thumb: await thumbnail(i.crop),
      hasCrop,
    });
  }
  const saved: SavedPage = {
    key, sessionId, pageNumber, revision: page.revision, savedAt: previous?.savedAt ?? new Date().toISOString(),
    partial: page.instances.some((i) => i.state !== 'confirmed' && i.state !== 'back') || page.overLimit > 0, cards,
  };
  return writePage(saved);
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

/**
 * An assumed language only makes a value provisional when the language versions are priced
 * differently: in the current snapshot, English and German quotes are identical.
 */
export function languageMatters(languageAssumed: boolean, v: { low: number | null; high: number | null }): boolean {
  return languageAssumed && (v.low === null || v.high === null || v.low !== v.high);
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
      if (!v.cue && !languageMatters(c.languageAssumed, v) && v.preferred !== null) t.confirmed += v.preferred;
      else if (v.high === null) t.provisionalOpen = true;
      else t.provisionalHigh += v.high;
    }
  }
  return t;
}
