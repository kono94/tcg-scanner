// Batch scanning ("scan the binder now, review at the end"):
//  capture -> detect -> straightened crops of every card stored (IndexedDB) -> capture released
//  -> the page is saved to the Collection right away, its cards 'queued'
//  -> one background job at a time embeds + searches a stored crop and writes the result.
// The queue is derived from the saved pages, so it survives a reload or a killed tab: queued cards
// are picked up again on the next boot. Nothing capture-sized is kept between pages.
import type { Catalog } from '../recognition/catalog';
import type { Hit } from '../recognition/gallery';
import { decideMatch, type MatchThresholds } from '../recognition/policy';
import { summarizeValue, type ValueSettings } from '../recognition/value';
import type { PriceService } from '../prices/service';
import type { Quality } from '../inference/protocol';
import type { Quad } from '../vision/image';
import { boxCorners, portraitQuad, readingOrder, type OrientedBox } from '../vision/obb';
import { attentionReason } from '../session/page';
import { matchFields, savedNeedsUser, type SavedCard, type SavedPage } from '../collection/store';
import type { SavedCrops } from '../collection/crops';
import { probableDuplicate } from './duplicates';

export interface Detected {
  width: number;
  height: number;
  boxes: OrientedBox[];
  quads?: Quad[];
  ms: number;
}

export interface ExtractedCard {
  appearance?: string;
  canonical: Blob;
  display: Blob;
  thumb: Blob;
  quality: Quality;
}

/** Worker-backed operations (the WorkerRecognizer in the app, fakes in tests). */
export interface BatchBackend {
  fingerprint?(canonical: Blob): Promise<string>;
  detect(captureId: number, bitmap: ImageBitmap): Promise<Detected>;
  extract(captureId: number, quad: Quad): Promise<ExtractedCard>;
  /** Free the capture (and the worker's capture-sized buffer). */
  release(captureId: number): void;
  /** Embed + search a stored canonical crop. */
  search(canonical: Blob): Promise<Hit[]>;
  /** Runtime failure (worker crashed / models failed): background work cannot continue until a reload. */
  failure(): string | null;
}

export interface BatchRecord {
  pendingDuplicate?: { page: SavedPage; matchedPageKey: string; matchedPageNumber: number };
  id: string;
  sessionId?: string;
  startedAt: string;
  pageKeys: string[];
  nextPage: number;
}

/** Persistence: saved pages (localStorage), crops (IndexedDB), the active batch record. */
export interface BatchStore {
  loadPages(): SavedPage[];
  writePage(page: SavedPage): boolean;
  patchCard(pageKey: string, number: number, patch: Partial<SavedCard>): boolean;
  /** Remove a card from its saved page (a page left without cards is removed). */
  removeCard(pageKey: string, number: number): void;
  putCrops(pageKey: string, number: number, crops: SavedCrops): Promise<boolean>;
  getCrops(pageKey: string, number: number): Promise<SavedCrops | null>;
  removeCrops?(pageKey: string): Promise<void>;
  thumbUrl(blob: Blob): Promise<string | null>;
  loadBatch(): BatchRecord | null;
  saveBatch(batch: BatchRecord | null): boolean | void;
  sessionId(): string;
  nextCaptureNumber?(sessionId: string): number;
}

export interface BatchOptions {
  catalog: Catalog;
  prices: PriceService;
  preferredLanguage: () => string;
  bundleVersion: string;
  thresholds?: MatchThresholds;
  valueSettings?: ValueSettings;
  /** Explicit limit on cards per capture (as in single-page mode). */
  maxInstances?: number;
  /**
   * Backlog bound: no new capture while this many cards wait, until the backlog drops to `resumeAt`.
   * Crops wait on disk, so memory stays flat; the bound keeps the review from falling far behind.
   */
  backlogLimit?: number;
  resumeAt?: number;
  /** Per-card budget; a stuck job becomes an explicit 'timeout' (slow phones need ~2-15 s per card). */
  cardTimeoutMs?: number;
  /** Extra condition for starting a job (e.g. the single-page session is idle). */
  canRun?: () => boolean;
  now?: () => number;
}

export interface BatchStatus {
  duplicate: { matchedPageNumber: number; cards: number } | null;
  /** The batch being scanned (null: none; the next capture starts one). */
  active: BatchRecord | null;
  pages: number;
  cards: number;
  /** Cards waiting to be checked, across all saved pages (includes the running one). */
  waiting: number;
  running: { pageKey: string; number: number } | null;
  /** Cards of the active batch the user should look at. */
  needsUser: number;
  capturing: boolean;
  /** Background work paused (page hidden). */
  paused: boolean;
  /** Background work stopped by a runtime failure; queued cards stay saved for after a reload. */
  stalled: string | null;
  /** Backlog limit reached: no captures until it drops. */
  full: boolean;
}

interface Job {
  pageKey: string;
  number: number;
  token: number;
}

/** Captures of the batch session use their own ID range (the worker holds one capture at a time). */
const CAPTURE_BASE = 1_000_000;

export class BatchSession {
  private queue: Job[] = [];
  private running: Job | null = null;
  private capturing = false;
  private paused = false;
  private stalled: string | null = null;
  private full = false;
  private captureCounter = 0;
  private tokenCounter = 0;
  private listeners = new Set<() => void>();
  private cached: BatchStatus | null = null;
  private readonly now: () => number;

  constructor(private readonly backend: BatchBackend, private readonly store: BatchStore, private readonly options: BatchOptions) {
    this.now = options.now ?? (() => Date.now());
    this.restore();
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    this.cached = null;
    for (const fn of this.listeners) fn();
  }

  /** Saved pages changed outside the session (user decisions, deletions): recompute the status on next read. */
  invalidate(): void {
    this.cached = null;
  }

  /** Rebuild the queue from saved pages: every card still 'queued' with a stored crop, oldest first. */
  restore(): void {
    const pages = [...this.store.loadPages()].sort((a, b) => a.savedAt.localeCompare(b.savedAt));
    const known = new Set(this.queue.map((j) => `${j.pageKey}#${j.number}`));
    if (this.running) known.add(`${this.running.pageKey}#${this.running.number}`);
    for (const p of pages) {
      for (const c of p.cards) {
        if (c.state !== 'queued' || known.has(`${p.key}#${c.number}`)) continue;
        let token = c.queued;
        if (token === undefined) { token = this.token(); this.store.patchCard(p.key, c.number, { queued: token }); }
        this.queue.push({ pageKey: p.key, number: c.number, token });
      }
    }
    this.emit();
  }

  private token(): number {
    return this.now() * 1000 + (this.tokenCounter++ % 1000);
  }

  private get limit() { return this.options.backlogLimit ?? 90; }
  private get resumeAt() { return this.options.resumeAt ?? 60; }

  private waiting(): number {
    return this.queue.length + (this.running ? 1 : 0);
  }

  private updateFull() {
    const w = this.waiting();
    if (w >= this.limit) this.full = true;
    else if (w <= this.resumeAt) this.full = false;
  }

  status(): BatchStatus {
    if (this.cached) return this.cached;
    this.updateFull();
    const active = this.store.loadBatch();
    const keys = new Set(active?.pageKeys ?? []);
    const pages = active ? this.store.loadPages().filter((p) => keys.has(p.key) || p.batchId === active.id) : [];
    this.cached = {
      active, pages: pages.length, cards: pages.reduce((n, p) => n + p.cards.length, 0), waiting: this.waiting(),
      running: this.running ? { pageKey: this.running.pageKey, number: this.running.number } : null,
      needsUser: pages.reduce((n, p) => n + p.cards.filter(savedNeedsUser).length, 0),
      capturing: this.capturing, paused: this.paused, stalled: this.stalled, full: this.full,
      duplicate: active?.pendingDuplicate && !pages.some((p) => p.key === active.pendingDuplicate!.page.key)
        ? { matchedPageNumber: active.pendingDuplicate.matchedPageNumber, cards: active.pendingDuplicate.page.cards.length } : null,
    };
    return this.cached;
  }

  /** Pages of the active batch, in scan order. */
  pages(): SavedPage[] {
    const active = this.store.loadBatch();
    if (!active) return [];
    const byKey = new Map(this.store.loadPages().map((p) => [p.key, p]));
    return [...byKey.values()].filter((p) => active.pageKeys.includes(p.key) || p.batchId === active.id).sort((a, b) => a.pageNumber - b.pageNumber);
  }

  /** Admission control before a capture is taken (not after the work is queued). */
  canCapture(): boolean {
    this.updateFull();
    // A held possible repeat needs an explicit Keep/Dismiss first: a new capture never discards it.
    return !this.capturing && !this.full && !this.stalled && !this.status().duplicate;
  }

  /**
   * One page: detect, store every card's crops, release the capture, save the page with its cards
   * queued. The bitmap is transferred to the worker. Returns the page number and card count.
   */
  async capture(bitmap: ImageBitmap, onDetected?: (captureId: number, detected: Detected, quads: Quad[]) => Promise<void>,
    { holdRepeat = true }: { holdRepeat?: boolean } = {}):
    Promise<{ pageNumber: number | null; cards: number; overLimit: number; failed: number; duplicate?: number; skipped?: boolean }> {
    if (!this.canCapture()) {
      bitmap.close();
      throw new Error(this.full ? 'Batch backlog is full' : this.status().duplicate ? 'Keep or dismiss the possible repeat first' : 'A capture is already being saved');
    }
    const sessionId = this.store.sessionId();
    const existing = this.store.loadBatch();
    const batch = existing && (!existing.sessionId || existing.sessionId === sessionId) ? existing
      : { id: `${this.now().toString(36)}-${++this.tokenCounter}`, sessionId, startedAt: new Date(this.now()).toISOString(), pageKeys: [], nextPage: 1 };
    const pageNumber = this.store.nextCaptureNumber?.(sessionId) ?? batch.nextPage;
    this.capturing = true;
    this.emit();
    const captureId = CAPTURE_BASE + ++this.captureCounter;
    try {
      delete batch.pendingDuplicate;
      const detected = await this.backend.detect(captureId, bitmap);
      const limit = this.options.maxInstances ?? 48;
      const ordered = readingOrder(detected.boxes.map((box, i) => ({ box, corners: detected.quads?.[i] })));
      const quads = ordered.slice(0, limit).map(({ box, corners }) => portraitQuad(corners ?? boxCorners(box)));
      const overLimit = Math.max(0, ordered.length - limit);
      if (onDetected) await onDetected(captureId, detected, quads).catch(() => {});
      if (!quads.length) return { pageNumber: null, cards: 0, overLimit, failed: 0 };
      const pageKey = `batch-${batch.id}-${pageNumber}`;
      const cards: SavedCard[] = [];
      const appearance: string[] = [];
      let failed = 0;
      // One card at a time: a few hundred kB in flight, whatever the card count.
      for (let i = 0; i < quads.length; i++) {
        const number = i + 1;
        const base: SavedCard = { number, identity: null, name: null, cardId: null, state: 'queued', languageAssumed: false, checkedByUser: false, value: null, thumb: null };
        try {
          const crops = await this.backend.extract(captureId, quads[i]);
          if (crops.appearance) appearance.push(crops.appearance);
          const stored = await this.store.putCrops(pageKey, number, { canonical: crops.canonical, display: crops.display });
          const thumb = await this.store.thumbUrl(crops.thumb);
          if (!stored) failed++;
          cards.push(stored ? { ...base, thumb, hasCrop: true, quality: crops.quality, queued: this.token() }
            : { ...base, thumb, hasCrop: false, quality: crops.quality, state: 'needs-attention', reason: 'error' });
        } catch (err) {
          if (this.backend.failure()) throw err;
          failed++;
          cards.push({ ...base, state: 'needs-attention', reason: 'error', hasCrop: false });
        }
      }
      const page: SavedPage = { key: pageKey, sessionId, pageNumber, revision: 1, savedAt: new Date(this.now()).toISOString(),
        partial: true, cards, batchId: batch.id, appearance: !overLimit && appearance.length === cards.length ? appearance : undefined };
      const duplicate = page.appearance ? await this.findDuplicate(page) : null;
      // Automatic captures skip a probable repeat outright (the page was not turned); only a
      // manual capture holds it for an explicit Keep anyway / Dismiss.
      if (duplicate && !holdRepeat) {
        await this.store.removeCrops?.(pageKey);
        return { pageNumber: null, cards: 0, overLimit, failed, duplicate: duplicate.pageNumber, skipped: true };
      }
      if (duplicate) {
        if (this.store.saveBatch({ ...batch, sessionId, pendingDuplicate: { page, matchedPageKey: duplicate.key, matchedPageNumber: duplicate.pageNumber } }) === false) {
          await this.store.removeCrops?.(pageKey);
          throw new Error('Could not hold the possible repeat: browser storage is full. Capture not saved.');
        }
        return { pageNumber: null, cards: 0, overLimit, failed, duplicate: duplicate.pageNumber };
      }
      if (!this.store.writePage(page)) throw new Error('Could not save the page: browser storage is full or disabled.');
      this.store.saveBatch({ ...batch, sessionId, pageKeys: [...batch.pageKeys, pageKey], nextPage: pageNumber + 1 });
      for (const c of cards) if (c.state === 'queued') this.queue.push({ pageKey, number: c.number, token: c.queued! });
      return { pageNumber, cards: cards.length, overLimit, failed };
    } finally {
      this.backend.release(captureId);
      this.capturing = false;
      this.emit();
      this.kick();
    }
  }

  private async findDuplicate(page: SavedPage): Promise<SavedPage | null> {
    const candidates = this.store.loadPages().filter((p) => p.sessionId === page.sessionId).sort((a, b) => b.savedAt.localeCompare(a.savedAt));
    let legacyBudget = 48; // Bounded old-crop decoding per capture; new signatures need no decoding.
    let inspected = 0;
    for (const previous of candidates) {
      if (++inspected % 8 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
      let appearance = previous.appearance;
      if (!appearance && this.backend.fingerprint && previous.cards.length <= legacyBudget) {
        legacyBudget -= previous.cards.length;
        const values: string[] = [];
        for (const card of previous.cards) {
          try {
            const crops = await this.store.getCrops(previous.key, card.number);
            if (!crops) break;
            values.push(await this.backend.fingerprint(crops.canonical));
          } catch (err) { if (this.backend.failure()) throw err; break; }
        }
        if (values.length === previous.cards.length && values.length) {
          appearance = values;
          // Reload before the patch: a background recognition result may have arrived meanwhile.
          const latest = this.store.loadPages().find((p) => p.key === previous.key);
          if (latest) this.store.writePage({ ...latest, appearance });
        }
      }
      if (appearance && probableDuplicate(page.appearance!, appearance)) return previous;
    }
    return null;
  }

  /** Possible repeats are excluded from collection counts/recognition until explicit override. */
  keepDuplicate(): boolean {
    if (this.capturing) return false;
    const batch = this.store.loadBatch(), pending = batch?.pendingDuplicate;
    if (!batch || !pending) return false;
    const existing = this.store.loadPages().find((p) => p.key === pending.page.key);
    const page = existing ?? { ...pending.page, pageNumber: this.store.nextCaptureNumber?.(pending.page.sessionId) ?? pending.page.pageNumber,
      duplicateOverride: { pageKey: pending.matchedPageKey, at: new Date(this.now()).toISOString() } };
    if (!existing && !this.store.writePage(page)) return false;
    this.store.saveBatch({ ...batch, pageKeys: [...new Set([...batch.pageKeys, page.key])], nextPage: page.pageNumber + 1, pendingDuplicate: undefined });
    this.restore();
    this.kick();
    return true;
  }

  async discardDuplicate(): Promise<void> {
    const batch = this.store.loadBatch(), pending = batch?.pendingDuplicate;
    if (!batch || !pending) return;
    // Never delete an explicitly kept page if a previous metadata write was interrupted.
    if (!this.store.loadPages().some((p) => p.key === pending.page.key)) await this.store.removeCrops?.(pending.page.key);
    this.store.saveBatch({ ...batch, pendingDuplicate: undefined });
    this.emit();
  }

  /** Deletion retires queued jobs; running callbacks already check that their saved card exists. */
  pruneDeleted(): void {
    this.queue = this.queue.filter((j) => this.current(j) !== null);
    this.emit();
    this.kick();
  }

  /** Background work stops while the page is hidden (a running job finishes; nothing new starts). */
  setPaused(paused: boolean): void {
    if (this.paused === paused) return;
    this.paused = paused;
    this.emit();
    this.kick();
  }

  /** Close the active batch (after the review). Cards still waiting keep being checked. */
  finish(): boolean {
    // An admitted capture must finish storing, and a held repeat must be kept or dismissed, first.
    if (this.capturing || this.status().duplicate) return false;
    this.store.saveBatch(null);
    this.emit();
    return true;
  }

  /** Start the next job when allowed; at most one runs. */
  kick(): void {
    if (this.running || this.paused || this.stalled || this.capturing || !this.queue.length) return;
    if (this.options.canRun && !this.options.canRun()) return;
    const job = this.queue.shift()!;
    this.running = job;
    this.emit();
    void this.run(job).finally(() => {
      this.running = null;
      this.emit();
      setTimeout(() => this.kick(), 0);
    });
  }

  private current(job: Job): SavedCard | null {
    const card = this.store.loadPages().find((p) => p.key === job.pageKey)?.cards.find((c) => c.number === job.number);
    // Deleted, decided by the user meanwhile, or re-queued under another token: the result is stale.
    return card && card.state === 'queued' && card.queued === job.token ? card : null;
  }

  private async run(job: Job): Promise<void> {
    const card = this.current(job);
    if (!card) return;
    const crops = await this.store.getCrops(job.pageKey, job.number);
    if (!crops) {
      this.store.patchCard(job.pageKey, job.number, { state: 'needs-attention', reason: 'error', queued: undefined, hasCrop: false });
      return;
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    let hits: Hit[];
    try {
      const timeout = new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('timeout')), this.options.cardTimeoutMs ?? 60000); });
      hits = await Promise.race([this.backend.search(crops.canonical), timeout]);
    } catch (err) {
      const failure = this.backend.failure();
      if (failure) {
        // Keep the card queued (it is saved that way): it is checked after a reload.
        this.stalled = failure;
        this.queue.unshift(job);
        return;
      }
      if (!this.current(job)) return;
      const timedOut = (err as Error).message === 'timeout';
      this.store.patchCard(job.pageKey, job.number, { state: 'needs-attention', reason: timedOut ? 'timeout' : 'error', queued: undefined, bundle: this.options.bundleVersion });
      return;
    } finally {
      clearTimeout(timer);
    }
    const decision = decideMatch(hits, this.options.catalog, { preferredLanguage: this.options.preferredLanguage(), languageFromSetting: true, thresholds: this.options.thresholds });
    const reason = attentionReason(decision, card.quality ?? null);
    let value = null;
    if (decision.kind === 'matched') {
      try {
        value = summarizeValue(decision, await this.options.prices.quotes(decision.plausible.map((c) => c.id)), this.options.valueSettings);
      } catch { value = null; /* unknown, never zero */ }
    }
    if (!this.current(job)) return;
    // Card backs are not collection cards (as in single-page mode, they are never saved).
    if (decision.kind === 'back') { this.store.removeCard(job.pageKey, job.number); return; }
    const m = decision.kind === 'matched' ? decision : null;
    const fields = matchFields(m, value, this.options.catalog);
    const keepHits = !m || fields.checkVariant;
    this.store.patchCard(job.pageKey, job.number, {
      ...fields, state: m ? 'confirmed' : 'needs-attention', reason: reason ?? undefined, queued: undefined, bundle: this.options.bundleVersion,
      hits: keepHits ? hits.slice(0, 10).map((h) => ({ identity: h.identity, score: +h.score.toFixed(4) })) : undefined,
    });
  }
}
