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

export interface Detected {
  width: number;
  height: number;
  boxes: OrientedBox[];
  quads?: Quad[];
  ms: number;
}

export interface ExtractedCard {
  canonical: Blob;
  display: Blob;
  thumb: Blob;
  quality: Quality;
}

/** Worker-backed operations (the WorkerRecognizer in the app, fakes in tests). */
export interface BatchBackend {
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
  id: string;
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
  thumbUrl(blob: Blob): Promise<string | null>;
  loadBatch(): BatchRecord | null;
  saveBatch(batch: BatchRecord | null): void;
  sessionId(): string;
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
    const pages = active ? this.store.loadPages().filter((p) => keys.has(p.key)) : [];
    this.cached = {
      active, pages: pages.length, cards: pages.reduce((n, p) => n + p.cards.length, 0), waiting: this.waiting(),
      running: this.running ? { pageKey: this.running.pageKey, number: this.running.number } : null,
      needsUser: pages.reduce((n, p) => n + p.cards.filter(savedNeedsUser).length, 0),
      capturing: this.capturing, paused: this.paused, stalled: this.stalled, full: this.full,
    };
    return this.cached;
  }

  /** Pages of the active batch, in scan order. */
  pages(): SavedPage[] {
    const active = this.store.loadBatch();
    if (!active) return [];
    const byKey = new Map(this.store.loadPages().map((p) => [p.key, p]));
    return active.pageKeys.map((k) => byKey.get(k)).filter((p): p is SavedPage => !!p);
  }

  /** Admission control before a capture is taken (not after the work is queued). */
  canCapture(): boolean {
    this.updateFull();
    return !this.capturing && !this.full && !this.stalled;
  }

  /**
   * One page: detect, store every card's crops, release the capture, save the page with its cards
   * queued. The bitmap is transferred to the worker. Returns the page number and card count.
   */
  async capture(bitmap: ImageBitmap, onDetected?: (captureId: number, detected: Detected, quads: Quad[]) => Promise<void>):
    Promise<{ pageNumber: number | null; cards: number; overLimit: number; failed: number }> {
    if (!this.canCapture()) {
      bitmap.close();
      throw new Error(this.full ? 'Batch backlog is full' : 'A capture is already being saved');
    }
    this.capturing = true;
    this.emit();
    const captureId = CAPTURE_BASE + ++this.captureCounter;
    try {
      const detected = await this.backend.detect(captureId, bitmap);
      const limit = this.options.maxInstances ?? 48;
      const ordered = readingOrder(detected.boxes.map((box, i) => ({ box, corners: detected.quads?.[i] })));
      const quads = ordered.slice(0, limit).map(({ box, corners }) => portraitQuad(corners ?? boxCorners(box)));
      const overLimit = Math.max(0, ordered.length - limit);
      if (onDetected) await onDetected(captureId, detected, quads).catch(() => {});
      if (!quads.length) return { pageNumber: null, cards: 0, overLimit, failed: 0 };
      const batch = this.store.loadBatch() ?? { id: this.now().toString(36), startedAt: new Date(this.now()).toISOString(), pageKeys: [], nextPage: 1 };
      const pageNumber = batch.nextPage;
      const pageKey = `batch-${batch.id}-${pageNumber}`;
      const cards: SavedCard[] = [];
      let failed = 0;
      // One card at a time: a few hundred kB in flight, whatever the card count.
      for (let i = 0; i < quads.length; i++) {
        const number = i + 1;
        const base: SavedCard = { number, identity: null, name: null, cardId: null, state: 'queued', languageAssumed: false, checkedByUser: false, value: null, thumb: null };
        try {
          const crops = await this.backend.extract(captureId, quads[i]);
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
      const page: SavedPage = { key: pageKey, sessionId: this.store.sessionId(), pageNumber, revision: 1, savedAt: new Date(this.now()).toISOString(),
        partial: true, cards, batchId: batch.id };
      if (!this.store.writePage(page)) throw new Error('Could not save the page: browser storage is full or disabled.');
      this.store.saveBatch({ ...batch, pageKeys: [...batch.pageKeys, pageKey], nextPage: pageNumber + 1 });
      for (const c of cards) if (c.state === 'queued') this.queue.push({ pageKey, number: c.number, token: c.queued! });
      return { pageNumber, cards: cards.length, overLimit, failed };
    } finally {
      this.backend.release(captureId);
      this.capturing = false;
      this.emit();
      this.kick();
    }
  }

  /** Background work stops while the page is hidden (a running job finishes; nothing new starts). */
  setPaused(paused: boolean): void {
    if (this.paused === paused) return;
    this.paused = paused;
    this.emit();
    this.kick();
  }

  /** Close the active batch (after the review). Cards still waiting keep being checked. */
  finish(): void {
    this.store.saveBatch(null);
    this.emit();
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
