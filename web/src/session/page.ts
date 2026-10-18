// Page session: one immutable capture, a variable-length list of physical card instances,
// a single bounded recognizer and explicit per-card states (docs/SCANNER_UX.md).
import { Catalog } from '../recognition/catalog';
import type { Hit } from '../recognition/gallery';
import { decideMatch, userChoiceMatch, type BackThresholds, type MatchDecision, type MatchThresholds } from '../recognition/policy';
import { summarizeValue, type ValueSettings, type ValueSummary } from '../recognition/value';
import type { PriceService } from '../prices/service';
import type { Quality, RecognitionStep } from '../inference/protocol';
import type { Quad } from '../vision/image';
import { boxCorners, portraitQuad, readingOrder, type OrientedBox } from '../vision/obb';

/**
 * 'back': the crop shows a card back. Terminal like 'confirmed', but not a card: never priced,
 * reviewed, counted as ready/attention or saved to the collection.
 */
export type CardState = 'queued' | 'recognizing' | 'verifying' | 'confirmed' | 'needs-attention' | 'skipped' | 'back';
export type Step = 'rectify' | 'embed' | 'search' | 'verify';
export type AttentionReason = 'glare' | 'move-closer' | 'not-found' | 'match-unclear' | 'timeout' | 'error';
export type ValueState = 'idle' | 'loading' | 'ready' | 'failed';

export interface CardInstance {
  id: string;
  /** Physical-instance number in reading order, stable within the capture. */
  number: number;
  box: OrientedBox;
  /** Card outline: predicted corners, or the rotated box; capture source pixels, TL, TR, BR, BL. */
  quad: Quad;
  state: CardState;
  step: Step | null;
  revision: number;
  queuedAt: number;
  startedAt: number | null;
  settledAt: number | null;
  crop: ImageBitmap | null;
  decision: MatchDecision | null;
  quality: Quality | null;
  reason: AttentionReason | null;
  value: ValueSummary | null;
  valueState: ValueState;
  ms: { rectify: number; embed: number; search: number } | null;
  /** Full retrieval shortlist, kept for the compare view. */
  hits: Hit[] | null;
  /** Set when the user confirmed or corrected the result; machine evidence stays in `hits`. */
  userChoice: { identity: string | null; at: number } | null;
}

export type PageStatus = 'empty' | 'detecting' | 'processing' | 'settled' | 'failed';

export interface PageState {
  /** Persistent collection session that owned this capture when pixels were acquired. */
  scanSessionId?: string;
  generation: number;
  pageNumber: number;
  captureId: number | null;
  /** Candidate revision of this page; a repeat capture replaces the previous candidate. */
  revision: number;
  captureSize: { width: number; height: number } | null;
  capturedAt: number | null;
  status: PageStatus;
  instances: CardInstance[];
  /** Detections beyond the explicit resource limit; surfaced, never silently dropped. */
  overLimit: number;
  detectMs: number | null;
  error: string | null;
  savedRevision: number | null;
}

export interface RecognitionJob {
  captureId: number;
  instanceId: string;
  revision: number;
  quad: Quad;
}

export interface RecognitionOutput {
  hits: Hit[];
  quality: Quality;
  ms: { rectify: number; embed: number; search: number };
}

/** Inference backend (worker client in the app, fakes in tests). */
export interface Recognizer {
  /** `quads` (same order as boxes) are predicted card corners when the detector provides them. */
  detect(captureId: number, bitmap: ImageBitmap): Promise<{ width: number; height: number; boxes: OrientedBox[]; quads?: Quad[]; ms: number }>;
  recognize(job: RecognitionJob, onStep: (step: RecognitionStep, crop?: ImageBitmap) => void): Promise<RecognitionOutput>;
  release(captureId: number): void;
}

export interface SessionOptions {
  catalog: Catalog;
  prices: PriceService;
  preferredLanguage: () => string;
  thresholds?: MatchThresholds;
  backThresholds?: BackThresholds;
  valueSettings?: ValueSettings;
  /** Per-card budget before an explicit timeout state. */
  cardTimeoutMs?: number;
  /** Explicit resource limit on instances per capture. */
  maxInstances?: number;
  now?: () => number;
}

const GLARE_LIMIT = 0.08; // uncalibrated heuristic
const MIN_SHORT_SIDE = 80; // capture pixels, uncalibrated

/** Why an unmatched card needs the user (null when matched). Shared with batch recognition. */
export function attentionReason(decision: MatchDecision, quality: Quality | null): AttentionReason | null {
  if (decision.kind === 'matched' || decision.kind === 'back') return null;
  return quality && quality.glare > GLARE_LIMIT ? 'glare'
    : quality && quality.shortSide < MIN_SHORT_SIDE ? 'move-closer'
      : decision.kind === 'match-unclear' ? 'match-unclear' : 'not-found';
}

class Stale extends Error {}

export class PageSession {
  page: PageState;
  private captureCounter = 0;
  private queue: string[] = [];
  private running = false;
  private listeners = new Set<() => void>();
  private readonly now: () => number;

  constructor(private readonly recognizer: Recognizer, private readonly options: SessionOptions) {
    this.now = options.now ?? (() => performance.now());
    this.page = this.emptyPage(1, 1);
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    for (const fn of this.listeners) fn();
  }

  private emptyPage(generation: number, pageNumber: number): PageState {
    return { generation, pageNumber, captureId: null, revision: 0, captureSize: null, capturedAt: null, status: 'empty',
      instances: [], overLimit: 0, detectMs: null, error: null, savedRevision: null };
  }

  get busy(): boolean {
    return this.page.status === 'detecting' || this.page.status === 'processing';
  }

  instance(id: string): CardInstance | undefined {
    return this.page.instances.find((i) => i.id === id);
  }

  /** Start a batch on one immutable capture. Replaces this page's candidate result revision. */
  async capture(bitmap: ImageBitmap): Promise<void> {
    if (this.busy) {
      bitmap.close();
      throw new Error('A batch is already active');
    }
    const generation = this.page.generation;
    const captureId = ++this.captureCounter;
    if (this.page.captureId !== null) this.recognizer.release(this.page.captureId);
    for (const i of this.page.instances) i.crop?.close();
    this.queue = [];
    this.page = { ...this.page, captureId, revision: this.page.revision + 1, captureSize: { width: bitmap.width, height: bitmap.height },
      capturedAt: this.now(), status: 'detecting', instances: [], overLimit: 0, detectMs: null, error: null };
    this.emit();
    let detected;
    try {
      detected = await this.recognizer.detect(captureId, bitmap);
    } catch (err) {
      if (this.page.generation !== generation || this.page.captureId !== captureId) return;
      this.page = { ...this.page, status: 'failed', error: (err as Error).message };
      this.emit();
      return;
    }
    if (this.page.generation !== generation || this.page.captureId !== captureId) return;
    const limit = this.options.maxInstances ?? 48;
    const ordered = readingOrder(detected.boxes.map((box, i) => ({ box, corners: detected.quads?.[i] })));
    const t = this.now();
    const instances = ordered.slice(0, limit).map(({ box, corners }, i): CardInstance => ({
      id: `${captureId}:${i + 1}`, number: i + 1, box, quad: portraitQuad(corners ?? boxCorners(box)),
      state: 'queued', step: null, revision: 0, queuedAt: t, startedAt: null, settledAt: null, crop: null,
      decision: null, quality: null, reason: null, value: null, valueState: 'idle', ms: null, hits: null, userChoice: null,
    }));
    this.page = { ...this.page, instances, overLimit: Math.max(0, ordered.length - limit), detectMs: detected.ms,
      status: instances.length ? 'processing' : 'settled' };
    this.queue = instances.map((i) => i.id);
    this.emit();
    await this.drain();
  }

  private update(id: string, patch: Partial<CardInstance>) {
    this.page = { ...this.page, instances: this.page.instances.map((i) => (i.id === id ? { ...i, ...patch } : i)) };
  }

  private settleIfDone() {
    if (this.page.status !== 'processing') return;
    if (this.page.instances.every((i) => i.state === 'confirmed' || i.state === 'needs-attention' || i.state === 'skipped' || i.state === 'back')) {
      this.page = { ...this.page, status: 'settled' };
    }
  }

  /** One recognizer, sequential and fair: retries rejoin the back of the queue. */
  private async drain(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      while (this.queue.length) {
        const id = this.queue.shift()!;
        const card = this.instance(id);
        if (!card || card.state !== 'queued' || this.page.captureId === null) continue;
        await this.recognizeOne(card, this.page.generation, this.page.captureId);
      }
    } finally {
      this.running = false;
    }
    this.settleIfDone();
    this.emit();
  }

  private async recognizeOne(card: CardInstance, generation: number, captureId: number) {
    const revision = card.revision;
    const current = () => {
      const c = this.instance(card.id);
      if (this.page.generation !== generation || this.page.captureId !== captureId || !c || c.revision !== revision) throw new Stale();
      return c;
    };
    this.update(card.id, { state: 'recognizing', step: 'rectify', startedAt: this.now(), settledAt: null, reason: null, decision: null });
    this.emit();
    const timeoutMs = this.options.cardTimeoutMs ?? 10000;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const work = this.recognizer.recognize({ captureId, instanceId: card.id, revision, quad: card.quad }, (step, crop) => {
        try {
          current();
        } catch {
          crop?.close();
          return;
        }
        if (step === 'rectified') this.update(card.id, { step: 'embed', crop: crop ?? null });
        if (step === 'embedded') this.update(card.id, { step: 'search' });
        if (step === 'searched') this.update(card.id, { state: 'verifying', step: 'verify' });
        this.emit();
      });
      const timeout = new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('timeout')), timeoutMs); });
      const out = await Promise.race([work, timeout]);
      current();
      const decision = decideMatch(out.hits, this.options.catalog, { preferredLanguage: this.options.preferredLanguage(), languageFromSetting: true,
        thresholds: this.options.thresholds, backThresholds: this.options.backThresholds });
      const matched = decision.kind === 'matched';
      const back = decision.kind === 'back';
      const reason = attentionReason(decision, out.quality);
      this.update(card.id, { state: matched ? 'confirmed' : back ? 'back' : 'needs-attention', step: null, decision, quality: out.quality, reason,
        settledAt: this.now(), ms: out.ms, valueState: matched ? 'loading' : 'idle', hits: out.hits });
      this.emit();
      if (matched) void this.price(card.id, revision, generation);
    } catch (err) {
      if (err instanceof Stale) return;
      const timedOut = (err as Error).message === 'timeout';
      // Bump the revision so a late result for this attempt is ignored.
      this.update(card.id, { state: 'needs-attention', step: null, reason: timedOut ? 'timeout' : 'error', settledAt: this.now(), revision: revision + 1 });
      this.emit();
    } finally {
      clearTimeout(timer);
    }
  }

  /** Prices resolve independently of recognition and are tagged with instance revision. */
  private async price(id: string, revision: number, generation: number) {
    const card = this.instance(id);
    if (!card || card.decision?.kind !== 'matched') return;
    const decision = card.decision;
    try {
      const quotes = await this.options.prices.quotes(decision.plausible.map((c) => c.id));
      const c = this.instance(id);
      if (!c || c.revision !== revision || this.page.generation !== generation || c.decision !== decision) return;
      this.update(id, { value: summarizeValue(decision, quotes, this.options.valueSettings), valueState: 'ready' });
    } catch {
      const c = this.instance(id);
      if (!c || c.revision !== revision || this.page.generation !== generation) return;
      this.update(id, { valueState: 'failed' });
    }
    this.emit();
  }

  /** Retry one card on the same crop. */
  retry(id: string): void {
    const card = this.instance(id);
    if (!card || (card.state !== 'needs-attention' && card.state !== 'skipped' && card.state !== 'back') || this.page.captureId === null) return;
    card.crop?.close();
    this.update(id, { state: 'queued', step: null, revision: card.revision + 1, queuedAt: this.now(), crop: null, reason: null, value: null, valueState: 'idle', userChoice: null, hits: null });
    this.queue.push(id);
    if (this.page.status === 'settled') this.page = { ...this.page, status: 'processing' };
    this.emit();
    void this.drain();
  }

  skip(id: string): void {
    const card = this.instance(id);
    if (!card || card.state === 'confirmed' || card.state === 'back' || card.state === 'recognizing' || card.state === 'verifying') return;
    this.update(id, { state: 'skipped', step: null, revision: card.revision + 1, settledAt: this.now() });
    this.queue = this.queue.filter((q) => q !== id);
    this.settleIfDone();
    this.emit();
  }

  /** The user picks the correct catalog entry (also resolves needs-attention cards). */
  choose(id: string, identity: string, languageConfirmed = false): void {
    const card = this.instance(id);
    if (!card || card.state === 'queued' || card.state === 'recognizing' || card.state === 'verifying') return;
    const decision = userChoiceMatch(this.options.catalog, identity, card.hits ?? [], this.options.preferredLanguage(), languageConfirmed, this.options.thresholds);
    this.update(id, { state: 'confirmed', reason: null, decision, userChoice: { identity, at: this.now() }, valueState: 'loading', value: null });
    this.emit();
    void this.price(id, card.revision, this.page.generation);
  }

  /** The user says none of the candidates is right: not a recognized card, never counted as one. */
  chooseNone(id: string): void {
    const card = this.instance(id);
    if (!card || card.state === 'queued' || card.state === 'recognizing' || card.state === 'verifying') return;
    this.update(id, { state: 'skipped', reason: 'not-found', decision: card.decision, userChoice: { identity: null, at: this.now() }, value: null, valueState: 'idle' });
    this.settleIfDone();
    this.emit();
  }

  /** Re-evaluate preferred-language interpretation without re-running recognition. */
  refreshPreference(): void {
    for (const card of this.page.instances) {
      if (card.decision?.kind !== 'matched' || card.userChoice) continue;
      const decision = decideMatch([card.decision.top, ...(card.decision.competitor ? [card.decision.competitor] : [])], this.options.catalog,
        { preferredLanguage: this.options.preferredLanguage(), languageFromSetting: true, thresholds: card.decision.thresholds });
      if (decision.kind !== 'matched') continue;
      this.update(card.id, { decision, valueState: 'loading' });
      void this.price(card.id, card.revision, this.page.generation);
    }
    this.emit();
  }

  /** Price every matched card again (e.g. after switching the price market). */
  reprice(): void {
    for (const card of this.page.instances) {
      if (card.decision?.kind !== 'matched' || card.state !== 'confirmed') continue;
      this.update(card.id, { valueState: 'loading' });
      void this.price(card.id, card.revision, this.page.generation);
    }
    this.emit();
  }

  /** Explicit page boundary: invalidate all work of the previous generation. */
  nextPage(pageNumber = this.page.pageNumber + 1): void {
    this.reset(pageNumber);
  }

  /** Cancel the active batch (stop, hide, background). Confirmed results of a settled page stay reviewable. */
  cancel(): void {
    if (!this.busy) return;
    this.reset(this.page.pageNumber);
  }

  private reset(pageNumber: number) {
    if (this.page.captureId !== null) this.recognizer.release(this.page.captureId);
    for (const i of this.page.instances) i.crop?.close();
    this.queue = [];
    this.page = this.emptyPage(this.page.generation + 1, pageNumber);
    this.emit();
  }

  /** Idempotent save of this page revision: saving again does not add copies. */
  markSaved(): void {
    this.page = { ...this.page, savedRevision: this.page.revision };
    this.emit();
  }
}

export function pageCounts(page: PageState) {
  const count = (s: CardState) => page.instances.filter((i) => i.state === s).length;
  return {
    found: page.instances.length,
    ready: count('confirmed'),
    checking: count('queued') + count('recognizing') + count('verifying'),
    attention: count('needs-attention'),
    skipped: count('skipped'),
    /** Card backs: physical cards shown face down, not counted as cards to review or value. */
    backs: count('back'),
  };
}
