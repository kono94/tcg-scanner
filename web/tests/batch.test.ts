import { describe, expect, it } from 'vitest';
import { BatchSession, type BatchBackend, type BatchRecord, type BatchStore } from '../src/batch/session';
import { CaptureCycle } from '../src/capture/cycle';
import { SceneChangeGate, type StabilityState } from '../src/capture/stability';
import type { SavedCard, SavedPage } from '../src/collection/store';
import type { SavedCrops } from '../src/collection/crops';
import { Catalog } from '../src/recognition/catalog';
import type { Hit } from '../src/recognition/gallery';
import type { PriceService } from '../src/prices/service';
import { boxCorners, type OrientedBox } from '../src/vision/quad';
import { appearanceFingerprint } from '../src/batch/duplicates';

const ids = ['onepiece:en:OP01-001', 'onepiece:en:OP01-002', 'onepiece:en:OP01-003'];
const catalog = new Catalog(ids.map((id) => ({ id, game: 'onepiece', language: 'en', cardId: id.split(':')[2], name: id, family: id, finish: 'unknown', printing: 'unknown' })));
const prices: PriceService = { description: 'test', quotes: async (list) => new Map(list.map((id) => [id, null])) };
const bitmap = () => ({ width: 1000, height: 800, close() {} }) as unknown as ImageBitmap;
const grid = (n: number): OrientedBox[] => Array.from({ length: n }, (_, i) => ({ cx: 100 + (i % 3) * 200, cy: 100 + Math.floor(i / 3) * 250, w: 220, h: 160, angle: Math.PI / 2, score: 0.9 }));
const settle = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 0)); };
const blob = (tag: string) => ({ tag }) as unknown as Blob;

/** In-memory store with the same semantics as the localStorage/IndexedDB one. */
function memoryStore() {
  let pages: SavedPage[] = [];
  let batch: BatchRecord | null = null;
  const crops = new Map<string, SavedCrops>();
  const store: BatchStore = {
    loadPages: () => JSON.parse(JSON.stringify(pages)),
    writePage: (p) => { pages = [...pages.filter((x) => x.key !== p.key), JSON.parse(JSON.stringify(p))]; return true; },
    patchCard: (key, number, patch) => {
      const page = pages.find((p) => p.key === key);
      const card = page?.cards.find((c) => c.number === number);
      if (!page || !card) return false;
      Object.assign(card, patch);
      for (const [k, v] of Object.entries(patch)) if (v === undefined) delete (card as unknown as Record<string, unknown>)[k];
      page.partial = page.cards.some((c) => c.state !== 'confirmed');
      return true;
    },
    removeCard: (key, number) => {
      const page = pages.find((p) => p.key === key);
      if (!page) return;
      page.cards = page.cards.filter((c) => c.number !== number);
      if (!page.cards.length) pages = pages.filter((p) => p !== page);
    },
    putCrops: async (key, n, c) => { crops.set(`${key}#${n}`, c); return true; },
    getCrops: async (key, n) => crops.get(`${key}#${n}`) ?? null,
    removeCrops: async (key) => { for (const k of crops.keys()) if (k.startsWith(`${key}#`)) crops.delete(k); },
    thumbUrl: async () => 'data:thumb',
    loadBatch: () => (batch ? JSON.parse(JSON.stringify(batch)) : null),
    saveBatch: (b) => { batch = b; },
    sessionId: () => 's1',
  };
  return { store, crops, pages: () => pages, deletePage: (key: string) => { pages = pages.filter((p) => p.key !== key); } };
}

function backend(boxes: () => OrientedBox[], hitsFor: (b: Blob) => Hit[] | Promise<Hit[]>) {
  const released: number[] = [];
  const searches: Blob[] = [];
  let failure: string | null = null;
  let held: number | null = null;
  const b: BatchBackend = {
    detect: async (id) => { held = id; const b = boxes(); return { width: 1000, height: 800, boxes: b, quads: b.map(boxCorners), ms: 1 }; },
    extract: async (id, quad) => {
      if (held !== id) throw new Error('Capture was released');
      return { canonical: blob(`c${Math.round(quad[0][0])}`), display: blob('d'), thumb: blob('t'), quality: { shortSide: 200, glare: 0 } };
    },
    release: (id) => { released.push(id); if (held === id) held = null; },
    search: async (c) => { searches.push(c); return hitsFor(c); },
    failure: () => failure,
  };
  return { b, released, searches, fail: (m: string) => { failure = m; } };
}

const good = (i: number): Hit[] => [{ identity: ids[i % 3], row: 0, score: 0.8 }, { identity: ids[(i + 1) % 3], row: 1, score: 0.3 }];
const options = { catalog, prices, preferredLanguage: () => 'en', bundleVersion: 'v1' };

describe('batch session', () => {
  it('quarantines a probable duplicate across reload, and only an explicit override adds quantities', async () => {
    const mem = memoryStore(), be = backend(() => grid(2), () => good(0));
    const extract = be.b.extract;
    const appearance = appearanceFingerprint({ width: 6, height: 8, data: new Uint8Array(6 * 8 * 3).fill(100) });
    be.b.extract = async (...args) => ({ ...await extract(...args), appearance });
    const s = new BatchSession(be.b, mem.store, options);
    await s.capture(bitmap());
    const duplicate = await s.capture(bitmap());
    expect(duplicate).toMatchObject({ duplicate: 1, pageNumber: null, cards: 0 });
    expect(mem.pages()).toHaveLength(1);
    expect(s.status().cards).toBe(2);
    // The held repeat is never discarded implicitly: no new capture and no finish until Keep/Dismiss.
    expect(s.canCapture()).toBe(false);
    await expect(s.capture(bitmap())).rejects.toThrow(/possible repeat/);
    expect(s.finish()).toBe(false);
    expect(mem.crops.size).toBe(4);
    const restored = new BatchSession(be.b, mem.store, options);
    expect(restored.status().duplicate).toMatchObject({ matchedPageNumber: 1, cards: 2 });
    expect(restored.keepDuplicate()).toBe(true);
    expect(restored.keepDuplicate()).toBe(false);
    expect(mem.pages()).toHaveLength(2);
    expect(mem.pages()[1]).toMatchObject({ pageNumber: 2, duplicateOverride: { pageKey: mem.pages()[0].key } });
    await settle();
  });

  it('skips an automatic repeat without holding it or keeping its crops', async () => {
    const mem = memoryStore(), be = backend(() => grid(2), () => good(0));
    const extract = be.b.extract;
    const appearance = appearanceFingerprint({ width: 6, height: 8, data: new Uint8Array(6 * 8 * 3).fill(100) });
    be.b.extract = async (...args) => ({ ...await extract(...args), appearance });
    const s = new BatchSession(be.b, mem.store, options);
    s.setPaused(true);
    await s.capture(bitmap());
    const repeat = await s.capture(bitmap(), undefined, { holdRepeat: false });
    expect(repeat).toMatchObject({ duplicate: 1, skipped: true, pageNumber: null, cards: 0 });
    expect(mem.pages()).toHaveLength(1);
    expect(mem.crops.size).toBe(2);
    expect(s.status().duplicate).toBeNull();
    expect(s.canCapture()).toBe(true);
  });

  it('discards unkept duplicate crops and prunes deleted queue work without resurrecting cards', async () => {
    const mem = memoryStore(), be = backend(() => grid(1), () => good(0));
    const extract = be.b.extract;
    const appearance = 'a1:' + '80'.repeat(144);
    be.b.extract = async (...args) => ({ ...await extract(...args), appearance });
    const s = new BatchSession(be.b, mem.store, options);
    s.setPaused(true);
    await s.capture(bitmap());
    await s.capture(bitmap());
    expect(mem.crops.size).toBe(2);
    await s.discardDuplicate();
    expect(mem.crops.size).toBe(1);
    expect(s.canCapture()).toBe(true);
    expect(mem.pages()).toHaveLength(1);
    mem.deletePage(mem.pages()[0].key);
    s.pruneDeleted();
    expect(s.status().waiting).toBe(0);
    s.setPaused(false);
    await settle();
    expect(mem.pages()).toHaveLength(0);
    expect(be.searches).toHaveLength(0);
  });
  it('does not retire a batch while an admitted capture is still being stored', async () => {
    const mem = memoryStore(), be = backend(() => grid(1), () => good(0));
    let resume!: () => void;
    const barrier = new Promise<void>((resolve) => { resume = resolve; });
    const detect = be.b.detect;
    be.b.detect = async (id, bitmap) => { await barrier; return detect(id, bitmap); };
    const s = new BatchSession(be.b, mem.store, options);
    const capture = s.capture(bitmap());
    expect(s.finish()).toBe(false);
    resume();
    await capture;
    expect(mem.pages()).toHaveLength(1);
    expect(s.finish()).toBe(true);
    expect(s.status().active).toBeNull();
    await settle();
    expect(s.status().active).toBeNull();
  });
  it('uses the shared capture counter across batch boundaries and binds session ownership before async detection', async () => {
    const mem = memoryStore();
    let session = 's1', next = 4;
    mem.store.sessionId = () => session;
    mem.store.nextCaptureNumber = () => next;
    const write = mem.store.writePage;
    mem.store.writePage = (page) => { next = page.pageNumber + 1; return write(page); };
    const s = new BatchSession(backend(() => grid(1), () => good(0)).b, mem.store, options);
    const first = s.capture(bitmap());
    session = 's2'; // A later session change cannot steal an already acquired capture.
    await first;
    expect(mem.pages()[0]).toMatchObject({ sessionId: 's1', pageNumber: 4 });
    session = 's1';
    s.finish();
    expect((await s.capture(bitmap())).pageNumber).toBe(5);
    next = 1; session = 's2';
    expect((await s.capture(bitmap())).pageNumber).toBe(1);
    expect(mem.pages()[2].sessionId).toBe('s2');
    expect(mem.pages()[2].batchId).not.toBe(mem.pages()[1].batchId);
    await settle();
  });
  it('saves each page right away, releases the capture, and checks the cards in the background', async () => {
    const mem = memoryStore();
    let n = 0;
    const be = backend(() => grid(9), () => good(n++));
    const s = new BatchSession(be.b, mem.store, options);
    const r1 = await s.capture(bitmap());
    expect(r1).toMatchObject({ pageNumber: 1, cards: 9 });
    expect(be.released).toHaveLength(1); // released before any recognition result
    expect(mem.pages()[0].cards.every((c) => c.hasCrop)).toBe(true);
    const r2 = await s.capture(bitmap());
    expect(r2.pageNumber).toBe(2);
    await settle();
    const pages = mem.pages();
    expect(pages).toHaveLength(2);
    expect(pages.flatMap((p) => p.cards).every((c) => c.state === 'confirmed' && c.identity)).toBe(true);
    expect(be.searches).toHaveLength(18);
    expect(s.status()).toMatchObject({ pages: 2, cards: 18, waiting: 0, needsUser: 0 });
    expect(pages.every((p) => !p.partial && p.batchId)).toBe(true);
  });

  it('keeps card count variable: zero cards saves no page, many cards are not truncated to nine', async () => {
    const mem = memoryStore();
    let boxes = grid(0);
    const s = new BatchSession(backend(() => boxes, () => good(0)).b, mem.store, options);
    expect((await s.capture(bitmap())).cards).toBe(0);
    expect(mem.pages()).toHaveLength(0);
    boxes = grid(14);
    expect((await s.capture(bitmap())).cards).toBe(14);
    expect(mem.pages()[0].pageNumber).toBe(1);
  });

  it('marks unclear cards as needing the user, with their closest matches', async () => {
    const mem = memoryStore();
    const s = new BatchSession(backend(() => grid(2), () => [{ identity: ids[0], row: 0, score: 0.5 }, { identity: ids[1], row: 1, score: 0.49 }]).b, mem.store, options);
    await s.capture(bitmap());
    await settle();
    const card = mem.pages()[0].cards[0];
    expect(card.state).toBe('needs-attention');
    expect(card.reason).toMatch(/not-found|match-unclear/);
    expect(card.hits?.length).toBeGreaterThan(0);
    expect(s.status().needsUser).toBe(2);
  });

  it('resumes queued cards after a restart (new session on the same store)', async () => {
    const mem = memoryStore();
    const hold: Array<(h: Hit[]) => void> = [];
    const first = backend(() => grid(3), () => new Promise<Hit[]>((res) => hold.push(res)));
    const s1 = new BatchSession(first.b, mem.store, options);
    await s1.capture(bitmap());
    await settle();
    expect(s1.status().waiting).toBe(3);
    // The tab is killed here: s1 is gone, its job never finishes.
    const second = backend(() => [], () => good(1));
    const s2 = new BatchSession(second.b, mem.store, options);
    expect(s2.status().waiting).toBe(3);
    s2.kick();
    await settle();
    expect(mem.pages()[0].cards.every((c: SavedCard) => c.state === 'confirmed')).toBe(true);
    // A late result from the killed session cannot overwrite anything (token no longer queued).
    hold[0]?.(good(2));
    await settle();
    expect(mem.pages()[0].cards[0].identity).toBe(ids[1]);
  });

  it('drops results for cards the user decided or deleted meanwhile', async () => {
    const mem = memoryStore();
    const hold: Array<(h: Hit[]) => void> = [];
    const s = new BatchSession(backend(() => grid(2), () => new Promise<Hit[]>((res) => hold.push(res))).b, mem.store, options);
    await s.capture(bitmap());
    await settle();
    mem.store.patchCard(mem.pages()[0].key, 1, { state: 'not-in-catalog', checkedByUser: true, queued: undefined });
    hold[0](good(0));
    await settle();
    expect(mem.pages()[0].cards[0].state).toBe('not-in-catalog');
    mem.deletePage(mem.pages()[0].key);
    hold[1]?.(good(0));
    await settle();
    expect(mem.pages()).toHaveLength(0);
  });

  it('bounds the backlog: admission is refused before capture until enough cards are checked', async () => {
    const mem = memoryStore();
    const hold: Array<(h: Hit[]) => void> = [];
    const s = new BatchSession(backend(() => grid(9), () => new Promise<Hit[]>((res) => hold.push(res))).b, mem.store, { ...options, backlogLimit: 18, resumeAt: 9 });
    await s.capture(bitmap());
    expect(s.canCapture()).toBe(true);
    await s.capture(bitmap());
    expect(s.canCapture()).toBe(false);
    expect(s.status().full).toBe(true);
    await expect(s.capture(bitmap())).rejects.toThrow(/backlog/);
    for (let i = 0; i < 9; i++) { hold[i](good(i)); await settle(); }
    expect(s.status().waiting).toBe(9);
    expect(s.canCapture()).toBe(true);
  });

  it('pauses while hidden and stalls on a worker crash without losing queued cards', async () => {
    const mem = memoryStore();
    const be = backend(() => grid(2), () => good(0));
    const s = new BatchSession(be.b, mem.store, options);
    s.setPaused(true);
    await s.capture(bitmap());
    await settle();
    expect(be.searches).toHaveLength(0);
    be.fail('Inference worker stopped');
    be.b.search = async () => { throw new Error('Inference worker stopped'); };
    s.setPaused(false);
    await settle();
    expect(s.status().stalled).toMatch(/worker/);
    expect(s.canCapture()).toBe(false);
    expect(mem.pages()[0].cards.every((c) => c.state === 'queued')).toBe(true);
  });
});

describe('batch auto capture: page-turn gate', () => {
  const steady: StabilityState = { steady: true, steadyMs: 2000, motion: 1, sharp: true, reason: 'steady' };
  const moving: StabilityState = { steady: false, steadyMs: 0, motion: 30, sharp: true, reason: 'motion' };
  const luma = (v: number) => Uint8Array.from({ length: 64 * 48 }, (_, i) => i % 2 ? v : 200 - v);

  it('does not latch a temporary scene change or mistake exposure for a new page', () => {
    const gate = new SceneChangeGate(12);
    gate.arm(luma(100), true);
    expect(gate.update(luma(140), steady)).toBe(true);
    expect(gate.update(luma(100), steady)).toBe(false);
    gate.arm(luma(60), true);
    expect(gate.update(luma(60).map((v) => v + 20), steady)).toBe(false);
  });

  it('recognizes a page turned during suspended delivery, but never recaptures the unchanged steady view', () => {
    const gate = new SceneChangeGate(12);
    gate.arm(luma(100), true);
    expect(gate.update(luma(102), steady)).toBe(false);
    expect(gate.update(luma(140), moving)).toBe(false);
    expect(gate.update(luma(140), steady)).toBe(true);
    gate.arm(luma(100), true);
    // No motion samples are available during processing; the changed image is still evidence.
    expect(gate.update(luma(140), steady)).toBe(true);
  });

  it('does not capture the same page again; captures once the scene moved and changed', () => {
    const gate = new SceneChangeGate(12);
    const cycle = new CaptureCycle({ mode: 'auto', dwellMs: 800, cooldownMs: 3000 });
    cycle.beginBatch();
    cycle.settle(0);
    cycle.rearm();
    gate.arm(luma(100));
    cycle.requireSceneChange();
    const same = cycle.tick(10, steady, gate.update(luma(102), steady));
    expect(same.capture).toBe(false);
    expect(same.view.awaitingChange).toBe(true);
    // Moved, but came back to the same view: still waiting.
    cycle.tick(20, moving, gate.update(luma(160), moving));
    expect(cycle.tick(30, steady, gate.update(luma(101), steady)).capture).toBe(false);
    // Next page: different content, steady.
    expect(cycle.tick(40, steady, gate.update(luma(140), steady)).capture).toBe(true);
  });
});
