import { describe, expect, it } from 'vitest';
import { BatchSession, type BatchBackend, type BatchRecord, type BatchStore } from '../src/batch/session';
import { CaptureCycle } from '../src/capture/cycle';
import { SceneChangeGate, type StabilityState } from '../src/capture/stability';
import type { SavedCard, SavedPage } from '../src/collection/store';
import type { SavedCrops } from '../src/collection/crops';
import { Catalog } from '../src/recognition/catalog';
import type { Hit } from '../src/recognition/gallery';
import type { PriceService } from '../src/prices/service';
import type { OrientedBox } from '../src/vision/obb';

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
    putCrops: async (key, n, c) => { crops.set(`${key}#${n}`, c); return true; },
    getCrops: async (key, n) => crops.get(`${key}#${n}`) ?? null,
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
    detect: async (id) => { held = id; return { width: 1000, height: 800, boxes: boxes(), ms: 1 }; },
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
  const luma = (v: number) => new Uint8Array(64 * 48).fill(v);

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
