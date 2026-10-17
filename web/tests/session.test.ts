import { describe, expect, it } from 'vitest';
import { Catalog } from '../src/recognition/catalog';
import type { PriceService } from '../src/prices/service';
import { PageSession, pageCounts, type RecognitionJob, type RecognitionOutput, type Recognizer } from '../src/session/page';
import type { OrientedBox } from '../src/vision/obb';

const ids = ['onepiece:en:OP01-001', 'onepiece:en:OP01-002', 'onepiece:en:OP01-003', 'onepiece:en:OP01-004'];
const catalog = new Catalog(ids.map((id) => ({ id, game: 'onepiece', language: 'en', cardId: id.split(':')[2], name: id, family: id, finish: 'unknown', printing: 'unknown' })));
const prices: PriceService = { description: 'test', quotes: async (list) => new Map(list.map((id) => [id, null])) };
const bitmap = () => ({ width: 1000, height: 800, close() {} }) as unknown as ImageBitmap;
const grid = (n: number): OrientedBox[] => Array.from({ length: n }, (_, i) => ({ cx: 100 + (i % 3) * 200, cy: 100 + Math.floor(i / 3) * 250, w: 220, h: 160, angle: Math.PI / 2, score: 0.9 }));
const tick = () => new Promise((r) => setTimeout(r, 0));

type Plan = (job: RecognitionJob, index: number) => Promise<RecognitionOutput> | 'hang';

function fake(boxes: OrientedBox[], plan: Plan) {
  const calls: RecognitionJob[] = [];
  const released: number[] = [];
  const recognizer: Recognizer = {
    detect: async () => ({ width: 1000, height: 800, boxes, ms: 5 }),
    recognize: (job, onStep) => {
      calls.push(job);
      onStep('rectified');
      onStep('embedded');
      onStep('searched');
      const r = plan(job, calls.length - 1);
      return r === 'hang' ? new Promise(() => {}) : r;
    },
    release: (id) => released.push(id),
  };
  return { recognizer, calls, released };
}

const hit = (i: number, score = 0.8): RecognitionOutput => ({
  hits: [{ identity: ids[i % ids.length], row: 0, score }, { identity: ids[(i + 1) % ids.length], row: 1, score: 0.3 }],
  quality: { shortSide: 200, glare: 0 }, ms: { rectify: 1, embed: 1, search: 1 },
});

describe('page session', () => {
  it('processes nine cards independently, keeping duplicate copies', async () => {
    const { recognizer, calls } = fake(grid(9), async (_j, i) => (i === 4 ? { ...hit(0, 0.2), quality: { shortSide: 200, glare: 0.3 } } : hit(i % 2)));
    const s = new PageSession(recognizer, { catalog, prices, preferredLanguage: () => 'en' });
    await s.capture(bitmap());
    await tick();
    expect(calls).toHaveLength(9);
    expect(pageCounts(s.page)).toMatchObject({ found: 9, ready: 8, attention: 1, checking: 0 });
    expect(s.page.instances[4].reason).toBe('glare');
    expect(s.page.status).toBe('settled');
    // Two pockets with the same identity are two physical copies.
    const same = s.page.instances.filter((i) => i.decision?.kind === 'matched' && i.decision.top.identity === ids[0]);
    expect(same.length).toBeGreaterThan(1);
  });

  it('supports zero and more than nine detections without padding or truncation', async () => {
    const zero = new PageSession(fake([], async () => hit(0)).recognizer, { catalog, prices, preferredLanguage: () => 'en' });
    await zero.capture(bitmap());
    expect(zero.page.status).toBe('settled');
    expect(zero.page.instances).toHaveLength(0);
    const many = new PageSession(fake(grid(14), async (_j, i) => hit(i)).recognizer, { catalog, prices, preferredLanguage: () => 'en' });
    await many.capture(bitmap());
    expect(many.page.instances).toHaveLength(14);
    const limited = new PageSession(fake(grid(14), async (_j, i) => hit(i)).recognizer, { catalog, prices, preferredLanguage: () => 'en', maxInstances: 10 });
    await limited.capture(bitmap());
    expect(limited.page.instances).toHaveLength(10);
    expect(limited.page.overLimit).toBe(4);
  });

  it('times out a stuck card without blocking the others, and ignores its late result', async () => {
    let release: (v: RecognitionOutput) => void = () => {};
    const { recognizer } = fake(grid(3), (_j, i) => (i === 0 ? new Promise((r) => { release = r; }) : Promise.resolve(hit(i))));
    const s = new PageSession(recognizer, { catalog, prices, preferredLanguage: () => 'en', cardTimeoutMs: 20 });
    await s.capture(bitmap());
    expect(s.page.instances[0]).toMatchObject({ state: 'needs-attention', reason: 'timeout' });
    expect(pageCounts(s.page).ready).toBe(2);
    release(hit(0));
    await tick();
    expect(s.page.instances[0].state).toBe('needs-attention');
  });

  it('retries one card at the back of the queue and skips another', async () => {
    let attempt = 0;
    const { recognizer, calls } = fake(grid(2), async (job) => (job.instanceId.endsWith(':1') && attempt++ === 0 ? hit(0, 0.1) : hit(1)));
    const s = new PageSession(recognizer, { catalog, prices, preferredLanguage: () => 'en' });
    await s.capture(bitmap());
    expect(s.page.instances[0].state).toBe('needs-attention');
    s.retry(s.page.instances[0].id);
    await tick();
    await tick();
    expect(s.page.instances[0].state).toBe('confirmed');
    expect(calls.at(-1)?.revision).toBe(1);
    expect(s.page.status).toBe('settled');
  });

  it('drops results from a previous page generation', async () => {
    let finish: (v: RecognitionOutput) => void = () => {};
    const { recognizer, released } = fake(grid(1), () => new Promise((r) => { finish = r; }));
    const s = new PageSession(recognizer, { catalog, prices, preferredLanguage: () => 'en' });
    const done = s.capture(bitmap());
    await tick();
    s.nextPage();
    finish(hit(0));
    await done;
    expect(s.page.instances).toHaveLength(0);
    expect(s.page.pageNumber).toBe(2);
    expect(released).toContain(1);
  });

  it('refuses a second batch while one is active', async () => {
    const { recognizer } = fake(grid(1), () => 'hang');
    const s = new PageSession(recognizer, { catalog, prices, preferredLanguage: () => 'en', cardTimeoutMs: 50 });
    void s.capture(bitmap());
    await tick();
    await expect(s.capture(bitmap())).rejects.toThrow('already active');
  });

  it('saves idempotently per revision', async () => {
    const s = new PageSession(fake(grid(2), async (_j, i) => hit(i)).recognizer, { catalog, prices, preferredLanguage: () => 'en' });
    await s.capture(bitmap());
    s.markSaved();
    s.markSaved();
    expect(s.page.savedRevision).toBe(1);
    expect(s.page.instances).toHaveLength(2);
  });
});

describe('user corrections', () => {
  it('resolves a needs-attention card by choosing a candidate, and "none" never counts as recognized', async () => {
    const s = new PageSession(fake(grid(2), async () => hit(0, 0.1)).recognizer, { catalog, prices, preferredLanguage: () => 'en' });
    await s.capture(bitmap());
    expect(s.page.instances.every((i) => i.state === 'needs-attention')).toBe(true);
    s.choose(s.page.instances[0].id, ids[2]);
    expect(s.page.instances[0]).toMatchObject({ state: 'confirmed', userChoice: { identity: ids[2] } });
    s.chooseNone(s.page.instances[1].id);
    expect(s.page.instances[1].state).toBe('skipped');
    expect(pageCounts(s.page)).toMatchObject({ ready: 1, skipped: 1 });
  });
});
