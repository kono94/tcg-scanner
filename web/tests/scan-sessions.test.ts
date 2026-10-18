import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { activeScanSession, currentSession, deleteScanSession, finishScanSession, loadPages, loadScanSessions, nextCaptureNumber, renameScanSession, resumeScanSession, savePage, startScanSession, writePage, type SavedPage } from '../src/collection/store';
import { deletePageCrops } from '../src/collection/crops';
import { Catalog } from '../src/recognition/catalog';
import type { PageState } from '../src/session/page';

vi.mock('../src/collection/crops', async (original) => ({ ...await original<typeof import('../src/collection/crops')>(), deletePageCrops: vi.fn(async () => {}) }));

beforeEach(() => {
  vi.clearAllMocks();
  const data = new Map<string, string>();
  vi.stubGlobal('localStorage', { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => data.set(key, value) });
});
afterEach(() => vi.unstubAllGlobals());

const saved = (key: string, sessionId: string, pageNumber: number): SavedPage =>
  ({ key, sessionId, pageNumber, revision: 1, savedAt: '2026-10-17T16:00:00Z', cards: [], partial: false });

describe('persistent scan sessions', () => {
  it('deletes a whole collection and all crop keys while preserving other sessions', async () => {
    const red = startScanSession('Red Binder');
    writePage(saved('red-1', red.id, 1));
    writePage(saved('red-2', red.id, 2));
    const blue = startScanSession('Blue Binder');
    writePage(saved('blue-1', blue.id, 1));
    resumeScanSession(red.id);
    expect(await deleteScanSession(red.id)).toBe(true);
    expect(activeScanSession()).toBeNull();
    expect(loadScanSessions().map((s) => s.id)).toEqual([blue.id]);
    expect(loadPages().map((p) => p.key)).toEqual(['blue-1']);
    expect(deletePageCrops).toHaveBeenCalledWith('red-1');
    expect(deletePageCrops).toHaveBeenCalledWith('red-2');
    expect(deletePageCrops).not.toHaveBeenCalledWith('blue-1');
  });

  it('does not delete crop data if collection metadata cannot be updated', async () => {
    const red = startScanSession('Red Binder');
    writePage(saved('red', red.id, 1));
    const set = localStorage.setItem;
    localStorage.setItem = (key, value) => { if (key === 'tcg-scanner-sessions-v2') throw new Error('quota'); set(key, value); };
    expect(await deleteScanSession(red.id)).toBe(false);
    expect(loadPages()).toHaveLength(1);
    expect(deletePageCrops).not.toHaveBeenCalled();
  });
  it('continues across days, retains a user name, and explicitly finishes/resumes', () => {
    const red = startScanSession('Red Binder', Date.UTC(2026, 9, 17));
    expect(currentSession(Date.UTC(2026, 10, 25))).toBe(red.id);
    expect(renameScanSession(red.id, 'Red Pokémon Binder')).toBe(true);
    expect(activeScanSession()?.name).toBe('Red Pokémon Binder');
    expect(finishScanSession()).toBe(true);
    expect(activeScanSession()).toBeNull();
    expect(loadScanSessions()[0].finishedAt).toBeTruthy();
    expect(resumeScanSession(red.id)).toBe(true);
    expect(activeScanSession()).toMatchObject({ id: red.id, name: 'Red Pokémon Binder' });
    expect(activeScanSession()?.finishedAt).toBeUndefined();
  });

  it('shares numbering between workflows and never reuses deleted capture numbers', () => {
    const red = startScanSession('Red Binder');
    writePage(saved('single', red.id, 1));
    expect(nextCaptureNumber()).toBe(2);
    writePage({ ...saved('batch', red.id, nextCaptureNumber()), batchId: 'b' });
    expect(nextCaptureNumber()).toBe(3);
    localStorage.setItem('tcg-scanner-collection-v1', '[]');
    expect(nextCaptureNumber()).toBe(3);
    const blue = startScanSession('Blue Binder');
    expect(blue.id).not.toBe(red.id);
    expect(nextCaptureNumber()).toBe(1);
    resumeScanSession(red.id);
    expect(nextCaptureNumber()).toBe(3);
  });

  it('migrates old session groups without changing saved keys, duplicate numbering, or quantities', () => {
    const pages = [saved('a', 'legacy', 1), saved('b', 'legacy', 1), saved('c', 'legacy', 8)];
    localStorage.setItem('tcg-scanner-collection-v1', JSON.stringify(pages));
    localStorage.setItem('tcg-scanner-session-v1', JSON.stringify({ id: 'legacy', last: Date.now() }));
    expect(activeScanSession()?.id).toBe('legacy');
    expect(nextCaptureNumber()).toBe(9);
    expect(loadPages()).toEqual(pages);
  });

  it('does not reactivate a legacy session that had already expired under the old three-hour rule', () => {
    localStorage.setItem('tcg-scanner-collection-v1', JSON.stringify([saved('a', 'legacy', 4)]));
    localStorage.setItem('tcg-scanner-session-v1', JSON.stringify({ id: 'legacy', last: Date.now() - 4 * 60 * 60 * 1000 }));
    expect(activeScanSession()).toBeNull();
    expect(loadScanSessions().map((x) => x.id)).toEqual(['legacy']);
  });

  it('drops an unnamed session that never saved a capture, but keeps named or used ones', () => {
    const unused = startScanSession();
    const named = startScanSession('Red Binder');
    expect(loadScanSessions().map((x) => x.id)).toEqual([named.id]);
    expect(finishScanSession()).toBe(true);
    expect(loadScanSessions().map((x) => x.id)).toEqual([named.id]);
    const used = startScanSession();
    writePage(saved('p', used.id, 1));
    startScanSession();
    expect(loadScanSessions().map((x) => x.id)).toEqual([named.id, used.id, activeScanSession()!.id]);
    expect(loadScanSessions().some((x) => x.id === unused.id)).toBe(false);
  });

  it('binds repeat saves to the original session and number, even after another session starts', async () => {
    const red = startScanSession('Red Binder');
    const page = { generation: 10, pageNumber: 1, revision: 1, instances: [], overLimit: 0 } as unknown as PageState;
    await savePage(page, new Catalog([]));
    startScanSession('Blue Binder');
    await savePage({ ...page, revision: 2 }, new Catalog([]));
    expect(loadPages()).toHaveLength(1);
    expect(loadPages()[0]).toMatchObject({ sessionId: red.id, pageNumber: 1, revision: 2 });
    expect(nextCaptureNumber()).toBe(1);
  });

  it('reports failed session persistence instead of silently replacing an active session', () => {
    const red = startScanSession('Red Binder');
    localStorage.setItem = () => { throw new Error('quota'); };
    expect(() => startScanSession('Blue Binder')).toThrow(/storage/);
    expect(activeScanSession()?.id).toBe(red.id);
  });
});
