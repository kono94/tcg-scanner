import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { activeScanSession, loadPages, loadScanSessions, startScanSession, writePage, type SavedCard, type SavedPage } from '../src/collection/store';
import { exportCollection, exportFileName, importCollection, parseCollection } from '../src/collection/transfer';

let data: Map<string, string>;
let failWrites = false;

beforeEach(() => {
  data = new Map();
  failWrites = false;
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => { if (failWrites) throw new Error('quota'); data.set(key, value); },
  });
});
afterEach(() => vi.unstubAllGlobals());

const card = (number: number, extra: Partial<SavedCard> = {}): SavedCard => ({
  number, identity: `pokemon:en:c${number}`, name: `C${number}`, cardId: `c${number}`, state: 'confirmed', languageAssumed: false, checkedByUser: false,
  value: { currency: 'EUR', preferred: 2, low: 2, high: 2, cue: false }, thumb: 'data:image/jpeg;base64,AAAA', hasCrop: true, ...extra,
});
const page = (key: string, sessionId: string, pageNumber: number, cards = [card(1), card(2)]): SavedPage =>
  ({ key, sessionId, pageNumber, revision: 1, savedAt: '2026-10-19T10:00:00Z', partial: false, cards });

/** Export from one fresh browser, then start an empty one. */
function exported(setup: () => void, sessionIds?: string[]) {
  setup();
  const text = JSON.stringify(exportCollection('bundle-1', sessionIds, new Date('2026-10-19T12:00:00Z')));
  data = new Map();
  return text;
}

describe('collection export and import', () => {
  it('restores binders, pages, cards and thumbnails in an empty browser, without crops', () => {
    let id = '';
    const text = exported(() => {
      id = startScanSession('Red Binder').id;
      writePage(page('p-1', id, 1));
      writePage(page('p-2', id, 2, [card(1)]));
    });
    const result = importCollection(parseCollection(text));
    expect(result).toEqual({ addedPages: 2, skippedPages: 0, addedSessions: 1 });
    expect(loadScanSessions()).toMatchObject([{ id, name: 'Red Binder', nextCapture: 3 }]);
    expect(activeScanSession()).toBeNull();
    const pages = loadPages();
    expect(pages.map((p) => p.key)).toEqual(['p-1', 'p-2']);
    expect(pages[0].cards[0]).toMatchObject({ identity: 'pokemon:en:c1', thumb: 'data:image/jpeg;base64,AAAA', hasCrop: false, value: { preferred: 2 } });
  });

  it('never doubles pages when the same file is imported twice', () => {
    const text = exported(() => writePage(page('p-1', startScanSession('A').id, 1)));
    importCollection(parseCollection(text));
    expect(importCollection(parseCollection(text))).toEqual({ addedPages: 0, skippedPages: 1, addedSessions: 0 });
    expect(loadPages()).toHaveLength(1);
    expect(loadScanSessions()).toHaveLength(1);
  });

  it('keeps local binders and their names next to imported ones', () => {
    let shared = '';
    const text = exported(() => {
      shared = startScanSession('Imported name').id;
      writePage(page('p-old', shared, 1));
      writePage(page('p-new', shared, 4));
    });
    const local = startScanSession('Local').id;
    writePage(page('p-local', local, 1));
    // The shared binder already exists here (synced earlier) under another name with one page.
    writePage(page('p-old', shared, 1));
    data.set('tcg-scanner-sessions-v2', JSON.stringify({ activeId: local, sessions: [...loadScanSessions(), { id: shared, name: 'Mine', startedAt: '2026-10-01T00:00:00Z', nextCapture: 2 }] }));
    expect(importCollection(parseCollection(text))).toEqual({ addedPages: 1, skippedPages: 1, addedSessions: 0 });
    expect(loadScanSessions().map((s) => [s.name, s.nextCapture])).toEqual([['Local', 2], ['Mine', 5]]);
    expect(activeScanSession()?.id).toBe(local);
    expect(loadPages().map((p) => p.key).sort()).toEqual(['p-local', 'p-new', 'p-old']);
  });

  it('exports one binder only', () => {
    writePage(page('a-1', startScanSession('A').id, 1));
    const b = startScanSession('B').id;
    writePage(page('b-1', b, 1));
    const only = exportCollection('bundle-1', [b]);
    expect(only.sessions.map((s) => s.name)).toEqual(['B']);
    expect(only.pages.map((p) => p.key)).toEqual(['b-1']);
  });

  it('turns cards still waiting for the batch queue into cards that need a check', () => {
    const text = exported(() => writePage(page('p-1', startScanSession('A').id, 1, [card(1, { state: 'queued', identity: null, queued: 123 })])));
    importCollection(parseCollection(text));
    const [c] = loadPages()[0].cards;
    expect(c).toMatchObject({ state: 'needs-attention', reason: 'error', hasCrop: false });
    expect(c.queued).toBeUndefined();
  });

  it('rejects other files with a message', () => {
    expect(() => parseCollection('not json')).toThrow('not a collection export');
    expect(() => parseCollection(JSON.stringify({ format: 'other' }))).toThrow('not a collection export');
    expect(() => parseCollection(JSON.stringify({ format: 'tcg-scanner-collection', version: 2 }))).toThrow('newer scanner version');
    expect(() => parseCollection(JSON.stringify({ format: 'tcg-scanner-collection', version: 1, sessions: [], pages: [{ key: 1 }] }))).toThrow('damaged');
  });

  it('changes nothing when storage is full', () => {
    const text = exported(() => writePage(page('p-1', startScanSession('A').id, 1)));
    failWrites = true;
    expect(() => importCollection(parseCollection(text))).toThrow('storage is full');
    failWrites = false;
    expect(loadPages()).toEqual([]);
  });

  it('names the file by date and binder', () => {
    const at = new Date('2026-10-19T12:00:00Z');
    expect(exportFileName(at)).toBe('tcg-collection-2026-10-19.json');
    expect(exportFileName(at, 'Red Binder #2')).toBe('tcg-collection-red-binder-2-2026-10-19.json');
  });
});
