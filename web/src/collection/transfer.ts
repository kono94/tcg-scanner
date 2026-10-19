// Export and import of the Collection as a JSON file: a backup and a way to move binders to another
// browser without accounts or a server. The file holds binders, pages and saved cards (identities,
// choices, scan-time values, small thumbnails); the full crops stay in this browser's IndexedDB.
import { importRecords, loadPages, loadScanSessions, type ImportResult, type SavedPage, type ScanSession } from './store';

const FORMAT = 'tcg-scanner-collection';
const VERSION = 1;

export interface CollectionFile {
  format: typeof FORMAT;
  version: typeof VERSION;
  exportedAt: string;
  bundle: string;
  sessions: ScanSession[];
  pages: SavedPage[];
}

/** The whole collection, or only the given binders. */
export function exportCollection(bundle: string, sessionIds?: readonly string[], now = new Date()): CollectionFile {
  const keep = (id: string) => !sessionIds || sessionIds.includes(id);
  return {
    format: FORMAT, version: VERSION, exportedAt: now.toISOString(), bundle,
    sessions: loadScanSessions().filter((s) => keep(s.id)),
    pages: loadPages().filter((p) => keep(p.sessionId)).map((p) => ({ ...p, cards: p.cards.map(({ current: _, ...c }) => c) })),
  };
}

export function exportFileName(now = new Date(), binder?: string): string {
  const day = now.toISOString().slice(0, 10);
  const slug = binder?.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  return `tcg-collection-${slug ? `${slug}-` : ''}${day}.json`;
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isString = (v: unknown): v is string => typeof v === 'string';

function validPage(p: unknown): p is SavedPage {
  return isObject(p) && isString(p.key) && isString(p.sessionId) && typeof p.pageNumber === 'number' && isString(p.savedAt) && Array.isArray(p.cards)
    && p.cards.every((c) => isObject(c) && typeof c.number === 'number' && isString(c.state) && (c.identity === null || isString(c.identity)));
}

function validSession(s: unknown): s is ScanSession {
  return isObject(s) && isString(s.id) && isString(s.name) && isString(s.startedAt);
}

/** Parse and check an exported file; throws an Error with a message for the user. */
export function parseCollection(text: string): CollectionFile {
  let data: unknown;
  try { data = JSON.parse(text); } catch { throw new Error('This file is not a collection export.'); }
  if (!isObject(data) || data.format !== FORMAT) throw new Error('This file is not a collection export.');
  if (data.version !== VERSION) throw new Error('This export comes from a newer scanner version.');
  if (!Array.isArray(data.pages) || !data.pages.every(validPage) || !Array.isArray(data.sessions) || !data.sessions.every(validSession)) {
    throw new Error('This collection export is damaged.');
  }
  return data as unknown as CollectionFile;
}

/** Merge a parsed export into this browser; throws when storage is full. */
export function importCollection(file: CollectionFile): ImportResult {
  const result = importRecords(file.sessions, file.pages);
  if (!result) throw new Error('Could not import: browser storage is full or disabled.');
  return result;
}
