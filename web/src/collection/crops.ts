// Straightened crops of saved cards, in IndexedDB (localStorage is too small for images).
// `canonical`: the exact encoder-contract crop as PNG, so a re-check embeds the same pixels.
// `display`: a larger JPEG for viewing and as a training example after a correction.
// Per viewer and per browser, like the rest of the pilot collection; absent crops are normal.

export interface SavedCrops {
  canonical: Blob;
  display: Blob | null;
}

const DB = 'tcg-scanner-crops';
const STORE = 'crops';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(mode: IDBTransactionMode, run: (s: IDBObjectStore) => IDBRequest<T> | void): Promise<T | undefined> {
  const db = await open();
  try {
    return await new Promise<T | undefined>((resolve, reject) => {
      const t = db.transaction(STORE, mode);
      const req = run(t.objectStore(STORE));
      t.oncomplete = () => resolve(req ? req.result : undefined);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  } finally {
    db.close();
  }
}

const key = (pageKey: string, number: number) => `${pageKey}#${number}`;

export async function putCrops(pageKey: string, number: number, crops: SavedCrops): Promise<boolean> {
  try { await tx('readwrite', (s) => s.put(crops, key(pageKey, number))); return true; } catch { return false; }
}

export async function getCrops(pageKey: string, number: number): Promise<SavedCrops | null> {
  try { return ((await tx<SavedCrops>('readonly', (s) => s.get(key(pageKey, number)))) ?? null); } catch { return null; }
}

export async function deletePageCrops(pageKey: string): Promise<void> {
  try { await tx('readwrite', (s) => s.delete(IDBKeyRange.bound(`${pageKey}#`, `${pageKey}#￿`))); } catch { /* best effort */ }
}

export async function encodeBitmap(bitmap: ImageBitmap, type: 'image/png' | 'image/jpeg', quality?: number): Promise<Blob> {
  const c = new OffscreenCanvas(bitmap.width, bitmap.height);
  c.getContext('2d')!.drawImage(bitmap, 0, 0);
  return c.convertToBlob({ type, quality });
}
