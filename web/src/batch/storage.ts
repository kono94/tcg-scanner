// BatchStore on the Collection's own storage: pages in localStorage, crops in IndexedDB.
import { currentSession, loadPages, patchSavedCard, writePage } from '../collection/store';
import { getCrops, putCrops } from '../collection/crops';
import type { BatchRecord, BatchStore } from './session';

const BATCH = 'tcg-scanner-batch-v1';

function blobToDataUrl(blob: Blob): Promise<string | null> {
  return new Promise((resolve) => {
    const r = new FileReader();
    r.onload = () => resolve(typeof r.result === 'string' ? r.result : null);
    r.onerror = () => resolve(null);
    r.readAsDataURL(blob);
  });
}

export const browserBatchStore: BatchStore = {
  loadPages,
  writePage,
  patchCard: patchSavedCard,
  putCrops,
  getCrops,
  thumbUrl: blobToDataUrl,
  loadBatch() {
    try { return JSON.parse(localStorage.getItem(BATCH) ?? 'null') as BatchRecord | null; } catch { return null; }
  },
  saveBatch(batch) {
    try {
      if (batch) localStorage.setItem(BATCH, JSON.stringify(batch));
      else localStorage.removeItem(BATCH);
    } catch { /* storage unavailable */ }
  },
  sessionId: () => currentSession(),
};
