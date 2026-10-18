// BatchStore on the Collection's own storage: pages in localStorage, crops in IndexedDB.
import { currentSession, deleteSavedCard, loadPages, nextCaptureNumber, patchSavedCard, writePage } from '../collection/store';
import { deletePageCrops, getCrops, putCrops } from '../collection/crops';
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
  removeCard: deleteSavedCard,
  putCrops,
  getCrops,
  removeCrops: deletePageCrops,
  thumbUrl: blobToDataUrl,
  loadBatch() {
    try {
      const batch = JSON.parse(localStorage.getItem(BATCH) ?? 'null') as BatchRecord | null;
      if (batch && !batch.sessionId) batch.sessionId = loadPages().find((p) => batch.pageKeys.includes(p.key))?.sessionId;
      return batch;
    } catch { return null; }
  },
  saveBatch(batch) {
    try {
      if (batch) localStorage.setItem(BATCH, JSON.stringify(batch));
      else localStorage.removeItem(BATCH);
      return true;
    } catch { return false; }
  },
  sessionId: () => currentSession(),
  nextCaptureNumber,
};
