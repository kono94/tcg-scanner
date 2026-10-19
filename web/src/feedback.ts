// User decisions as labeled training examples. In dev they go to the dev server (/__feedback),
// which stores the crop and metadata under datasets/sources/phone/feedback/. A deployed app needs a real endpoint.
import type { CardInstance } from './session/page';

export interface FeedbackContext {
  bundleVersion: string;
  pageNumber: number;
  captureSize: { width: number; height: number } | null;
  /** Test video instead of a camera. */
  cameraMock: boolean;
}

/** Automated test runs mark themselves so their (arbitrary) choices never become training labels. */
export function isTestRun(): boolean {
  try { return localStorage.getItem('tcg-test-run') === '1'; } catch { return false; }
}

async function jpeg(bitmap: ImageBitmap): Promise<string> {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

/** decision: a catalog identity the user picked, or null for "none of these" (not in the catalog). */
export async function sendFeedback(card: CardInstance, crop: ImageBitmap | null, decision: { identity: string | null; languageConfirmed: boolean },
  ctx: FeedbackContext): Promise<void> {
  if (!import.meta.env.DEV || !crop) return;
  try {
    const body = {
      schema_version: 1,
      kind: decision.identity ? 'user_selected_identity' : 'user_rejected_all_candidates',
      label_origin: 'user_decision_in_scanner',
      identity: decision.identity,
      language_confirmed: decision.languageConfirmed,
      machine: {
        state: card.state, reason: card.reason,
        top: card.hits?.slice(0, 10).map((h) => ({ identity: h.identity, score: +h.score.toFixed(4) })) ?? [],
      },
      geometry: { quad_capture_px: card.quad, capture_size: ctx.captureSize, crop_size: [crop.width, crop.height] },
      bundle_version: ctx.bundleVersion,
      camera: ctx.cameraMock ? 'test_video' : 'camera',
      test_run: isTestRun(),
      page: ctx.pageNumber,
      card_number: card.number,
      created_at: new Date().toISOString(),
      crop_jpeg_base64: await jpeg(crop),
    };
    await fetch('/__feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  } catch { /* feedback must never break scanning */ }
}

/** A correction made later in the Collection, from the saved crop (re-checked with the current bundle). */
export async function sendCollectionFeedback(saved: { identity: string | null; number: number; pageNumber: number }, crop: Blob,
  hits: readonly { identity: string; score: number }[] | null, decision: { identity: string | null; languageConfirmed: boolean }, bundleVersion: string): Promise<void> {
  if (!import.meta.env.DEV) return;
  try {
    const bitmap = await createImageBitmap(crop);
    const body = {
      schema_version: 1,
      kind: decision.identity ? 'user_selected_identity' : 'user_rejected_all_candidates',
      label_origin: 'user_decision_in_collection',
      identity: decision.identity,
      language_confirmed: decision.languageConfirmed,
      machine: { saved_identity: saved.identity, top: hits?.slice(0, 10).map((h) => ({ identity: h.identity, score: +h.score.toFixed(4) })) ?? [] },
      geometry: { crop_size: [bitmap.width, bitmap.height] },
      bundle_version: bundleVersion,
      camera: 'unknown',
      test_run: isTestRun(),
      page: saved.pageNumber,
      card_number: saved.number,
      created_at: new Date().toISOString(),
      crop_jpeg_base64: await jpeg(bitmap),
    };
    bitmap.close();
    await fetch('/__feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  } catch { /* feedback must never break the collection */ }
}

/**
 * Dev only ("Save captures for training"): a full capture as JPEG plus its detections, POSTed to the
 * dev server (/__capture), which stores them under datasets/sources/phone/captures/<date>/. Never in production builds.
 */
export async function uploadCapture(jpeg: Blob, meta: {
  captureSize: { width: number; height: number }; quads: number[][][]; scores: number[]; bundleVersion: string;
  cameraMock: boolean; workflow: string; page: number | null;
  captureId: string; sessionId: string; sourceVisitId: string; userAgent: string; provider: string | null;
}): Promise<void> {
  if (!import.meta.env.DEV) return;
  const body = {
    schema_version: 1, kind: 'scanner_capture', label_origin: 'detector_proposals_unverified',
    capture_size: meta.captureSize, detections: meta.quads.map((q, i) => ({ quad_capture_px: q, score: meta.scores[i] ?? null })),
    bundle_version: meta.bundleVersion, camera: meta.cameraMock ? 'test_video' : 'camera', workflow: meta.workflow,
    page: meta.page, test_run: isTestRun(), created_at: new Date().toISOString(),
    capture_id: meta.captureId, scan_session_id: meta.sessionId, source_visit_id: meta.sourceVisitId,
    device: { user_agent: meta.userAgent, execution_provider: meta.provider },
  };
  try {
    await fetch('/__capture?meta=' + encodeURIComponent(JSON.stringify(body)), { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: jpeg });
  } catch { /* uploads must never break scanning */ }
}
