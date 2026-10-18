// Typed messages between the UI thread and the inference worker.
// Every job carries its capture ID (and instance revision) so obsolete results can be dropped.
import type { Hit } from '../recognition/gallery';
import type { Quad } from '../vision/image';
import type { OrientedBox } from '../vision/obb';

export type Provider = 'webgpu' | 'wasm';

export interface BundleManifest {
  version: string;
  /** `int8`: quantized encoder used on the WASM/CPU path (much smaller; top-1 parity checked offline). */
  encoder: { path: string; input: string; output: string; size: number; dimension: number; preprocessing: { version: string }; int8?: { path: string; sha256: string } };
  /** kind 'pose': box + 4 card-corner keypoints; 'obb': rotated rectangles (corners derived). */
  detector: { path: string; size: number; pad_value: number; conf: number; iou: number; weights_origin: string; kind?: 'obb' | 'pose' };
  gallery: { path: string; rows: number; dimension: number; dtype: string };
  catalog: string;
  prices: string;
  thresholds_calibrated: boolean;
}

export type ToWorker =
  | { type: 'init'; baseUrl: string; manifest: BundleManifest; identities: string[]; preferWebGPU: boolean }
  | { type: 'detect'; captureId: number; bitmap: ImageBitmap }
  | { type: 'recognize'; captureId: number; instanceId: string; revision: number; quad: Quad }
  /** `drop`: also free the worker's capture-sized buffer (batch mode keeps nothing between pages). */
  | { type: 'release'; captureId: number; drop?: boolean }
  /** Display-only crop at a larger size from the same full-resolution capture. */
  | { type: 'crop'; requestId: number; captureId: number; quad: Quad; width: number; height: number }
  /** Re-check a saved canonical crop (already straightened, encoder-contract size). */
  | { type: 'embed'; requestId: number; bitmap: ImageBitmap }
  | { type: 'fingerprint'; requestId: number; bitmap: ImageBitmap }
  /** Batch mode: straightened crops of one card from the held capture, encoded for storage. */
  | { type: 'extract'; requestId: number; captureId: number; quad: Quad; display: { width: number; height: number }; thumb: { width: number; height: number } }
  /** Dev only: the held capture as JPEG (training uploads). */
  | { type: 'encode-capture'; requestId: number; captureId: number; quality: number };

export interface Quality {
  /** Short side of the card in capture pixels. */
  shortSide: number;
  /** Fraction of near-saturated pixels in the rectified crop (glare heuristic, uncalibrated). */
  glare: number;
}

export type RecognitionStep = 'rectified' | 'embedded' | 'searched';

export type FromWorker =
  | { type: 'progress'; stage: string; loaded?: number; total?: number }
  | { type: 'ready'; provider: Provider; fallbackReason?: string; loadMs: number }
  | { type: 'detected'; captureId: number; width: number; height: number; boxes: OrientedBox[]; quads?: Quad[]; ms: number }
  | { type: 'step'; captureId: number; instanceId: string; revision: number; step: RecognitionStep; crop?: ImageBitmap; at: number }
  | { type: 'recognized'; captureId: number; instanceId: string; revision: number; hits: Hit[]; quality: Quality; ms: { rectify: number; embed: number; search: number } }
  | { type: 'provider-fallback'; provider: Provider; reason: string }
  | { type: 'cropped'; requestId: number; bitmap: ImageBitmap }
  | { type: 'searched'; requestId: number; hits: Hit[] }
  /** canonical: PNG (exact encoder-contract pixels); display/thumb: JPEG. */
  | { type: 'extracted'; requestId: number; canonical: Blob; display: Blob; thumb: Blob; quality: Quality; appearance: string }
  | { type: 'fingerprinted'; requestId: number; appearance: string }
  | { type: 'encoded'; requestId: number; blob: Blob }
  | { type: 'debug'; stage: string; data: Record<string, unknown> }
  | { type: 'error'; captureId?: number; instanceId?: string; revision?: number; requestId?: number; message: string };
