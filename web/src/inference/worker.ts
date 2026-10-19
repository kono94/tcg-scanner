/// <reference lib="webworker" />
// Inference worker: owns the ONNX sessions, the gallery and at most one capture's pixels.
import * as ort from 'onnxruntime-web/webgpu';
import { Gallery } from '../recognition/gallery';
import { appearanceFingerprint } from '../batch/duplicates';
import { rgbFromRGBA, type Quad, type RGBImage } from '../vision/image';
import { letterboxFor } from '../vision/quad';
import { boxOfQuad, decodePose } from '../vision/pose';
import { CANONICAL_HEIGHT, CANONICAL_WIDTH, encoderTensor, warpCard } from '../vision/preprocess';
import type { BundleManifest, FromWorker, Provider, Quality, ToWorker } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

interface Models {
  provider: Provider;
  encoder: ort.InferenceSession;
  detector: ort.InferenceSession;
}

let manifest: BundleManifest | null = null;
let gallery: Gallery | null = null;
let modelBytes: { encoder: Uint8Array; detector: Uint8Array } | null = null;
let models: Models | null = null;
let capture: { id: number; image: RGBImage } | null = null;
// Serialize jobs so the UI thread's admission control is the only queue.
let chain: Promise<void> = Promise.resolve();

const post = (msg: FromWorker, transfer: Transferable[] = []) => self.postMessage(msg, transfer);
/** Breadcrumb for the dev crash log (forwarded by the UI thread). */
const debug = (stage: string, data: Record<string, unknown> = {}) => post({ type: 'debug', stage, data });

async function fetchCached(url: string, cacheName: string, stage: string): Promise<ArrayBuffer> {
  let cache: Cache | null = null;
  try {
    cache = await caches.open(cacheName);
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      post({ type: 'progress', stage, loaded: buf.byteLength, total: buf.byteLength });
      return buf;
    }
  } catch {
    cache = null; // Cache Storage can be unavailable (private mode); fetch directly.
  }
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Failed to load ${url} (${response.status})`);
  const total = Number(response.headers.get('content-length')) || undefined;
  const reader = response.body.getReader();
  // One buffer when the size is known (no chunk list + copy: big models are tight on phones).
  let bytes = new Uint8Array(total ?? 0);
  let loaded = 0, lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.byteLength > bytes.byteLength) {
      const grown = new Uint8Array(Math.max(bytes.byteLength * 2, loaded + value.byteLength));
      grown.set(bytes.subarray(0, loaded));
      bytes = grown;
    }
    bytes.set(value, loaded);
    loaded += value.byteLength;
    if (performance.now() - lastPost > 150) {
      post({ type: 'progress', stage, loaded, total: total ?? loaded });
      lastPost = performance.now();
    }
  }
  post({ type: 'progress', stage, loaded, total: loaded });
  const exact = loaded === bytes.byteLength ? bytes : bytes.slice(0, loaded);
  try { await cache?.put(url, new Response(exact)); } catch { /* quota: run without caching */ }
  return exact.buffer as ArrayBuffer;
}

async function dropOldCaches(current: string) {
  try {
    for (const name of await caches.keys()) if (name.startsWith('tcg-bundle-') && name !== current) await caches.delete(name);
  } catch { /* ignore */ }
}

let modelUrls: { encoder: string; detector: string; cache: string } | null = null;

async function createModels(provider: Provider): Promise<Models> {
  if (!modelBytes && modelUrls) {
    modelBytes = {
      encoder: new Uint8Array(await fetchCached(modelUrls.encoder, modelUrls.cache, 'encoder')),
      detector: new Uint8Array(await fetchCached(modelUrls.detector, modelUrls.cache, 'detector')),
    };
  }
  if (!modelBytes) throw new Error('Models not loaded');
  const options: ort.InferenceSession.SessionOptions = { executionProviders: [provider], graphOptimizationLevel: 'all' };
  const encoder = await ort.InferenceSession.create(modelBytes.encoder, options);
  const detector = await ort.InferenceSession.create(modelBytes.detector, options);
  // Warm up so the first card is not penalized by shader compilation.
  await encoder.run({ [encoder.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * 224 * 224), [1, 3, 224, 224]) });
  const size = manifest!.detector.size;
  await detector.run({ [detector.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]) });
  return { provider, encoder, detector };
}

const isIOS = () => /iPhone|iPad|iPod/.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);

async function init(msg: Extract<ToWorker, { type: 'init' }>) {
  const started = performance.now();
  manifest = msg.manifest;
  ort.env.wasm.wasmPaths = new URL('ort/', msg.baseUrl.replace(/bundle\/?$/, '')).href;
  // iOS/iPadOS (every browser there is WebKit): the tab memory limit is tight. Use single-threaded
  // WASM with the int8 encoder by default; threaded WASM and the fp32 encoder can exceed it.
  const ios = isIOS();
  if (ios) ort.env.wasm.numThreads = 1;
  const gpu = (navigator as Navigator & { gpu?: unknown }).gpu;
  const tryGPU = msg.preferWebGPU && !!gpu && !ios;
  const cacheName = `tcg-bundle-${manifest.version}`;
  const base = msg.baseUrl.endsWith('/') ? msg.baseUrl : msg.baseUrl + '/';
  const v = '?v=' + manifest.version;
  // The GPU path runs the fp32 encoder; the CPU path the int8 one (when the bundle has it).
  const encoderPath = !tryGPU && manifest.encoder.int8 ? manifest.encoder.int8.path : manifest.encoder.path;
  debug('init', { threads: ort.env.wasm.numThreads ?? 'default', encoder: encoderPath, tryGPU });
  modelUrls = { encoder: base + encoderPath + v, detector: base + manifest.detector.path + v, cache: cacheName };
  const [encoder, detector, vectors] = await Promise.all([
    fetchCached(modelUrls.encoder, cacheName, 'encoder'),
    fetchCached(modelUrls.detector, cacheName, 'detector'),
    fetchCached(base + manifest.gallery.path + v, cacheName, 'gallery'),
  ]);
  modelBytes = { encoder: new Uint8Array(encoder), detector: new Uint8Array(detector) };
  // iOS: int8 gallery rows (~9 MB instead of ~37 MB float32); top-1 parity is unit-tested.
  gallery = ios ? Gallery.fromFloat16Int8(vectors, manifest.gallery.dimension, msg.identities)
    : Gallery.fromFloat16(vectors, manifest.gallery.dimension, msg.identities);
  post({ type: 'progress', stage: 'sessions' });
  let fallbackReason: string | undefined;
  if (tryGPU) {
    try {
      models = await createModels('webgpu');
    } catch (err) {
      fallbackReason = `WebGPU unavailable: ${(err as Error).message}`;
    }
  } else {
    fallbackReason = ios ? 'iOS: CPU path for memory' : gpu ? 'WebGPU disabled in settings' : 'WebGPU not supported by this browser';
  }
  if (!models) {
    if (tryGPU && manifest.encoder.int8) {
      // WebGPU failed: switch to the smaller CPU encoder.
      modelUrls.encoder = base + manifest.encoder.int8.path + v;
      modelBytes = null;
    }
    models = await createModels('wasm');
  }
  if (models.provider === 'wasm') modelBytes = null; // nothing left to fall back to
  void dropOldCaches(cacheName);
  post({ type: 'ready', provider: models.provider, fallbackReason, loadMs: performance.now() - started });
}

/** Run with the current provider; on a WebGPU failure, fall back to WASM once and retry. */
async function runWithFallback<T>(fn: (m: Models) => Promise<T>): Promise<T> {
  if (!models) throw new Error('Models not ready');
  try {
    return await fn(models);
  } catch (err) {
    if (models.provider !== 'webgpu') throw err;
    const reason = `WebGPU inference failed: ${(err as Error).message}`;
    if (manifest?.encoder.int8 && modelUrls) {
      modelUrls.encoder = modelUrls.encoder.replace(manifest.encoder.path, manifest.encoder.int8.path);
      modelBytes = null;
    }
    models = await createModels('wasm');
    modelBytes = null;
    post({ type: 'provider-fallback', provider: 'wasm', reason });
    return fn(models);
  }
}

// Reused across captures: WebKit frees canvas backing stores late, so fresh 15 MB canvases per
// scan pile up until the tab is killed. One full-size canvas, one letterbox canvas, one RGB buffer.
let fullCtx: OffscreenCanvasRenderingContext2D | null = null;
let boxCtx: OffscreenCanvasRenderingContext2D | null = null;
let rgbBuffer: Uint8Array | null = null;

function context(ctx: OffscreenCanvasRenderingContext2D | null, w: number, h: number): OffscreenCanvasRenderingContext2D {
  if (!ctx) return new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true })!;
  if (ctx.canvas.width !== w || ctx.canvas.height !== h) { ctx.canvas.width = w; ctx.canvas.height = h; }
  return ctx;
}

async function detect(msg: Extract<ToWorker, { type: 'detect' }>) {
  const started = performance.now();
  const { bitmap } = msg;
  const { width, height } = bitmap;
  debug('detect:start', { width, height });
  // Keep exactly one capture: the new one replaces any previous pixels (and reuses its buffer).
  capture = null;
  fullCtx = context(fullCtx, width, height);
  fullCtx.drawImage(bitmap, 0, 0);
  const rgba = fullCtx.getImageData(0, 0, width, height).data;
  if (!rgbBuffer || rgbBuffer.length !== width * height * 3) rgbBuffer = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < width * height; i++, j += 3) {
    rgbBuffer[j] = rgba[i * 4]; rgbBuffer[j + 1] = rgba[i * 4 + 1]; rgbBuffer[j + 2] = rgba[i * 4 + 2];
  }
  capture = { id: msg.captureId, image: { width, height, data: rgbBuffer } };
  fullCtx.canvas.width = fullCtx.canvas.height = 1; // free the full-size RGBA backing store now
  debug('detect:pixels', { rgbMB: +(rgbBuffer.byteLength / 1e6).toFixed(1) });
  const size = manifest!.detector.size;
  const lb = letterboxFor(width, height, size);
  const box = boxCtx = context(boxCtx, size, size);
  const pad = manifest!.detector.pad_value;
  box.fillStyle = `rgb(${pad},${pad},${pad})`;
  box.fillRect(0, 0, size, size);
  box.imageSmoothingQuality = 'high';
  box.drawImage(bitmap, lb.padX, lb.padY, Math.round(width * lb.scale), Math.round(height * lb.scale));
  bitmap.close();
  const small = box.getImageData(0, 0, size, size).data;
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    input[i] = small[i * 4] / 255;
    input[plane + i] = small[i * 4 + 1] / 255;
    input[2 * plane + i] = small[i * 4 + 2] / 255;
  }
  debug('detect:infer');
  const output = await runWithFallback(async (m) => {
    const result = await m.detector.run({ [m.detector.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, size, size]) });
    return result[m.detector.outputNames[0]];
  });
  const d = manifest!.detector;
  // Corner model: the predicted corners are used directly for perspective correction.
  const found = decodePose(output.data as Float32Array, output.dims[2], lb, d.conf, d.iou);
  post({ type: 'detected', captureId: msg.captureId, width, height, boxes: found.map((f) => boxOfQuad(f.corners, f.score)),
    quads: found.map((f) => f.corners), ms: performance.now() - started });
}

function quality(quad: Quad, crop: RGBImage): Quality {
  const edges = quad.map((p, i) => Math.hypot(p[0] - quad[(i + 1) % 4][0], p[1] - quad[(i + 1) % 4][1]));
  let bright = 0;
  const d = crop.data;
  for (let i = 0; i < d.length; i += 3) if (d[i] > 240 && d[i + 1] > 240 && d[i + 2] > 240) bright++;
  return { shortSide: Math.min(...edges), glare: bright / (d.length / 3) };
}

async function bitmapOf(image: RGBImage): Promise<ImageBitmap> {
  const rgba = new Uint8ClampedArray(image.width * image.height * 4);
  for (let i = 0, j = 0; j < image.data.length; i += 4, j += 3) {
    rgba[i] = image.data[j]; rgba[i + 1] = image.data[j + 1]; rgba[i + 2] = image.data[j + 2]; rgba[i + 3] = 255;
  }
  return createImageBitmap(new ImageData(rgba, image.width, image.height));
}

async function displayCrop(msg: Extract<ToWorker, { type: 'crop' }>) {
  if (!capture || capture.id !== msg.captureId) throw new Error('Capture was released');
  const bitmap = await bitmapOf(warpCard(capture.image, msg.quad, msg.width, msg.height));
  post({ type: 'cropped', requestId: msg.requestId, bitmap }, [bitmap]);
}

/** RGB pixels on a reusable canvas (copied to RGBA). */
function rgbCanvas(image: RGBImage, ctx: OffscreenCanvasRenderingContext2D | null): OffscreenCanvasRenderingContext2D {
  const c = context(ctx, image.width, image.height);
  const rgba = new Uint8ClampedArray(image.width * image.height * 4);
  for (let i = 0, j = 0; j < image.data.length; i += 4, j += 3) {
    rgba[i] = image.data[j]; rgba[i + 1] = image.data[j + 1]; rgba[i + 2] = image.data[j + 2]; rgba[i + 3] = 255;
  }
  c.putImageData(new ImageData(rgba, image.width, image.height), 0, 0);
  return c;
}

let cropCtx: OffscreenCanvasRenderingContext2D | null = null;
let thumbCtx: OffscreenCanvasRenderingContext2D | null = null;

/** Batch mode: canonical crop (PNG), display crop and thumbnail (JPEG) of one card, then nothing is kept. */
async function extract(msg: Extract<ToWorker, { type: 'extract' }>) {
  if (!capture || capture.id !== msg.captureId) throw new Error('Capture was released');
  const crop = warpCard(capture.image, msg.quad);
  const q = quality(msg.quad, crop);
  cropCtx = rgbCanvas(crop, cropCtx);
  const canonical = await cropCtx.canvas.convertToBlob({ type: 'image/png' });
  thumbCtx = context(thumbCtx, msg.thumb.width, msg.thumb.height);
  thumbCtx.imageSmoothingQuality = 'high';
  thumbCtx.drawImage(cropCtx.canvas, 0, 0, msg.thumb.width, msg.thumb.height);
  const thumb = await thumbCtx.canvas.convertToBlob({ type: 'image/jpeg', quality: 0.7 });
  cropCtx = rgbCanvas(warpCard(capture.image, msg.quad, msg.display.width, msg.display.height), cropCtx);
  const display = await cropCtx.canvas.convertToBlob({ type: 'image/jpeg', quality: 0.85 });
  post({ type: 'extracted', requestId: msg.requestId, canonical, display, thumb, quality: q, appearance: appearanceFingerprint(crop) });
}

async function encodeCapture(msg: Extract<ToWorker, { type: 'encode-capture' }>) {
  if (!capture || capture.id !== msg.captureId) throw new Error('Capture was released');
  const ctx = rgbCanvas(capture.image, fullCtx);
  const blob = await ctx.canvas.convertToBlob({ type: 'image/jpeg', quality: msg.quality });
  ctx.canvas.width = ctx.canvas.height = 1; // free the full-size backing store now
  fullCtx = ctx;
  post({ type: 'encoded', requestId: msg.requestId, blob });
}

async function recognize(msg: Extract<ToWorker, { type: 'recognize' }>) {
  if (!capture || capture.id !== msg.captureId) throw new Error('Capture was released');
  const tag = { captureId: msg.captureId, instanceId: msg.instanceId, revision: msg.revision };
  debug('recognize:start', { card: msg.instanceId });
  const t0 = performance.now();
  const quad = msg.quad;
  const crop = warpCard(capture.image, quad);
  const bitmap = await bitmapOf(crop);
  const t1 = performance.now();
  post({ type: 'step', ...tag, step: 'rectified', crop: bitmap, at: t1 }, [bitmap]);
  const tensor = encoderTensor(crop);
  const embedding = await runWithFallback(async (m) => {
    const out = await m.encoder.run({ [m.encoder.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, 224, 224]) });
    return Float32Array.from(out[m.encoder.outputNames[0]].data as Float32Array);
  });
  const t2 = performance.now();
  post({ type: 'step', ...tag, step: 'embedded', at: t2 });
  const hits = gallery!.search(embedding, 20);
  const t3 = performance.now();
  post({ type: 'step', ...tag, step: 'searched', at: t3 });
  debug('recognize:done', { card: msg.instanceId, embedMs: Math.round(t2 - t1) });
  post({ type: 'recognized', ...tag, hits, quality: quality(quad, crop), ms: { rectify: t1 - t0, embed: t2 - t1, search: t3 - t2 } });
}

let embedCtx: OffscreenCanvasRenderingContext2D | null = null;

async function embedSaved(msg: Extract<ToWorker, { type: 'embed' }>) {
  const { bitmap } = msg;
  if (bitmap.width !== CANONICAL_WIDTH || bitmap.height !== CANONICAL_HEIGHT) throw new Error('Saved crop is not a canonical card crop');
  // One reused canvas: batch mode embeds hundreds of saved crops; fresh canvases pile up on WebKit.
  const ctx = embedCtx = context(embedCtx, bitmap.width, bitmap.height);
  const c = ctx.canvas;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, c.width, c.height).data;
  const data = new Uint8Array(c.width * c.height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) { data[j] = rgba[i]; data[j + 1] = rgba[i + 1]; data[j + 2] = rgba[i + 2]; }
  const tensor = encoderTensor({ width: c.width, height: c.height, data });
  const embedding = await runWithFallback(async (m) => {
    const out = await m.encoder.run({ [m.encoder.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, 224, 224]) });
    return Float32Array.from(out[m.encoder.outputNames[0]].data as Float32Array);
  });
  post({ type: 'searched', requestId: msg.requestId, hits: gallery!.search(embedding, 20) });
}

self.onmessage = (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  if (msg.type === 'release') {
    if (capture?.id === msg.captureId) {
      capture = null;
      // The RGB buffer is reused for the next capture of the same size; drop it after a batch-mode
      // release so nothing capture-sized stays alive between pages.
      if (msg.drop) rgbBuffer = null;
    }
    return;
  }
  chain = chain.then(async () => {
    try {
      if (msg.type === 'init') await init(msg);
      else if (msg.type === 'detect') await detect(msg);
      else if (msg.type === 'recognize') await recognize(msg);
      else if (msg.type === 'crop') await displayCrop(msg);
      else if (msg.type === 'embed') await embedSaved(msg);
      else if (msg.type === 'extract') await extract(msg);
      else if (msg.type === 'fingerprint') {
        try {
          const ctx = context(null, msg.bitmap.width, msg.bitmap.height);
          ctx.drawImage(msg.bitmap, 0, 0);
          const rgba = ctx.getImageData(0, 0, msg.bitmap.width, msg.bitmap.height);
          post({ type: 'fingerprinted', requestId: msg.requestId, appearance: appearanceFingerprint(rgbFromRGBA(rgba.data, rgba.width, rgba.height)) });
          ctx.canvas.width = ctx.canvas.height = 1;
        }
        finally { msg.bitmap.close(); }
      }
      else if (msg.type === 'encode-capture') await encodeCapture(msg);
    } catch (err) {
      const tag = msg.type === 'recognize' ? { captureId: msg.captureId, instanceId: msg.instanceId, revision: msg.revision }
        : msg.type === 'detect' ? { captureId: msg.captureId } : msg.type === 'crop' || msg.type === 'embed' || msg.type === 'extract' || msg.type === 'fingerprint' || msg.type === 'encode-capture' ? { requestId: msg.requestId } : {};
      post({ type: 'error', ...tag, message: (err as Error).message ?? String(err) });
    }
  });
};
