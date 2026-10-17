// UI-thread side of the inference worker; implements the session's Recognizer interface.
import { trace } from '../debug';
import type { RecognitionJob, RecognitionOutput, Recognizer } from '../session/page';
import type { Hit } from '../recognition/gallery';
import type { Quad } from '../vision/image';
import type { OrientedBox } from '../vision/obb';
import type { BundleManifest, FromWorker, Provider, RecognitionStep } from './protocol';

interface Pending<T> {
  resolve: (v: T) => void;
  reject: (e: Error) => void;
}

export interface RuntimeStatus {
  stage: 'loading' | 'ready' | 'failed';
  detail: string;
  provider: Provider | null;
  fallbackReason?: string;
  loadMs?: number;
  /** Download progress 0..1 across all model files, while loading. */
  progress?: number;
}

export class WorkerRecognizer implements Recognizer {
  private readonly worker: Worker;
  private ready: Promise<void>;
  private detects = new Map<number, Pending<{ width: number; height: number; boxes: OrientedBox[]; quads?: Quad[]; ms: number }>>();
  private jobs = new Map<string, Pending<RecognitionOutput> & { onStep: (s: RecognitionStep, crop?: ImageBitmap) => void }>();
  private crops = new Map<number, Pending<ImageBitmap>>();
  private searches = new Map<number, Pending<Hit[]>>();
  private nextCrop = 1;
  status: RuntimeStatus = { stage: 'loading', detail: 'Starting', provider: null };
  private downloads = new Map<string, { loaded: number; total: number }>();
  /** Called when the worker dies (e.g. out of memory); the page cannot recover without a reload. */
  onCrash: (message: string) => void = () => {};
  onStatus: (s: RuntimeStatus) => void = () => {};

  constructor(baseUrl: string, manifest: BundleManifest, identities: string[], preferWebGPU: boolean) {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    let markReady: () => void = () => {};
    let fail: (e: Error) => void = () => {};
    this.ready = new Promise((res, rej) => { markReady = res; fail = rej; });
    this.ready.catch(() => {});
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === 'progress') {
        if (m.stage === 'sessions') {
          this.setStatus({ ...this.status, detail: 'Preparing models…', progress: 1 });
        } else {
          this.downloads.set(m.stage, { loaded: m.loaded ?? 0, total: m.total ?? 0 });
          let loaded = 0, total = 0;
          for (const d of this.downloads.values()) { loaded += d.loaded; total += d.total; }
          const mb = (b: number) => (b / 1e6).toFixed(0);
          this.setStatus({ ...this.status, progress: total ? loaded / total : undefined,
            detail: total ? `Downloading ${mb(loaded)} of ${mb(total)} MB` : `Loading ${m.stage}` });
        }
      } else if (m.type === 'ready') {
        this.setStatus({ stage: 'ready', detail: 'Ready', provider: m.provider, fallbackReason: m.fallbackReason, loadMs: m.loadMs });
        markReady();
      } else if (m.type === 'provider-fallback') {
        this.setStatus({ ...this.status, provider: m.provider, fallbackReason: m.reason });
      } else if (m.type === 'detected') {
        this.detects.get(m.captureId)?.resolve(m);
        this.detects.delete(m.captureId);
      } else if (m.type === 'step') {
        const job = this.jobs.get(key(m));
        if (job) job.onStep(m.step, m.crop);
        else m.crop?.close();
      } else if (m.type === 'recognized') {
        this.jobs.get(key(m))?.resolve(m);
        this.jobs.delete(key(m));
      } else if (m.type === 'debug') {
        trace(`worker:${m.stage}`, m.data);
      } else if (m.type === 'cropped') {
        this.crops.get(m.requestId)?.resolve(m.bitmap);
        this.crops.delete(m.requestId);
      } else if (m.type === 'searched') {
        this.searches.get(m.requestId)?.resolve(m.hits);
        this.searches.delete(m.requestId);
      } else if (m.type === 'error') {
        const err = new Error(m.message);
        if (m.requestId !== undefined) {
          this.crops.get(m.requestId)?.reject(err);
          this.crops.delete(m.requestId);
          this.searches.get(m.requestId)?.reject(err);
          this.searches.delete(m.requestId);
        } else if (m.instanceId !== undefined && m.captureId !== undefined && m.revision !== undefined) {
          const k = key({ captureId: m.captureId, instanceId: m.instanceId, revision: m.revision });
          this.jobs.get(k)?.reject(err);
          this.jobs.delete(k);
        } else if (m.captureId !== undefined) {
          this.detects.get(m.captureId)?.reject(err);
          this.detects.delete(m.captureId);
        } else {
          this.setStatus({ stage: 'failed', detail: m.message, provider: null });
          fail(err);
        }
      }
    };
    this.worker.onerror = (e) => {
      e.preventDefault();
      const message = e.message || 'Inference worker stopped (possibly out of memory)';
      for (const p of [...this.detects.values(), ...this.jobs.values(), ...this.crops.values()]) p.reject(new Error(message));
      this.detects.clear(); this.jobs.clear(); this.crops.clear();
      this.setStatus({ stage: 'failed', detail: message, provider: null });
      fail(new Error(message));
      this.onCrash(message);
    };
    this.worker.postMessage({ type: 'init', baseUrl, manifest, identities, preferWebGPU });
  }

  private setStatus(s: RuntimeStatus) {
    this.status = s;
    this.onStatus(s);
  }

  whenReady(): Promise<void> {
    return this.ready;
  }

  async detect(captureId: number, bitmap: ImageBitmap) {
    await this.ready;
    return new Promise<{ width: number; height: number; boxes: OrientedBox[]; quads?: Quad[]; ms: number }>((resolve, reject) => {
      this.detects.set(captureId, { resolve, reject });
      this.worker.postMessage({ type: 'detect', captureId, bitmap }, [bitmap]);
    });
  }

  recognize(job: RecognitionJob, onStep: (step: RecognitionStep, crop?: ImageBitmap) => void): Promise<RecognitionOutput> {
    return new Promise((resolve, reject) => {
      this.jobs.set(key(job), { resolve, reject, onStep });
      this.worker.postMessage({ type: 'recognize', ...job });
    });
  }

  /** Larger straightened crop for display, from the same capture. */
  displayCrop(captureId: number, quad: Quad, width = 630, height = 880): Promise<ImageBitmap> {
    const requestId = this.nextCrop++;
    return new Promise((resolve, reject) => {
      this.crops.set(requestId, { resolve, reject });
      this.worker.postMessage({ type: 'crop', requestId, captureId, quad, width, height });
    });
  }

  /** Embed and search a saved canonical crop (collection re-check). The bitmap is transferred. */
  async search(bitmap: ImageBitmap): Promise<Hit[]> {
    await this.ready;
    const requestId = this.nextCrop++;
    return new Promise((resolve, reject) => {
      this.searches.set(requestId, { resolve, reject });
      this.worker.postMessage({ type: 'embed', requestId, bitmap }, [bitmap]);
    });
  }

  release(captureId: number): void {
    for (const [k, job] of this.jobs) {
      if (k.startsWith(`${captureId}|`)) { job.reject(new Error('released')); this.jobs.delete(k); }
    }
    this.worker.postMessage({ type: 'release', captureId });
  }
}

function key(m: { captureId: number; instanceId: string; revision: number }): string {
  return `${m.captureId}|${m.instanceId}|${m.revision}`;
}
