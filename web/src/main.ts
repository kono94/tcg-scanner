// App controller: camera -> stability -> capture -> worker inference -> page session -> view.
import './ui/styles.css';
import { installCrashTrace, trace } from './debug';
import { sendCollectionFeedback, sendFeedback } from './feedback';
import { getCrops } from './collection/crops';
import { deletePage, deleteSavedCard, loadPages, savePage, updateSavedCard, withCurrentPrices } from './collection/store';
import { listMockVideos, openCamera, openVideoMock, type CameraHandle, type MockVideo } from './camera/source';
import { CaptureCycle } from './capture/cycle';
import { halve, lumaFromRGBA, sharpnessOf, StabilityMonitor, type StabilityState } from './capture/stability';
import { WorkerRecognizer } from './inference/client';
import type { BundleManifest } from './inference/protocol';
import { SnapshotPriceService } from './prices/service';
import { Catalog, type CatalogCard } from './recognition/catalog';
import { displayCard, userChoiceMatch } from './recognition/policy';
import { PageSession } from './session/page';
import { visibleRegion } from './ui/geometry';
import { Presenter } from './ui/presenter';
import { loadSettings, saveSettings, type Settings } from './ui/settings';
import { needsReview, View, type AppSnapshot, type CollectionState, type ViewerMode } from './ui/view';

const BUNDLE = new URL('bundle/', document.baseURI).href;

async function boot() {
  installCrashTrace();
  const root = document.getElementById('app')!;
  const settings = loadSettings();
  // ?wasm forces CPU inference (diagnosing GPU-related crashes on phones).
  if (new URLSearchParams(location.search).has('wasm')) settings.preferWebGPU = false;
  applyTheme(settings.theme);
  const view = new View(root);
  view.renderBoot('Loading catalog');
  let manifest: BundleManifest;
  let catalog: Catalog;
  let identities: string[];
  try {
    manifest = await (await fetch(BUNDLE + 'manifest.json', { cache: 'no-store' })).json();
    const cat = (await (await fetch(`${BUNDLE}${manifest.catalog}?v=${manifest.version}`)).json()) as { cards: CatalogCard[]; gallery_identities?: string[] };
    catalog = new Catalog(cat.cards);
    // Search rows: cards with a reference image. Other language versions (e.g. German) are catalog-only.
    identities = cat.gallery_identities ?? cat.cards.map((c) => c.id);
  } catch (err) {
    view.renderBoot(`Model bundle not found. Run scripts/build_browser_bundle.py first. (${(err as Error).message})`, true);
    return;
  }
  const recognizer = new WorkerRecognizer(BUNDLE, manifest, identities, settings.preferWebGPU);
  const prices = new SnapshotPriceService(`${BUNDLE}${manifest.prices}?v=${manifest.version}`, () => settings.priceMarket);
  const session = new PageSession(recognizer, { catalog, prices, preferredLanguage: () => settings.preferredLanguage });
  const app = new App(view, session, recognizer, settings, catalog, manifest, prices);
  let lastStage = '';
  recognizer.onStatus = () => {
    const st = recognizer.status;
    if (st.stage !== lastStage || st.fallbackReason) trace('runtime', { stage: st.stage, provider: st.provider, detail: st.detail, fallback: st.fallbackReason, loadMs: st.loadMs });
    lastStage = st.stage;
    app.render();
  };
  recognizer.onCrash = (message) => trace('worker-crash', { message });
  if (import.meta.env.DEV) {
    // Dev-only hook for parity checks of the in-browser detection path.
    (window as unknown as { tcgDebug: unknown }).tcgDebug = {
      ready: () => recognizer.whenReady(),
      runtime: () => recognizer.status,
      page: () => session.page,
      detectUrl: async (url: string) => recognizer.detect(-1, await createImageBitmap(await (await fetch(url)).blob())),
    };
  }
  session.subscribe(() => app.onSessionChange());
  app.showStart(await listMockVideos());
  const auto = new URLSearchParams(location.search).get('mock');
  if (auto !== null) {
    const videos = await listMockVideos();
    const v = videos.find((x) => x.url.endsWith(auto)) ?? videos[Number(auto) || 0];
    if (v) void app.useSource(() => openVideoMock(v.url, v.name));
  }
}

/** iPhone/iPad (every browser there is WebKit with a tight per-tab memory limit). */
export function isIOS(): boolean {
  return /iPhone|iPad|iPod/.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
}

function applyTheme(theme: Settings['theme']) {
  if (theme === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
}

export class App {
  camera: CameraHandle | null = null;
  private monitor: StabilityMonitor;
  private cycle: CaptureCycle;
  private stability: StabilityState = { steady: false, steadyMs: 0, motion: null, sharp: false, reason: 'warming-up' };
  private countdownMs: number | null = null;
  private sampler: ReturnType<typeof setInterval> | null = null;
  private sampleCanvas: HTMLCanvasElement = document.createElement('canvas');
  private frameId = 0;
  private presenter = new Presenter();
  private raf = 0;
  private selected: string | null = null;
  private viewerMode: ViewerMode = 'card';
  private compare: string | null = null;
  private expanded = false;
  /** Large display crops per instance revision; dropped with the capture. */
  private bigCrops = new Map<string, ImageBitmap>();
  private pendingCrops = new Set<string>();
  private lastStatus = '';
  /** Guided check: the cards that needed the user when it started, in page order. */
  private check: string[] | null = null;
  private overlay: 'none' | 'review' | 'settings' | 'collection' | 'start' = 'start';
  private collection: CollectionState = { pages: loadPages(), open: null, usdToEur: null, card: null, rev: 0 };
  private collectionToken = 0;
  private cardToken = 0;
  private sweepAt = 0;
  private hiddenStop = false;
  private videos: MockVideo[] = [];
  private sourceFactory: (() => Promise<CameraHandle>) | null = null;
  private error: string | null = null;

  constructor(
    private readonly view: View,
    private readonly session: PageSession,
    private readonly recognizer: WorkerRecognizer,
    private readonly settings: Settings,
    private readonly catalog: Catalog,
    private readonly manifest: BundleManifest,
    private readonly prices: SnapshotPriceService,
  ) {
    this.monitor = new StabilityMonitor(settings.tolerance);
    this.cycle = new CaptureCycle({ mode: settings.mode, dwellMs: settings.dwellMs, cooldownMs: settings.cooldownMs });
    this.sampleCanvas.width = 128;
    this.sampleCanvas.height = 96;
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    this.presenter.options.reducedMotion = motion.matches;
    motion.addEventListener('change', () => { this.presenter.options.reducedMotion = motion.matches; });
    document.addEventListener('visibilitychange', () => this.onVisibility());
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { this.view.themeChanged(); this.render(); });
    window.addEventListener('resize', () => { this.view.redrawStill(); this.render(); });
    view.handlers = {
      primary: () => this.primary(),
      rescan: () => this.rescan(),
      settings: () => { this.overlay = 'settings'; this.render(); },
      theme: () => this.toggleTheme(),
      select: (id) => this.open(id),
      viewerMode: (mode) => { this.viewerMode = mode; this.render(); },
      compare: (family) => { this.compare = family; this.render(); },
      step: (delta) => this.stepViewer(delta),
      choose: (id, identity, languageConfirmed, advance) => {
        this.recordDecision(id, identity, languageConfirmed);
        this.session.choose(id, identity, languageConfirmed);
        this.compare = null;
        if (advance) this.stepViewer(1, true);
        else this.render();
      },
      chooseNone: (id) => { this.recordDecision(id, null, false); this.session.chooseNone(id); this.compare = null; this.stepViewer(1, true); },
      expand: (on) => { this.expanded = on; this.render(); },
      retry: (id) => this.session.retry(id),
      skip: (id) => { this.session.skip(id); if (this.check) this.stepViewer(1, true); },
      closeOverlay: () => { this.overlay = this.camera ? 'none' : 'start'; this.cycle.setFrozen(this.selected !== null); this.render(); },
      review: () => { this.overlay = 'review'; this.cycle.setFrozen(true); this.render(); },
      startCheck: () => this.startCheck(),
      tab: (name) => {
        this.selected = null;
        this.check = null;
        if (name === 'scan') {
          if (!this.camera) { void this.useSource(openCamera); return; }
          this.overlay = 'none';
          this.cycle.setFrozen(false);
          if (this.cycle.current === 'live') this.setCameraActive(true);
        } else {
          // Leaving the scanner: no camera frames or automatic captures in the background.
          if (this.cycle.current === 'live') this.setCameraActive(false);
          this.cycle.setFrozen(true);
          this.overlay = name === 'home' ? 'start' : name;
          this.reloadCollection();
          if (name === 'collection') this.collection.open = null;
        }
        this.render();
      },
      openSaved: (key) => { this.closeSavedCard(); this.collection.open = key; this.render(); },
      deleteSaved: (key) => { deletePage(key); this.closeSavedCard(); this.collection.open = null; this.reloadCollection(); this.render(); },
      deleteSavedCard: (key, number) => { deleteSavedCard(key, number); this.closeSavedCard(); if (!loadPages().some((p) => p.key === key)) this.collection.open = null; this.reloadCollection(); this.render(); },
      openSavedCard: (key, number) => { if (number === null) { this.closeSavedCard(); this.render(); } else void this.openSavedCard(key, number); },
      savedPick: (identity) => { if (this.collection.card) { this.collection.card.pick = identity; this.touchCollection(); } },
      savedSearch: (query) => {
        const c = this.collection.card;
        if (!c) return;
        c.query = query;
        c.results = this.catalog.search(query, this.settings.preferredLanguage);
        this.touchCollection();
      },
      savedChoose: (identity, languageConfirmed) => this.chooseSaved(identity, languageConfirmed),
      nextPage: () => this.nextPage(),
      save: () => this.save(),
      useCamera: () => this.useSource(openCamera),
      useVideo: (v) => this.useSource(() => openVideoMock(v.url, v.name)),
      resume: () => this.resumeSource(),
      changeSetting: (patch) => this.changeSetting(patch),
    };
  }

  /** Open (or close with null) the card viewer. */
  private open(id: string | null) {
    if (id === null) this.check = null;
    this.selected = id;
    this.compare = null;
    this.cycle.setFrozen(id !== null || this.overlay === 'review');
    if (id) {
      this.loadBigCrop(id);
      // Prepare the neighbours so stepping through cards shows the sharp crop immediately.
      const list = this.session.page.instances, i = list.findIndex((c) => c.id === id);
      for (const n of [list[i + 1], list[i - 1]]) if (n) this.loadBigCrop(n.id);
    }
    this.render();
  }

  /** Move through cards; at the end of a review pass, close and show the page review. */
  /** Store the user's decision as a training example (crop at display size, candidates, answer). */
  private recordDecision(id: string, identity: string | null, languageConfirmed: boolean) {
    const card = this.session.instance(id);
    if (!card) return;
    // Confirming the machine's own match on a clean card adds little; corrections and rejections matter.
    const m = card.decision?.kind === 'matched' ? card.decision : null;
    if (identity && m && m.top.identity === identity && card.state === 'confirmed' && !languageConfirmed) return;
    const crop = this.bigCrops.get(`${id}|${card.revision}`) ?? card.crop;
    const page = this.session.page;
    void sendFeedback(card, crop, { identity, languageConfirmed },
      { bundleVersion: this.manifest.version, pageNumber: page.pageNumber, captureSize: page.captureSize, cameraMock: !!this.camera?.mock });
    trace('feedback', { card: card.number, identity, none: identity === null });
  }

  /** Walk through the cards that need the user: pick a match or skip, one after another. */
  private startCheck() {
    const ids = this.session.page.instances.filter(needsReview).map((c) => c.id);
    if (!ids.length) return;
    this.check = ids;
    this.open(ids[0]);
  }

  private stepViewer(delta: number, afterDecision = false) {
    const list = this.session.page.instances;
    if (this.check) {
      const i = this.check.indexOf(this.selected ?? '');
      // After a decision: next card in the check that still needs the user; arrows: neighbours.
      const next = afterDecision
        ? this.check.slice(i + 1).find((id) => { const c = this.session.instance(id); return !!c && needsReview(c); })
        : this.check[i + delta];
      if (next) return this.open(next);
      if (afterDecision) { this.check = null; this.open(null); } // done: back to the page, "Next scan"
      return;
    }
    const i = list.findIndex((c) => c.id === this.selected);
    const next = list[i + delta];
    if (next) this.open(next.id);
    else if (afterDecision) this.open(null);
  }

  /** Save this page (idempotent) and start the next one. */
  private async nextScan() {
    if (this.session.page.instances.length) await this.save();
    this.nextPage();
  }

  private loadBigCrop(id: string) {
    const card = this.session.instance(id);
    const captureId = this.session.page.captureId;
    if (!card || captureId === null) return;
    const key = `${id}|${card.revision}`;
    if (this.bigCrops.has(key) || this.pendingCrops.has(key)) return;
    this.pendingCrops.add(key);
    this.recognizer.displayCrop(captureId, card.quad).then((bmp) => {
      this.pendingCrops.delete(key);
      if (this.session.page.captureId !== captureId) { bmp.close(); return; }
      this.bigCrops.set(key, bmp);
      this.render();
    }, () => this.pendingCrops.delete(key));
  }

  private dropBigCrops() {
    for (const b of this.bigCrops.values()) b.close();
    this.bigCrops.clear();
    this.pendingCrops.clear();
  }

  showStart(videos: MockVideo[]) {
    this.videos = videos;
    this.overlay = 'start';
    this.render();
  }

  async useSource(factory: () => Promise<CameraHandle>) {
    this.error = null;
    this.camera?.stop();
    this.camera = null;
    try {
      this.camera = await factory();
    } catch (err) {
      this.error = (err as Error).message;
      this.overlay = 'start';
      this.render();
      return;
    }
    this.sourceFactory = factory;
    trace('camera', { label: this.camera.label, mock: this.camera.mock });
    this.hiddenStop = false;
    this.overlay = 'none';
    await this.view.attachStream(this.camera.stream, (n) => { this.frameId = n; });
    this.monitor.reset();
    this.startSampling();
    this.render();
  }

  private async resumeSource() {
    if (this.sourceFactory) await this.useSource(this.sourceFactory);
  }

  private startSampling() {
    if (this.sampler) clearInterval(this.sampler);
    // No willReadFrequently: that keeps the canvas on the CPU, so every camera frame would be read
    // back at full size (10x per second) before shrinking. GPU-downscale, read back only 64x48.
    // If a browser returns blank pixels from the GPU path, switch once to a CPU canvas.
    let ctx = this.sampleCanvas.getContext('2d')!;
    let blank = 0, cpuFallback = false, shown = '', lastReason = '';
    this.sampler = setInterval(() => {
      const video = this.view.video;
      if (!this.camera || video.readyState < 2) return;
      // Sample only when a capture can be triggered: live, or review in repeat mode. During review
      // (and while the viewer or settings are open) the camera frame is not needed at all.
      const phase = this.cycle.current;
      if (!(phase === 'live' || (phase === 'review' && this.settings.mode === 'repeat'))) {
        if (this.countdownMs !== null) { this.countdownMs = null; this.render(); }
        return;
      }
      // 128x96: sharpness (blur) on this; motion on the 2x2-binned 64x48 (its calibrated scale).
      ctx.drawImage(video, 0, 0, 128, 96);
      const fine = lumaFromRGBA(ctx.getImageData(0, 0, 128, 96).data, 128, 96);
      const sharpness = sharpnessOf(fine, 128, 96);
      const luma = halve(fine, 128, 96);
      let sum = 0, hash = 0;
      for (let i = 0; i < luma.length; i++) { sum += luma[i]; hash = (hash * 31 + luma[i]) | 0; }
      if (sum === 0 && !cpuFallback && ++blank >= 3) {
        const c = document.createElement('canvas');
        c.width = 128; c.height = 96;
        this.sampleCanvas = c;
        ctx = c.getContext('2d', { willReadFrequently: true })!;
        cpuFallback = true;
        trace('sampler', { fallback: 'cpu-canvas' });
      }
      const t = performance.now();
      // New-frame identity: the frame callback when it is alive; otherwise the pixel content itself
      // (live sensor noise always changes it, a frozen preview does not). Some WebKit builds stop
      // frame callbacks after the stream is paused and resumed, which made every sample look stale.
      const callbackAlive = this.view.hasFrameCallback && t - this.view.lastFrameAt < 500;
      const frameId = callbackAlive ? this.frameId : hash;
      this.stability = this.monitor.push({ t, frameId, width: 64, height: 48, luma, sharpness });
      if (this.stability.reason !== lastReason) {
        lastReason = this.stability.reason;
        trace('steady', { reason: lastReason, motion: this.stability.motion === null ? null : +this.stability.motion.toFixed(2),
          sharpness: +sharpness.toFixed(2), sharp: this.stability.sharp,
          mean: +(sum / luma.length).toFixed(1), via: callbackAlive ? 'frame-callback' : 'pixels' });
      }
      // Captures wait for the models; the countdown is only shown when a capture is really possible.
      const ready = this.recognizer.status.stage === 'ready';
      const tick = this.cycle.tick(t, this.stability);
      this.countdownMs = ready && tick.view.armed ? tick.view.countdownMs : null;
      if (ready && tick.capture && !this.session.busy) void this.capture();
      const now = `${this.countdownMs === null ? '-' : Math.ceil(this.countdownMs / 100)}|${this.stability.reason}|${this.recognizer.status.stage}`;
      if (now !== shown) { shown = now; this.render(); }
    }, 100);
  }

  private stopSampling() {
    if (this.sampler) clearInterval(this.sampler);
    this.sampler = null;
  }

  private async capture() {
    if (this.session.busy || !this.camera || this.recognizer.status.stage !== 'ready') return;
    const video = this.view.video;
    if (video.readyState < 2) return;
    this.cycle.beginBatch();
    this.presenter.reset();
    this.selected = null;
    this.expanded = false;
    this.dropBigCrops();
    let full: ImageBitmap;
    try {
      // Capture exactly what the preview shows: the centre crop that object-fit: cover displays,
      // cut from the full-resolution frame. Then cap the size (a big frame is copied several times
      // on the way to the worker; phones have a tight per-tab memory limit).
      const [sx, sy, sw, sh] = visibleRegion(video, this.view.cameraBox());
      const longSide = Math.max(sw, sh), cap = isIOS() ? 2048 : 2560;
      const k = Math.min(1, cap / longSide);
      full = await createImageBitmap(video, sx, sy, sw, sh,
        k < 1 ? { resizeWidth: Math.round(sw * k), resizeHeight: Math.round(sh * k), resizeQuality: 'high' } : undefined);
    } catch (err) {
      this.cycle.settle(performance.now());
      this.cycle.rearm();
      this.error = (err as Error).message;
      return;
    }
    trace('capture', { video: `${video.videoWidth}x${video.videoHeight}`, capture: `${full.width}x${full.height}` });
    // The still frame is taken: no camera frames or steadiness checks until the user scans again.
    this.setCameraActive(false);
    // Display copy first, then hand the full-resolution capture to the worker.
    await this.view.setStill(full);
    this.sweepAt = performance.now();
    const done = this.session.capture(full);
    this.loop();
    await done.catch((err) => { this.error = (err as Error).message; });
  }

  onSessionChange() {
    const page = this.session.page;
    const status = `${page.generation}|${page.revision}|${page.status}`;
    const justSettled = status !== this.lastStatus && page.status === 'settled' && this.lastStatus.endsWith('processing');
    if (status !== this.lastStatus) trace('page', { status: page.status, cards: page.instances.length, detectMs: page.detectMs && Math.round(page.detectMs), error: page.error });
    this.lastStatus = status;
    if ((page.status === 'settled' || page.status === 'failed') && this.cycle.current === 'processing') {
      this.cycle.settle(performance.now());
      if (this.settings.mode === 'repeat') this.setCameraActive(true);
    }
    // "Review each card": step through the page right after it settles.
    if (justSettled && this.settings.reviewMode === 'review' && page.instances.length && !this.selected && this.overlay === 'none') {
      this.startCheck(); // only cards that need the user; nothing opens if everything matched
    }
    this.render();
    this.loop();
  }

  private loop() {
    if (this.raf) return;
    const frame = () => {
      this.raf = 0;
      this.render();
      if (this.presenter.animating || this.session.busy) this.raf = requestAnimationFrame(frame);
    };
    this.raf = requestAnimationFrame(frame);
  }

  private primary() {
    const status = this.session.page.status;
    if (!this.camera) { this.overlay = 'start'; this.render(); return; }
    if (this.cycle.current === 'live') {
      if (this.cycle.canCaptureManually()) void this.capture();
      return;
    }
    if (status === 'settled' && this.session.page.instances.length === 0) { this.rescan(); return; }
    if (status === 'failed') { this.rescan(); return; }
    if (status === 'processing') { this.overlay = 'review'; this.cycle.setFrozen(true); this.render(); return; }
    // Settled: check the cards that need the user, then move on.
    if (this.session.page.instances.some(needsReview)) this.startCheck();
    else this.nextScan();
  }

  /** Camera frames flow only while acquiring (live, or review in repeat mode). */
  private setCameraActive(active: boolean) {
    if (!this.camera) return;
    this.camera.setActive(active);
    if (active) void this.view.video.play().catch(() => {});
    else this.view.video.pause();
    this.monitor.reset();
  }

  /** Back to live acquisition; the next capture replaces this page's candidate results. */
  private rescan() {
    this.setCameraActive(true);
    if (this.session.busy) this.session.cancel();
    this.overlay = 'none';
    this.selected = null;
    this.expanded = false;
    this.dropBigCrops();
    this.cycle.setFrozen(false);
    this.cycle.rearm();
    this.monitor.reset();
    this.view.clearStill();
    this.render();
  }

  private nextPage() {
    this.session.nextPage();
    this.presenter.reset();
    this.rescan();
  }

  /** Save this page to the collection (idempotent per page; crops are read before any reset). */
  private async save(): Promise<void> {
    const page = this.session.page;
    const captureId = page.captureId;
    // Larger crop per card for viewing and re-checking later; from the still-held capture when not loaded yet.
    const display = async (id: string) => {
      const card = this.session.instance(id);
      if (!card) return null;
      const loaded = this.bigCrops.get(`${id}|${card.revision}`);
      if (loaded) return loaded;
      return captureId === null ? null : this.recognizer.displayCrop(captureId, card.quad);
    };
    const ok = await savePage(page, this.catalog, display);
    if (!ok) this.error = 'Could not save: browser storage is full or disabled.';
    this.session.markSaved();
    this.reloadCollection();
  }

  private touchCollection() {
    this.collection.rev++;
    this.render();
  }

  private closeSavedCard() {
    this.cardToken++;
    this.collection.card?.image?.close();
    this.collection.card = null;
    this.collection.rev++;
  }

  /** A saved card in the collection: show its crop and search the gallery again with the current bundle. */
  private async openSavedCard(pageKey: string, number: number) {
    this.closeSavedCard();
    const token = this.cardToken;
    const c: NonNullable<CollectionState['card']> = { pageKey, number, status: 'loading', hits: null, image: null, query: '', results: [], pick: null };
    this.collection.card = c;
    this.touchCollection();
    const crops = await getCrops(pageKey, number);
    if (token !== this.cardToken) return;
    if (!crops) { c.status = 'no-crop'; return this.touchCollection(); }
    try {
      c.image = await createImageBitmap(crops.display ?? crops.canonical);
      c.status = 'checking';
      this.touchCollection();
      const canonical = await createImageBitmap(crops.canonical, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
      const hits = await this.recognizer.search(canonical);
      if (token !== this.cardToken) return;
      c.hits = hits;
      c.status = 'ready';
    } catch (err) {
      if (token !== this.cardToken) return;
      trace('collection-recheck', { error: (err as Error).message });
      c.status = 'failed';
    }
    this.touchCollection();
  }

  /** The user's answer for a saved card: a catalog entry, or null for "not in the catalog". */
  private chooseSaved(identity: string | null, languageConfirmed: boolean) {
    const c = this.collection.card;
    if (!c) return;
    const page = loadPages().find((p) => p.key === c.pageKey);
    const saved = page?.cards.find((x) => x.number === c.number);
    if (!page || !saved) return;
    if (identity) {
      const m = userChoiceMatch(this.catalog, identity, c.hits ?? [], this.settings.preferredLanguage, languageConfirmed);
      const card = displayCard(m, this.catalog)!;
      const sameArtwork = saved.identity !== null && this.catalog.get(saved.identity)?.family === card.family;
      updateSavedCard(c.pageKey, c.number, {
        identity: card.id, name: card.name, cardId: card.cardId, state: 'confirmed', languageAssumed: m.languageAssumed, checkedByUser: true,
        versions: m.plausible.map((v) => v.id), preferredId: m.preferred?.id ?? null, checkVariant: false,
        // A different card has no scan-time price history.
        value: sameArtwork ? saved.value : null,
      });
    } else {
      updateSavedCard(c.pageKey, c.number, { identity: null, name: null, cardId: null, state: 'not-in-catalog', languageAssumed: false, checkedByUser: true,
        versions: [], preferredId: null, checkVariant: false, value: null });
    }
    c.pick = null;
    // Corrections become training examples, like decisions during scanning.
    if (saved.identity !== identity || languageConfirmed) {
      void getCrops(c.pageKey, c.number).then((crops) => crops && sendCollectionFeedback(
        { identity: saved.identity, number: saved.number, pageNumber: page.pageNumber }, crops.display ?? crops.canonical, c.hits, { identity, languageConfirmed }, this.manifest.version));
    }
    trace('collection-decision', { page: page.pageNumber, card: saved.number, identity });
    this.reloadCollection();
    this.touchCollection();
  }

  /** Saved pages, then their values from the current snapshot and market (scan-time values stay as history). */
  private reloadCollection(): void {
    const token = ++this.collectionToken;
    const pages = loadPages();
    this.collection.pages = pages;
    this.collection.rev++;
    void Promise.all([withCurrentPrices(pages, this.prices), this.prices.usdToEur()]).then(([priced, rate]) => {
      if (token !== this.collectionToken) return;
      this.collection.pages = priced;
      this.collection.usdToEur = rate;
      this.touchCollection();
    }).catch(() => { /* snapshot unavailable: scan-time values stay shown */ });
  }

  private toggleTheme() {
    const dark = this.view.isDark();
    this.changeSetting({ theme: dark ? 'light' : 'dark' });
  }

  private changeSetting(patch: Partial<Settings>) {
    Object.assign(this.settings, patch);
    saveSettings(this.settings);
    if (patch.theme) { applyTheme(patch.theme); this.view.themeChanged(); }
    if (patch.tolerance) this.monitor.tolerance = patch.tolerance;
    this.cycle.settings = { mode: this.settings.mode, dwellMs: this.settings.dwellMs, cooldownMs: this.settings.cooldownMs };
    if (patch.mode || patch.dwellMs || patch.tolerance) this.monitor.reset();
    if (patch.preferredLanguage) this.session.refreshPreference();
    if (patch.priceMarket) { this.session.reprice(); this.reloadCollection(); }
    this.render();
  }

  private onVisibility() {
    if (document.hidden) {
      // Hidden: invalidate pending work and release the camera.
      if (this.session.busy) this.session.cancel();
      this.cycle.pause();
      this.stopSampling();
      if (this.camera) { this.camera.stop(); this.camera = null; this.hiddenStop = true; }
    } else if (this.hiddenStop) {
      this.cycle.resume(performance.now());
      if (this.cycle.current === 'processing') this.cycle.rearm();
      this.render();
    }
  }

  render() {
    const snapshot: AppSnapshot = {
      page: this.session.page,
      runtime: this.recognizer.status,
      cyclePhase: this.cycle.current,
      countdownMs: this.countdownMs,
      stability: this.stability,
      settings: this.settings,
      camera: this.camera ? { label: this.camera.label, mock: this.camera.mock } : null,
      hiddenStop: this.hiddenStop,
      collection: this.collection,
      selected: this.selected,
      checking: this.check && this.selected ? { index: this.check.indexOf(this.selected), total: this.check.length } : null,
      viewerMode: this.viewerMode,
      compare: this.compare,
      bigCrop: this.selected ? this.bigCrops.get(`${this.selected}|${this.session.instance(this.selected)?.revision}`) ?? null : null,
      expanded: this.expanded,
      overlay: this.overlay,
      sweepAt: this.sweepAt,
      videos: this.videos,
      error: this.error,
      manifest: this.manifest,
      priceDescription: this.prices.description,
      now: performance.now(),
    };
    this.view.render(snapshot, this.catalog, this.presenter);
  }
}

void boot();
