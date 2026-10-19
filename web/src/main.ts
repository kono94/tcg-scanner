// App controller: camera -> stability -> capture -> worker inference -> page session -> view.
import './ui/styles.css';
import { installCrashTrace, trace } from './debug';
import { sendCollectionFeedback, sendFeedback, uploadCapture } from './feedback';
import { getCrops } from './collection/crops';
import { exportCollection, exportFileName, importCollection, parseCollection } from './collection/transfer';
import { activeScanSession, currentSession, deletePage, deleteSavedCard, deleteScanSession, finishScanSession, loadPages, loadScanSessions, nextCaptureNumber, renameScanSession, resumeScanSession, savedNeedsUser, savePage, startScanSession, updateSavedCard, withCurrentPrices } from './collection/store';
import { BatchSession } from './batch/session';
import { browserBatchStore } from './batch/storage';
import { listMockVideos, openCamera, openVideoMock, type CameraHandle, type MockVideo } from './camera/source';
import { CameraLifecycle, needsCameraFrames } from './camera/lifecycle';
import { CaptureCycle } from './capture/cycle';
import { halve, lumaFromRGBA, SceneChangeGate, sharpnessOf, StabilityMonitor, type StabilityState } from './capture/stability';
import { WorkerRecognizer } from './inference/client';
import type { BundleManifest } from './inference/protocol';
import type { Quad } from './vision/image';
import { SnapshotPriceService } from './prices/service';
import { Catalog, type CatalogCard } from './recognition/catalog';
import { displayCard, dominantGame, userChoiceMatch } from './recognition/policy';
import { PageSession } from './session/page';
import { visibleRegion } from './ui/geometry';
import { captureFeedback, primeCaptureFeedback } from './ui/haptics';
import { Presenter } from './ui/presenter';
import { loadSettings, saveSettings, type Settings } from './ui/settings';
import { needsReview, View, type AppSnapshot, type CollectionState, type ViewerMode } from './ui/view';

const BUNDLE = new URL('bundle/', document.baseURI).href;
/** Batch: how long the saved capture stays up with its outlines before the live preview returns. */
const BATCH_HOLD_MS = 1500;

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
    view.renderBoot(`Model bundle not found. Run scripts/release/build_browser_bundle.py first. (${(err as Error).message})`, true);
    return;
  }
  const recognizer = new WorkerRecognizer(BUNDLE, manifest, identities, settings.preferWebGPU);
  const prices = new SnapshotPriceService(`${BUNDLE}${manifest.prices}?v=${manifest.version}`, () => settings.priceMarket);
  const session = new PageSession(recognizer, { catalog, prices, preferredLanguage: () => settings.preferredLanguage });
  // Batch workflow: pages saved right away, their cards checked one at a time in the background.
  // Its queue is rebuilt from the saved pages, so it resumes after a reload or a killed tab.
  const batch = new BatchSession({
    detect: (id, bitmap) => recognizer.detect(id, bitmap),
    extract: (id, quad) => recognizer.extract(id, quad),
    release: (id) => recognizer.release(id, true),
    search: async (blob) => recognizer.search(await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' })),
    fingerprint: (blob) => recognizer.fingerprint(blob),
    failure: () => (recognizer.status.stage === 'failed' ? recognizer.status.detail : null),
  }, browserBatchStore, {
    catalog, prices, preferredLanguage: () => settings.preferredLanguage, bundleVersion: manifest.version,
    // Yield to the single-page session; start only once the models are ready.
    canRun: () => !session.busy && recognizer.status.stage === 'ready',
  });
  primeCaptureFeedback();
  const app = new App(view, session, recognizer, settings, catalog, manifest, prices, batch);
  let lastStage = '';
  recognizer.onStatus = () => {
    const st = recognizer.status;
    if (st.stage !== lastStage || st.fallbackReason) trace('runtime', { stage: st.stage, provider: st.provider, detail: st.detail, fallback: st.fallbackReason, loadMs: st.loadMs });
    lastStage = st.stage;
    batch.kick();
    app.render();
  };
  recognizer.onCrash = (message) => trace('worker-crash', { message });
  if (import.meta.env.DEV) {
    // Dev-only hook for parity checks of the in-browser detection path.
    (window as unknown as { tcgDebug: unknown }).tcgDebug = {
      ready: () => recognizer.whenReady(),
      runtime: () => recognizer.status,
      page: () => session.page,
      batch: () => batch.status(),
      recognizer,
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
  private readonly cameraLifecycle = new CameraLifecycle();
  get camera(): CameraHandle | null { return this.cameraLifecycle.current; }
  private sourceRequest = 0;
  private cameraOpening = false;
  private cameraReady = false;
  private cameraStartedAt = 0;
  private previewHash: number | null = null;
  private resumeHash: number | null = null;
  private nextNumber = nextCaptureNumber();
  private sessionChanging = false;
  private grabbing = false;
  private saving: Promise<boolean> | null = null;
  private readonly sourceVisitId = crypto.randomUUID();
  private captureProvenance: { captureId: string; sessionId: string; cameraMock: boolean } | null = null;
  private batchFrame: { width: number; height: number; number: number } | null = null;
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
  private overlay: 'none' | 'review' | 'settings' | 'collection' | 'start' | 'batch' = 'start';
  /** Batch mode: page-turn gate for auto capture, last stability luma sample, capture feedback. */
  private sceneGate = new SceneChangeGate();
  private lastLuma: Uint8Array | null = null;
  private awaitingChange = false;
  private toast: { text: string; until: number } | null = null;
  /** Batch: card outlines of the capture being saved, and its confirmation, held on the frozen photo. */
  private batchQuads: Quad[] | null = null;
  private batchSaved: { title: string; sub: string; repeat?: boolean } | null = null;
  private batchHoldUntil = 0;
  /** Batch: a page was just captured, so the pause before the next one means "turn the page". */
  private turnPage = false;
  /** Batch auto mode: automatic scanning runs only after Start, and stops on Pause or on leaving the scanner. */
  private batchRunning = false;
  /** Batch auto mode: stopped with Pause (not yet started is a different state); cleared on leaving the scanner. */
  private batchPaused = false;
  /** Batch: remaining page-turn window (from the capture cycle), for the countdown. */
  private turnLeftMs: number | null = null;
  private viewerSearch: { query: string; results: CatalogCard[] } = { query: '', results: [] };
  private uploadedCapture: number | null = null;
  private batchRender = 0;
  private collection: CollectionState = { pages: loadPages(), open: null, view: 'binders', filter: 'all', usdToEur: null, card: null, rev: 0 };
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
    private readonly batch: BatchSession,
  ) {
    this.monitor = new StabilityMonitor(settings.tolerance);
    this.cycle = new CaptureCycle(this.cycleSettings());
    this.session.page.pageNumber = nextCaptureNumber();
    batch.subscribe(() => this.onBatchChange());
    this.sampleCanvas.width = 128;
    this.sampleCanvas.height = 96;
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    this.presenter.options.reducedMotion = motion.matches;
    motion.addEventListener('change', () => { this.presenter.options.reducedMotion = motion.matches; });
    document.addEventListener('visibilitychange', () => this.onVisibility());
    window.addEventListener('pagehide', () => {
      this.batch.setPaused(true);
      if (this.session.busy) this.session.cancel();
      this.cycle.pause();
      this.hiddenStop = true;
      this.releaseCamera();
    });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { this.view.themeChanged(); this.render(); });
    window.addEventListener('resize', () => { this.view.redrawStill(); this.render(); });
    view.handlers = {
      primary: () => this.primary(),
      rescan: () => { if (!this.sessionChanging && !this.saving) this.rescan(); },
      captureNow: () => { if (this.settings.workflow === 'batch' && this.cycle.current === 'live') void this.captureBatch(true); },
      settings: () => { this.cancelCameraOpen(); this.overlay = 'settings'; this.render(); },
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
          if (!activeScanSession()) { void this.changeScanSession(null); return; }
          if (!this.camera) { void this.useSource(this.sourceFactory ?? openCamera); return; }
          this.overlay = 'none';
          this.cycle.setFrozen(false);
          if (this.cycle.current === 'live') this.setCameraActive(true);
        } else {
          this.cancelCameraOpen();
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
        const sessionId = this.collection.pages.find((p) => p.key === c.pageKey)?.sessionId;
        c.query = query;
        c.results = this.search(query, this.gameOf(this.collection.pages.filter((p) => p.sessionId === sessionId)));
        this.touchCollection();
      },
      viewerSearch: (query) => {
        this.viewerSearch = { query, results: this.search(query, this.expectedGame()) };
        this.render();
      },
      collectionView: (view, filter) => {
        this.view.handlers.tab('collection');
        this.collection.view = view;
        if (filter) this.collection.filter = filter;
        this.touchCollection();
      },
      savedChoose: (identity, languageConfirmed) => this.chooseSaved(identity, languageConfirmed),
      nextPage: () => { void this.nextScan(); },
      save: () => { void this.save(); },
      finishBatch: () => {
        this.overlay = 'batch';
        this.closeSavedCard();
        if (this.cycle.current === 'live') this.setCameraActive(false);
        this.cycle.setFrozen(true);
        this.reloadCollection();
        this.render();
      },
      scanMore: () => {
        this.closeSavedCard();
        if (!this.camera) { void this.useSource(openCamera); return; }
        this.overlay = 'none';
        this.cycle.setFrozen(false);
        if (this.cycle.current === 'live') this.setCameraActive(true);
        this.render();
      },
      doneBatch: () => {
        trace('batch-done', { ...this.batch.status(), active: undefined });
        if (this.grabbing || this.heldRepeat() || !this.batch.finish()) return;
        this.closeSavedCard();
        if (this.cycle.current === 'live') this.setCameraActive(false);
        this.cycle.setFrozen(true);
        this.overlay = 'collection';
        this.collection.open = null;
        this.reloadCollection();
        this.render();
      },
      useCamera: () => this.useSource(openCamera),
      useVideo: (v) => this.useSource(() => openVideoMock(v.url, v.name)),
      resume: () => this.resumeSource(),
      newSession: (name) => { void this.changeScanSession(null, name); },
      resumeSession: (id) => { void this.changeScanSession(id); },
      renameSession: (id, name) => {
        if (!renameScanSession(id, name)) this.error = 'Could not rename the binder.';
        this.render();
      },
      finishSession: () => { void this.endScanSession(); },
      deleteSession: (id) => { void this.deleteCollection(id); },
      exportCollection: () => { void this.exportCollection(); },
      importCollection: (file) => { void this.importCollection(file); },
      keepDuplicate: () => {
        if (!this.batch.keepDuplicate()) {
          this.error = 'Could not keep this capture. Browser storage may be full.';
          this.toast = { text: this.error, until: performance.now() + 5000 };
        } else this.error = null;
        this.reloadCollection();
      },
      discardDuplicate: () => { void this.batch.discardDuplicate().then(() => { this.error = null; this.render(); }); },
      changeSetting: (patch) => this.changeSetting(patch),
    };
  }

  /** Catalog search for manual picks, cards of the expected game first. */
  private search(query: string, game: string | null): CatalogCard[] {
    const results = this.catalog.search(query, this.settings.preferredLanguage);
    return game ? [...results].sort((a, b) => Number(b.game === game) - Number(a.game === game)) : results;
  }

  private gameOf(pages: readonly { cards: readonly { identity: string | null }[] }[]): string | null {
    return dominantGame(pages.flatMap((p) => p.cards.map((c) => (c.identity ? this.catalog.get(c.identity)?.game : null))));
  }

  /** The game of the open capture's matched cards, else of the binder's saved cards. */
  private expectedGame(): string | null {
    const page = this.session.page.instances.map((i) => (i.state === 'confirmed' && i.decision?.kind === 'matched' ? this.catalog.get(i.decision.top.identity)?.game : null));
    const id = activeScanSession()?.id;
    return dominantGame(page) ?? this.gameOf(this.collection.pages.filter((p) => p.sessionId === id));
  }

  /** Open (or close with null) the card viewer. */
  private open(id: string | null) {
    if (id === null) this.check = null;
    if (id !== this.selected) this.viewerSearch = { query: '', results: [] };
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
    if (this.settings.saveCaptures) void sendFeedback(card, crop, { identity, languageConfirmed },
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
    if (this.sessionChanging) return;
    this.sessionChanging = true;
    this.render();
    try {
      const page = this.session.page;
      const cards = page.instances.filter((i) => i.state !== 'back').length;
      if (page.instances.length && !await this.save()) return;
      if (cards) {
        this.toast = { text: `Page ${page.pageNumber} saved · ${cards} card${cards === 1 ? '' : 's'}`, until: performance.now() + 2600 };
        setTimeout(() => this.render(), 2700);
      }
      this.nextPage();
    } catch (err) { this.error = (err as Error).message; }
    finally { this.sessionChanging = false; this.render(); }
  }

  /** A held possible repeat is never discarded implicitly: the user keeps or dismisses it first. */
  private heldRepeat(): boolean {
    const held = this.batch.status().duplicate;
    if (held) {
      this.error = `Keep or dismiss the possible repeat of page ${held.matchedPageNumber} first.`;
      this.render();
    }
    return !!held;
  }

  /** Save the foreground revision before moving ownership to another scan session. */
  private async prepareSessionBoundary(): Promise<boolean> {
    const page = this.session.page;
    return !page.instances.length || page.savedRevision === page.revision || await this.save();
  }

  private async changeScanSession(id: string | null, name = '') {
    if (this.sessionChanging || this.saving || this.grabbing || this.session.busy || this.batch.status().capturing) return;
    if (id && id === activeScanSession()?.id) {
      if (!this.camera) await this.useSource(this.sourceFactory ?? openCamera);
      else { this.overlay = 'none'; this.render(); }
      return;
    }
    if (this.heldRepeat()) return;
    this.sessionChanging = true;
    this.render();
    let ready = false;
    try {
      if (!await this.prepareSessionBoundary()) return;
      if (id) {
        if (!resumeScanSession(id)) throw new Error('Could not continue the binder.');
      } else startScanSession(name);
      this.batch.finish();
      this.nextNumber = nextCaptureNumber();
      this.session.nextPage(this.nextNumber);
      this.presenter.reset();
      this.dropBigCrops();
      this.clearStill();
      this.selected = null;
      this.check = null;
      this.closeSavedCard();
      this.sceneGate.arm(null);
      this.awaitingChange = false;
      this.cycle.rearm();
      this.overlay = 'none';
      this.error = null;
      ready = true;
    } catch (err) { this.error = (err as Error).message; }
    finally { this.sessionChanging = false; this.render(); }
    if (ready && !this.camera && !document.hidden) await this.useSource(this.sourceFactory ?? openCamera);
  }

  private async endScanSession() {
    if (this.sessionChanging || this.saving || this.grabbing || this.session.busy || this.batch.status().capturing) return;
    if (this.heldRepeat()) return;
    this.sessionChanging = true;
    this.render();
    try {
      if (!await this.prepareSessionBoundary()) return;
      if (!finishScanSession()) throw new Error('Could not finish the binder.');
      this.batch.finish();
      this.overlay = 'start';
      this.releaseCamera();
      this.session.nextPage(1);
      this.presenter.reset();
      this.dropBigCrops();
      this.clearStill();
      this.selected = null;
      this.check = null;
      this.cycle.rearm();
      this.reloadCollection();
    } catch (err) { this.error = (err as Error).message; }
    finally { this.sessionChanging = false; this.render(); }
  }

  /** Hand the collection file to the share sheet where files can be shared (phones), otherwise download it. */
  private async exportCollection() {
    const now = new Date();
    const data = exportCollection(this.manifest.version, undefined, now);
    const name = exportFileName(now);
    const file = new File([JSON.stringify(data)], name, { type: 'application/json' });
    try {
      if (navigator.canShare?.({ files: [file] })) { await navigator.share({ files: [file], title: name }); return; }
    } catch (err) {
      if ((err as Error).name === 'AbortError') return; // the user closed the share sheet
    }
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
    const cards = data.pages.reduce((n, p) => n + p.cards.length, 0);
    this.toast = { text: `Exported ${data.pages.length} page${data.pages.length === 1 ? '' : 's'} · ${cards} card${cards === 1 ? '' : 's'}`, until: performance.now() + 2600 };
    this.render();
  }

  private async importCollection(file: File) {
    if (this.sessionChanging || this.saving) return;
    this.sessionChanging = true;
    this.render();
    try {
      const { addedPages, skippedPages, addedSessions } = importCollection(parseCollection(await file.text()));
      const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;
      const text = addedPages ? `Imported ${plural(addedPages, 'page')}` + (addedSessions ? ` · ${plural(addedSessions, 'new binder')}` : '') + (skippedPages ? ` · ${skippedPages} already here` : '')
        : 'Nothing new: these pages are already in this collection.';
      this.toast = { text, until: performance.now() + 4000 };
      this.error = null;
      this.reloadCollection();
    } catch (err) {
      this.error = (err as Error).message;
      this.toast = { text: this.error, until: performance.now() + 5000 };
    } finally { this.sessionChanging = false; this.render(); }
  }

  private async deleteCollection(id: string) {
    if (this.sessionChanging || this.saving || this.grabbing || this.session.busy || this.batch.status().capturing) return;
    this.sessionChanging = true;
    this.render();
    const active = activeScanSession()?.id === id;
    const batch = this.batch.status().active;
    try {
      if (!await deleteScanSession(id)) throw new Error('Could not delete the binder: browser storage is unavailable.');
      this.error = null;
      if (batch?.sessionId === id) { await this.batch.discardDuplicate(); this.batch.finish(); }
      this.batch.pruneDeleted();
      this.closeSavedCard();
      this.collection.open = null;
      if (active || this.session.page.scanSessionId === id) {
        this.releaseCamera();
        this.session.nextPage(1);
        this.presenter.reset();
        this.dropBigCrops();
        this.batchFrame = null;
        this.clearStill();
        this.selected = null;
        this.check = null;
        this.cycle.rearm();
      }
      this.reloadCollection();
    } catch (err) { this.error = (err as Error).message; }
    finally { this.sessionChanging = false; this.render(); }
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
      // Bounded: the open card and its neighbours; the oldest others are closed.
      const keep = this.selected ? new Set([this.selected]) : new Set<string>();
      for (const k of this.bigCrops.keys()) {
        if (this.bigCrops.size <= 4) break;
        if (k === key || keep.has(k.split('|')[0])) continue;
        this.bigCrops.get(k)!.close();
        this.bigCrops.delete(k);
      }
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
    this.reloadCollection(); // the collection value on Home, at today's prices
    this.render();
  }

  async useSource(factory: () => Promise<CameraHandle>) {
    if (this.camera && factory === this.sourceFactory) {
      this.overlay = 'none';
      this.cycle.resume(performance.now());
      this.render();
      return;
    }
    const request = ++this.sourceRequest;
    this.error = null;
    this.overlay = 'none';
    this.cameraOpening = true;
    this.view.detachStream();
    this.render();
    const started = performance.now();
    try {
      this.nextNumber = nextCaptureNumber(currentSession());
      if (this.session.page.status === 'empty') this.session.page.pageNumber = this.nextNumber;
      const camera = await this.cameraLifecycle.open(factory);
      if (!camera || request !== this.sourceRequest) return;
      this.sourceFactory = factory;
      this.hiddenStop = false;
      await this.view.attachStream(camera.stream, (n) => {
        this.frameId = n;
      });
      if (request !== this.sourceRequest) return;
      this.cycle.resume(performance.now());
      trace('camera', { label: camera.label, mock: camera.mock, openMs: Math.round(performance.now() - started) });
    } catch (err) {
      if (request !== this.sourceRequest) return;
      this.error = (err as Error).message;
      this.overlay = 'start';
    } finally {
      if (request === this.sourceRequest) { this.cameraOpening = false; this.render(); }
    }
  }

  private cancelCameraOpen() {
    ++this.sourceRequest;
    this.cameraOpening = false;
    this.cameraLifecycle.cancelOpen();
  }

  private releaseCamera() {
    this.cancelCameraOpen();
    this.stopSampling();
    this.cameraLifecycle.release();
    this.view.detachStream();
    this.countdownMs = null;
    this.cameraReady = false;
    this.monitor.reset();
  }

  private async resumeSource() {
    if (this.sourceFactory) await this.useSource(this.sourceFactory);
  }

  private clearStill() {
    this.batchFrame = null;
    this.batchQuads = null;
    this.batchSaved = null;
    this.view.clearStill();
  }

  /** Batch: back to the live preview once fresh pixels arrived and the saved capture was shown long enough. */
  private releaseBatchFrame() {
    const frame = this.batchFrame;
    const wait = this.batchHoldUntil - performance.now();
    if (!frame) return;
    if (wait > 0) {
      setTimeout(() => { if (this.batchFrame === frame && this.cameraReady) { this.clearStill(); this.render(); } }, wait);
      return;
    }
    this.clearStill();
  }

  private cameraFrameReady(via: string) {
    this.cameraReady = true;
    if (this.batchFrame) this.releaseBatchFrame();
    trace('camera-frame', { via, firstFrameMs: Math.round(performance.now() - this.cameraStartedAt) });
    this.render();
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
      if (!this.cameraLifecycle.active || video.readyState < 2) return;
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
      this.lastLuma = luma;
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
      // Some WebKit versions stop frame callbacks after resume. A nonblank pixel sample
      // provides a bounded fallback, rather than enabling capture on a stale/black video.
      const freshCallback = this.view.lastFrameAt >= this.cameraStartedAt;
      const freshPixels = this.resumeHash === null || hash !== this.resumeHash;
      if (!this.cameraReady && sum > 0 && (freshCallback || (freshPixels && (!this.view.hasFrameCallback || t - this.cameraStartedAt > 500)))) this.cameraFrameReady('pixels');
      this.previewHash = hash;
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
      const tick = this.cycle.tick(t, this.stability, this.sceneGate.update(luma, this.stability));
      if (!tick.view.awaitingChange && !tick.view.coolingDown) this.turnPage = false;
      this.awaitingChange = !!tick.view.awaitingChange;
      this.turnLeftMs = tick.view.coolingDown && this.turnPage ? tick.view.cooldownLeftMs ?? null : null;
      // Batch mode: admission control (backlog bound) before a capture is even taken.
      const admit = this.settings.workflow !== 'batch' || this.batch.canCapture();
      this.countdownMs = ready && admit && tick.view.armed ? tick.view.countdownMs : null;
      if (ready && this.cameraReady && admit && tick.capture && !this.session.busy) void this.capture();
      const now = `${this.countdownMs === null ? '-' : Math.ceil(this.countdownMs / 100)}|${this.stability.reason}|${this.recognizer.status.stage}|${this.awaitingChange}|${this.turnLeftMs === null ? '-' : Math.ceil(this.turnLeftMs / 100)}`;
      if (now !== shown) { shown = now; this.render(); }
    }, 100);
  }

  private stopSampling() {
    if (this.sampler) clearInterval(this.sampler);
    this.sampler = null;
  }

  /**
   * Capture exactly what the preview shows: the centre crop that object-fit: cover displays, cut
   * from the full-resolution frame. Then cap the size (a big frame is copied several times on the
   * way to the worker; phones have a tight per-tab memory limit).
   */
  private grab(video: HTMLVideoElement): Promise<ImageBitmap> {
    const [sx, sy, sw, sh] = visibleRegion(video, this.view.cameraBox());
    const longSide = Math.max(sw, sh), cap = isIOS() ? 2048 : 2560;
    const k = Math.min(1, cap / longSide);
    return createImageBitmap(video, sx, sy, sw, sh,
      k < 1 ? { resizeWidth: Math.round(sw * k), resizeHeight: Math.round(sh * k), resizeQuality: 'high' } : undefined);
  }

  /** What is alive on the UI thread (breadcrumbs for the memory investigation; WebKit has no heap API). */
  private memory(): Record<string, unknown> {
    const heap = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize;
    return {
      bigCrops: this.bigCrops.size, stillMP: +(this.view.stillPixels / 1e6).toFixed(2),
      cardCrops: this.session.page.instances.filter((i) => i.crop).length, collectionImage: !!this.collection.card?.image,
      batchWaiting: this.batch.status().waiting, ...(heap ? { jsHeapMB: Math.round(heap / 1e6) } : {}),
    };
  }

  /** Capture confirmation besides the flash (Settings → Capture feedback). */
  private feedback() {
    if (this.settings.captureFeedback !== 'off') captureFeedback();
  }

  private async capture(manual = false) {
    if (this.settings.workflow === 'batch') return this.captureBatch(manual);
    if (this.grabbing || this.sessionChanging || this.session.busy || !this.cameraReady || !this.cameraLifecycle.active || this.recognizer.status.stage !== 'ready') return;
    const video = this.view.video;
    if (video.readyState < 2) return;
    this.cycle.beginBatch();
    this.grabbing = true;
    const sourceRequest = this.sourceRequest;
    this.presenter.reset();
    this.selected = null;
    this.expanded = false;
    this.dropBigCrops();
    let full: ImageBitmap;
    try {
      full = await this.grab(video);
    } catch (err) {
      this.grabbing = false;
      this.cycle.settle(performance.now());
      this.cycle.rearm();
      this.error = (err as Error).message;
      this.render();
      return;
    }
    if (sourceRequest !== this.sourceRequest || document.hidden || this.overlay !== 'none') { this.grabbing = false; full.close(); this.cycle.rearm(); this.render(); return; }
    trace('capture', { video: `${video.videoWidth}x${video.videoHeight}`, capture: `${full.width}x${full.height}`, ...this.memory() });
    // The still frame is taken: no camera frames or steadiness checks until the user scans again.
    this.setCameraActive(false);
    // Display copy first, then hand the full-resolution capture to the worker.
    try { await this.view.setStill(full); }
    catch (err) {
      this.grabbing = false;
      full.close();
      this.cycle.rearm();
      this.error = (err as Error).message;
      this.render();
      return;
    }
    this.grabbing = false;
    if (sourceRequest !== this.sourceRequest || document.hidden || this.overlay !== 'none') {
      full.close(); this.clearStill(); this.cycle.rearm(); this.render(); return;
    }
    this.session.page.scanSessionId = currentSession();
    this.captureProvenance = { captureId: crypto.randomUUID(), sessionId: this.session.page.scanSessionId, cameraMock: !!this.camera?.mock };
    this.sweepAt = performance.now();
    this.feedback();
    const done = this.session.capture(full);
    this.loop();
    await done.catch((err) => { this.error = (err as Error).message; });
  }

  /**
   * Batch mode: detect, store each card's crops, release the capture, back to the live preview.
   * Camera delivery pauses during extraction; recognition continues from stored crops.
   */
  /** Manual captures hold a probable repeat for Keep anyway / Dismiss; automatic ones skip it. */
  private async captureBatch(manual = false) {
    if (this.grabbing || this.sessionChanging || !this.batch.canCapture() || this.session.busy || !this.cameraReady || !this.cameraLifecycle.active || this.recognizer.status.stage !== 'ready') return;
    const video = this.view.video;
    if (video.readyState < 2) return;
    this.cycle.beginBatch();
    this.grabbing = true;
    const sourceRequest = this.sourceRequest;
    const reference = this.lastLuma;
    let full: ImageBitmap;
    try {
      full = await this.grab(video);
    } catch (err) {
      this.grabbing = false;
      this.cycle.settle(performance.now());
      this.cycle.rearm();
      this.error = (err as Error).message;
      this.render();
      return;
    }
    if (sourceRequest !== this.sourceRequest || document.hidden || this.overlay !== 'none') { this.grabbing = false; full.close(); this.cycle.rearm(); this.render(); return; }
    // Copy the display image before muting the track: disabled tracks produce black frames.
    try { await this.view.setStill(full, 1024); }
    catch (err) { this.grabbing = false; full.close(); this.cycle.rearm(); this.error = (err as Error).message; this.render(); return; }
    this.grabbing = false;
    if (sourceRequest !== this.sourceRequest || document.hidden || this.overlay !== 'none') { full.close(); this.clearStill(); this.cycle.rearm(); this.render(); return; }
    this.batchFrame = { width: full.width, height: full.height, number: nextCaptureNumber() };
    this.batchQuads = null;
    this.batchSaved = null;
    this.turnPage = false;
    this.setCameraActive(false);
    this.captureProvenance = { captureId: crypto.randomUUID(), sessionId: currentSession(), cameraMock: !!this.camera?.mock };
    const size = { width: full.width, height: full.height };
    trace('capture', { workflow: 'batch', video: `${video.videoWidth}x${video.videoHeight}`, capture: `${size.width}x${size.height}`, ...this.memory() });
    this.sweepAt = performance.now();
    this.feedback();
    this.render();
    const started = performance.now();
    let text: string | null = null;
    let cards = 0, repeat = false;
    try {
      const nextPage = nextCaptureNumber();
      const r = await this.batch.capture(full, (captureId, det, quads) => {
        // Outline what was found on the frozen photo while the crops are stored.
        if (this.batchFrame && sourceRequest === this.sourceRequest) { this.batchQuads = quads; this.render(); }
        return this.uploadCapture(captureId, size, quads, det.boxes.map((b) => b.score), nextPage);
      }, { holdRepeat: manual });
      cards = r.cards;
      repeat = r.duplicate !== undefined;
      if (r.skipped && this.batchFrame) {
        this.batchSaved = { title: 'Already scanned', sub: `Same as page ${r.duplicate} · not saved again · turn the page`, repeat: true };
        this.batchHoldUntil = performance.now() + BATCH_HOLD_MS;
      } else if (r.skipped) text = `Already scanned (page ${r.duplicate}) · turn the page`;
      else if (r.duplicate) text = `Possible repeat of page ${r.duplicate} · not saved`;
      else if (!r.cards) text = 'No cards found · scan again';
      else if (this.batchFrame) {
        this.batchSaved = { title: `Page ${r.pageNumber} saved`, sub: `${r.cards} card${r.cards === 1 ? '' : 's'}${r.failed ? ` · ${r.failed} not stored` : ''} · turn the page` };
        this.batchHoldUntil = performance.now() + BATCH_HOLD_MS;
      } else text = `Page ${r.pageNumber} saved · ${r.cards} card${r.cards === 1 ? '' : 's'}`;
      trace('batch-page', { page: r.pageNumber, cards: r.cards, overLimit: r.overLimit, failed: r.failed, ms: Math.round(performance.now() - started), ...this.memory() });
    } catch (err) {
      text = 'Capture not saved';
      this.error = (err as Error).message;
      trace('batch-page', { error: this.error });
    }
    if (text) {
      this.toast = { text, until: performance.now() + 2600 };
      setTimeout(() => this.render(), 2700);
    }
    this.cycle.settle(performance.now());
    // Page-turn window: starts once the saved photo is no longer shown; the steady dwell follows it.
    // Nothing saved (no cards, failure): only a short pause before trying the same view again.
    const now = performance.now();
    this.cycle.rearm(cards || repeat ? Math.max(now, this.batchHoldUntil) + this.settings.batchCooldownMs : now + 1500);
    // Auto mode: the same page would be captured again; wait for the page to be turned.
    if (cards || repeat) {
      this.turnPage = true;
      // A page may turn while delivery is suspended: a fresh steady view must still
      // differ from the captured reference, even if the motion itself was not sampled.
      this.sceneGate.arm(reference, true);
      this.cycle.requireSceneChange();
    }
    this.render();
  }

  /** "Share scans to improve recognition": the held capture as JPEG plus its detections; skipped without cards. */
  private async uploadCapture(captureId: number, size: { width: number; height: number }, quads: Quad[], scores: number[], page: number | null) {
    if (!this.settings.saveCaptures || !quads.length || this.uploadedCapture === captureId) return;
    this.uploadedCapture = captureId;
    const provenance = this.captureProvenance;
    const workflow = this.settings.workflow;
    try {
      const jpeg = await this.recognizer.encodeCapture(captureId, 0.9);
      void uploadCapture(jpeg, { captureSize: size, quads: quads.map((q) => q.map((p) => [+p[0].toFixed(1), +p[1].toFixed(1)])), scores: scores.map((x) => +x.toFixed(3)),
        bundleVersion: this.manifest.version, cameraMock: provenance?.cameraMock ?? !!this.camera?.mock, workflow, page,
        captureId: provenance?.captureId ?? String(captureId), sessionId: provenance?.sessionId ?? currentSession(),
        sourceVisitId: this.sourceVisitId, userAgent: navigator.userAgent, provider: this.recognizer.status.provider });
    } catch (err) {
      trace('capture-upload', { error: (err as Error).message });
    }
  }

  private onBatchChange() {
    this.nextNumber = nextCaptureNumber();
    // The Collection is the batch's store: refresh what is shown (throttled; results arrive per card).
    if (this.overlay === 'batch' || this.overlay === 'collection') {
      cancelAnimationFrame(this.batchRender);
      this.batchRender = requestAnimationFrame(() => this.reloadCollection());
    }
    this.render();
  }

  onSessionChange() {
    const page = this.session.page;
    this.batch.kick();
    if (page.status !== 'detecting' && page.status !== 'empty' && page.captureId !== null && page.captureSize && this.settings.saveCaptures) {
      void this.uploadCapture(page.captureId, page.captureSize, page.instances.map((i) => i.quad), page.instances.map((i) => i.box.score), page.pageNumber);
    }
    const status = `${page.generation}|${page.revision}|${page.status}`;
    const justSettled = status !== this.lastStatus && page.status === 'settled' && this.lastStatus.endsWith('processing');
    if (status !== this.lastStatus) trace('page', { status: page.status, cards: page.instances.length, detectMs: page.detectMs && Math.round(page.detectMs), error: page.error });
    this.lastStatus = status;
    if ((page.status === 'settled' || page.status === 'failed') && this.cycle.current === 'processing') {
      this.cycle.settle(performance.now());
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
      if (this.overlay === 'none' && this.presenter.animating) this.raf = requestAnimationFrame(frame);
    };
    this.raf = requestAnimationFrame(frame);
  }

  private primary() {
    const status = this.session.page.status;
    if (!this.camera) { this.overlay = 'start'; this.render(); return; }
    if (this.settings.workflow === 'batch') {
      if (this.settings.mode === 'manual') { if (this.cycle.current === 'live') void this.captureBatch(true); return; }
      if (this.batchRunning) { this.batchRunning = false; this.batchPaused = true; }
      else if (this.cycle.current === 'live') {
        this.batchPaused = false;
        // Start: the page in view now is the one to scan (no page-turn gate), after the steady dwell.
        this.batchRunning = true;
        this.turnPage = false;
        this.sceneGate.arm(null);
        this.cycle.rearm();
        this.monitor.reset();
      }
      this.render();
      return;
    }
    if (this.cycle.current === 'live') {
      if (this.cycle.canCaptureManually()) void this.capture(true);
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
    if (!this.cameraLifecycle.setActive(active)) return;
    this.cameraReady = false;
    if (active) { this.cameraStartedAt = performance.now(); this.resumeHash = this.previewHash; }
    this.view.setStreamActive(active);
    this.monitor.reset();
    this.stability = { steady: false, steadyMs: 0, motion: null, sharp: false, reason: 'warming-up' };
    this.countdownMs = null;
    if (active) this.startSampling(); else this.stopSampling();
    trace('camera-active', { active });
  }

  /** Back to live acquisition; the next capture replaces this page's candidate results. */
  private rescan() {
    if (this.saving) return;
    if (this.session.busy) this.session.cancel();
    this.overlay = 'none';
    this.selected = null;
    this.expanded = false;
    this.dropBigCrops();
    this.cycle.setFrozen(false);
    this.cycle.rearm();
    this.monitor.reset();
    if (this.session.page.status === 'empty') this.session.page.pageNumber = nextCaptureNumber();
    this.clearStill();
    this.render();
  }

  private nextPage() {
    this.nextNumber = nextCaptureNumber();
    this.session.nextPage(this.nextNumber);
    this.presenter.reset();
    this.rescan();
  }

  /** Save this page to the collection (idempotent per page; crops are read before any reset). */
  private save(): Promise<boolean> {
    if (this.saving) return this.saving;
    this.saving = this.saveRevision().catch((err) => { this.error = (err as Error).message; return false; })
      .finally(() => { this.saving = null; this.render(); });
    this.render();
    return this.saving;
  }

  private async saveRevision(): Promise<boolean> {
    const page = this.session.page;
    const captureId = page.captureId;
    // Larger crop per card for viewing and re-checking later; from the still-held capture when not loaded yet.
    // Crops made only for saving are closed right after encoding (they used to stay alive: 2.2 MB each).
    const made: ImageBitmap[] = [];
    const display = async (id: string) => {
      const card = this.session.instance(id);
      if (!card) return null;
      const loaded = this.bigCrops.get(`${id}|${card.revision}`);
      if (loaded) return loaded;
      if (captureId === null) return null;
      const bmp = await this.recognizer.displayCrop(captureId, card.quad);
      made.push(bmp);
      return bmp;
    };
    let ok: boolean;
    try {
      ok = await savePage(page, this.catalog, display);
    } finally {
      for (const b of made) b.close();
    }
    if (!ok) this.error = 'Could not save: browser storage is full or disabled.';
    else if (this.session.page.generation === page.generation && this.session.page.revision === page.revision) this.session.markSaved();
    this.reloadCollection();
    return ok;
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
      if (token !== this.cardToken) { c.image.close(); c.image = null; return; }
      // Batch results keep their closest matches: no second embedding when the bundle is the same.
      const saved = loadPages().find((p) => p.key === pageKey)?.cards.find((x) => x.number === number);
      if (saved?.hits?.length && saved.bundle === this.manifest.version) {
        c.hits = saved.hits.map((h, row) => ({ identity: h.identity, score: h.score, row }));
        c.status = 'ready';
        return this.touchCollection();
      }
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
    if (this.settings.saveCaptures && (saved.identity !== identity || languageConfirmed)) {
      void getCrops(c.pageKey, c.number).then((crops) => crops && sendCollectionFeedback(
        { identity: saved.identity, number: saved.number, pageNumber: page.pageNumber }, crops.display ?? crops.canonical, c.hits, { identity, languageConfirmed }, this.manifest.version));
    }
    trace('collection-decision', { page: page.pageNumber, card: saved.number, identity });
    this.reloadCollection();
    // Batch review: straight on to the next card that needs the user.
    if (this.overlay === 'batch') {
      const next = this.batch.pages().flatMap((p) => p.cards.filter(savedNeedsUser).map((x) => [p.key, x.number] as const))
        .find(([k, n]) => !(k === c.pageKey && n === c.number));
      if (next) { void this.openSavedCard(next[0], next[1]); return; }
      this.closeSavedCard();
    }
    this.touchCollection();
  }

  /** Saved pages, then their values from the current snapshot and market (scan-time values stay as history). */
  private reloadCollection(): void {
    this.nextNumber = nextCaptureNumber();
    const token = ++this.collectionToken;
    this.batch.invalidate();
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
    this.cycle.settings = this.cycleSettings();
    if (patch.workflow) void this.switchWorkflow();
    if (patch.mode || patch.dwellMs || patch.tolerance) this.monitor.reset();
    if (patch.preferredLanguage) this.session.refreshPreference();
    if (patch.priceMarket) { this.session.reprice(); this.reloadCollection(); }
    this.render();
  }

  /** Batch mode has no "repeat" (every capture is a new page): it scans like auto-when-steady. */
  private cycleSettings() {
    const mode = this.settings.workflow === 'batch' && this.settings.mode === 'repeat' ? 'auto' : this.settings.mode;
    return { mode, dwellMs: this.settings.dwellMs, cooldownMs: this.settings.cooldownMs };
  }

  /** Switching workflows: a single page under review is saved first (as "Next scan" would); then live. */
  private async switchWorkflow() {
    if (this.sessionChanging) return;
    this.sessionChanging = true;
    this.render();
    this.sceneGate.arm(null);
    try {
      if (this.session.busy) this.session.cancel();
      if (!await this.prepareSessionBoundary()) return;
      this.session.nextPage(nextCaptureNumber());
      this.presenter.reset();
      this.dropBigCrops();
      this.clearStill();
      this.cycle.rearm();
    } finally { this.sessionChanging = false; this.render(); }
  }

  private onVisibility() {
    this.batch.setPaused(document.hidden);
    if (document.hidden) {
      // Hidden: invalidate pending work and release the camera.
      if (this.session.busy) this.session.cancel();
      this.cycle.pause();
      this.stopSampling();
      if (this.camera || this.cameraOpening) { this.hiddenStop = true; this.releaseCamera(); }
    } else if (this.hiddenStop) {
      this.cycle.resume(performance.now());
      if (this.cycle.current === 'processing') this.cycle.rearm();
      this.render();
    }
  }

  render() {
    const scannerVisible = this.overlay === 'none' && !this.sessionChanging;
    // Batch auto scanning never resumes by itself: leaving the scanner or losing the camera stops it.
    if (!scannerVisible || !this.camera || this.settings.workflow !== 'batch') { this.batchRunning = false; this.batchPaused = false; }
    const batchIdle = this.settings.workflow === 'batch' && this.settings.mode !== 'manual' && !this.batchRunning;
    if (batchIdle) { this.turnLeftMs = null; this.awaitingChange = false; }
    this.cycle.setFrozen(!scannerVisible || this.selected !== null || batchIdle);
    this.setCameraActive(needsCameraFrames({ hidden: document.hidden, scannerVisible, selected: this.selected !== null,
      phase: this.cycle.current, mode: this.settings.workflow === 'batch' ? 'auto' : this.settings.mode }) || (this.grabbing && scannerVisible && !document.hidden));
    const snapshot: AppSnapshot = {
      scanSession: activeScanSession(),
      scanSessions: loadScanSessions(),
      captureNumber: this.settings.workflow === 'batch' || this.session.page.status === 'empty' ? this.nextNumber : this.session.page.pageNumber,
      sessionChanging: this.sessionChanging || !!this.saving || this.session.busy || this.batch.status().capturing || this.grabbing,
      cameraOpening: this.cameraOpening,
      cameraReady: this.cameraReady,
      batchFrame: this.batchFrame,
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
      batch: this.settings.workflow === 'batch' || this.overlay === 'batch' ? this.batch.status() : null,
      toast: this.toast && this.toast.until > performance.now() ? this.toast.text : null,
      awaitingChange: this.awaitingChange,
      batchRunning: this.batchRunning,
      batchPaused: this.batchPaused,
      turn: this.turnLeftMs !== null ? { leftMs: this.turnLeftMs, totalMs: this.settings.batchCooldownMs } : null,
      batchQuads: this.batchQuads,
      batchSaved: this.batchSaved,
      viewerSearch: this.viewerSearch,
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
