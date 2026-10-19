// DOM rendering for the Darkroom scanner. Consumes immutable state snapshots; holds no
// recognition logic. Semantic states drive classes; text never relies on color alone.
import type { MockVideo } from '../camera/source';
import type { CyclePhase } from '../capture/cycle';
import type { StabilityState } from '../capture/stability';
import type { RuntimeStatus } from '../inference/client';
import type { BundleManifest } from '../inference/protocol';
import { Catalog, languageName, type CatalogCard } from '../recognition/catalog';
import type { Hit } from '../recognition/gallery';
import { candidates, displayCard, dominantGame, type Candidate, type MatchResult } from '../recognition/policy';
import { formatMoney, formatQuote, formatRange, type ValueSummary } from '../recognition/value';
import { pageCounts, type AttentionReason, type CardInstance, type PageState } from '../session/page';
import type { Quad } from '../vision/image';
import { bracketPath, centroid, mapQuad, PREVIEW_ZOOM, quadMatrix3d, rectQuad, zoomedFit, type ContainFit } from './geometry';
import type { Geometry, Presenter } from './presenter';
import type { Settings } from './settings';
import { languageMatters, savedNeedsUser, scanAmountIn, totals, type SavedCard, type SavedPage, type ScanSession, type Totals } from '../collection/store';
import type { BatchStatus } from '../batch/session';
import { confirmDialog, promptDialog } from './dialog';

export interface AppSnapshot {
  scanSession: ScanSession | null;
  scanSessions: ScanSession[];
  captureNumber: number;
  sessionChanging: boolean;
  cameraOpening: boolean;
  cameraReady: boolean;
  batchFrame: { width: number; height: number; number: number } | null;
  page: PageState;
  runtime: RuntimeStatus;
  cyclePhase: CyclePhase;
  countdownMs: number | null;
  stability: StabilityState;
  settings: Settings;
  camera: { label: string; mock: boolean } | null;
  hiddenStop: boolean;
  /** Card open in the viewer (also highlighted in the photo and strip). */
  selected: string | null;
  /** Position in the guided check, when it is running. */
  checking: { index: number; total: number } | null;
  viewerMode: ViewerMode;
  /** Candidate artwork family shown side by side in compare mode. */
  compare: string | null;
  /** Large display crop for the open card, when loaded. */
  bigCrop: ImageBitmap | null;
  expanded: boolean;
  overlay: 'none' | 'review' | 'settings' | 'collection' | 'start' | 'batch';
  /** Batch workflow status (null in single-page mode). */
  batch: BatchStatus | null;
  /** Short capture feedback over the live preview (batch mode), e.g. "Page 3 saved · 9 cards". */
  toast: string | null;
  /** Batch auto mode: waiting for the next page before scanning automatically. */
  awaitingChange: boolean;
  /** Batch auto mode: automatic scanning was started with Start (it never starts by itself). */
  batchRunning: boolean;
  /** Batch auto mode: stopped with Pause scanning; Continue scanning resumes. */
  batchPaused: boolean;
  /** Batch auto mode: the page-turn window after a capture, while it runs. */
  turn: { leftMs: number; totalMs: number } | null;
  /** Batch: cards found in the frozen capture (capture pixels), outlined while it is saved. */
  batchQuads: Quad[] | null;
  /** Batch: confirmation shown on the frozen capture after it was saved (instead of "Starting camera"). */
  batchSaved: { title: string; sub: string; repeat?: boolean } | null;
  /** Catalog search inside the card viewer. */
  viewerSearch: { query: string; results: CatalogCard[] };
  /** Saved pages for the Collection tab, and the one opened there. */
  collection: CollectionState;
  sweepAt: number;
  videos: MockVideo[];
  error: string | null;
  manifest: BundleManifest;
  priceDescription: string;
  now: number;
}

export type ViewerMode = 'card' | 'photo';

export interface CollectionState {
  pages: SavedPage[];
  open: string | null;
  /** Binders (grouped by binder and page) or every card sorted by value. */
  view: 'binders' | 'cards';
  /** All-cards view: every card, or only those that still need a check. */
  filter: 'all' | 'check';
  usdToEur: number | null;
  /** A saved card opened for checking again. */
  card: {
    pageKey: string;
    number: number;
    status: 'loading' | 'checking' | 'ready' | 'no-crop' | 'failed';
    hits: Hit[] | null;
    image: ImageBitmap | null;
    query: string;
    results: CatalogCard[];
    /** Candidate tapped for a closer look before using it. */
    pick: string | null;
  } | null;
  /** Bumped on every change so the screen redraws. */
  rev: number;
}

export interface Handlers {
  primary(): void;
  rescan(): void;
  /** Batch auto scanning: a manual capture now (bypasses the dwell, not the checks). */
  captureNow(): void;
  settings(): void;
  theme(): void;
  select(id: string | null): void;
  viewerMode(mode: ViewerMode): void;
  compare(family: string | null): void;
  step(delta: number): void;
  choose(id: string, identity: string, languageConfirmed: boolean, advance: boolean): void;
  chooseNone(id: string): void;
  expand(on: boolean): void;
  retry(id: string): void;
  skip(id: string): void;
  closeOverlay(): void;
  review(): void;
  startCheck(): void;
  tab(name: 'home' | 'scan' | 'collection' | 'settings'): void;
  openSaved(key: string | null): void;
  deleteSaved(key: string): void;
  deleteSavedCard(key: string, number: number): void;
  openSavedCard(key: string, number: number | null): void;
  savedPick(identity: string | null): void;
  savedSearch(query: string): void;
  /** null: not in the catalog. */
  savedChoose(identity: string | null, languageConfirmed: boolean): void;
  nextPage(): void;
  save(): void;
  /** Batch mode: open the review of the batch, go back to scanning, or close the batch. */
  finishBatch(): void;
  scanMore(): void;
  doneBatch(): void;
  useCamera(): void;
  useVideo(v: MockVideo): void;
  resume(): void;
  newSession(name: string): void;
  resumeSession(id: string): void;
  renameSession(id: string, name: string): void;
  finishSession(): void;
  deleteSession(id: string): void;
  /** Save the whole collection as a file; merge an exported file into this browser. */
  exportCollection(): void;
  importCollection(file: File): void;
  keepDuplicate(): void;
  discardDuplicate(): void;
  changeSetting(patch: Partial<Settings>): void;
  collectionView(view: CollectionState['view'], filter?: CollectionState['filter']): void;
  viewerSearch(query: string): void;
}

const SVG = 'http://www.w3.org/2000/svg';
const REASON: Record<AttentionReason, string> = {
  glare: 'Glare', 'move-closer': 'Move closer', 'not-found': 'Not found', 'match-unclear': 'Match unclear', timeout: 'Took too long', error: 'Error',
};
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = '', text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** Set innerHTML only when it changed; rewriting identical markup restarts CSS animations (flicker). */
function setHtml(node: Element, html: string) {
  if ((node as HTMLElement).dataset.html !== html) {
    node.innerHTML = html;
    (node as HTMLElement).dataset.html = html;
  }
}

function setText(node: Element, text: string) {
  if (node.textContent !== text) node.textContent = text;
}

function icon(path: string, size = 18): string {
  return `<svg width="${size}" height="${size}" viewBox="0 0 18 18" aria-hidden="true">${path}</svg>`;
}

const ICONS = {
  gear: icon('<path d="M3 5h12M3 13h12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="7" cy="5" r="1.9" fill="var(--bg)" stroke="currentColor" stroke-width="1.5"/><circle cx="11.5" cy="13" r="1.9" fill="var(--bg)" stroke="currentColor" stroke-width="1.5"/>'),
  sun: icon('<circle cx="9" cy="9" r="3.2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M9 1.8v1.8M9 14.4v1.8M1.8 9h1.8M14.4 9h1.8M3.9 3.9l1.3 1.3M12.8 12.8l1.3 1.3M3.9 14.1l1.3-1.3M12.8 5.2l1.3-1.3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>'),
  moon: icon('<path d="M14.5 11.2A6 6 0 0 1 6.8 3.5a6 6 0 1 0 7.7 7.7z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>'),
  list: icon('<path d="M6 5h9M6 9h9M6 13h9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="3" cy="5" r="1" fill="currentColor"/><circle cx="3" cy="9" r="1" fill="currentColor"/><circle cx="3" cy="13" r="1" fill="currentColor"/>'),
  scan: icon('<path d="M2.5 6V3.5a1 1 0 0 1 1-1H6M12 2.5h2.5a1 1 0 0 1 1 1V6M15.5 12v2.5a1 1 0 0 1-1 1H12M6 15.5H3.5a1 1 0 0 1-1-1V12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><rect x="6" y="5" width="6" height="8" rx="1" fill="none" stroke="currentColor" stroke-width="1.5"/>'),
  stack: icon('<rect x="3" y="5" width="9" height="11" rx="1.2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M6 2.5h7.5a1 1 0 0 1 1 1V13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>'),
  shutter: icon('<circle cx="9" cy="9" r="6.5" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="9" cy="9" r="3.6" fill="currentColor"/>'),
  rescan: icon('<path d="M3.5 9a5.5 5.5 0 1 0 1.7-4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M3.5 2.5v3h3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'),
  expand: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  shrink: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 2v4H2M14 6h-4V2M10 14v-4h4M2 10h4v4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  prev: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  next: '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="m6 3 5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  close: '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M2 2l10 10M12 2 2 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
  back: '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M8 2 4 6l4 4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  cue: '<svg class="cue-i" width="12" height="12" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 1 9 5 5 9 1 5z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>',
  att: '<svg class="att-i" width="13" height="12" viewBox="0 0 11 10" aria-hidden="true"><path d="M5.5 1 10 9H1z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/><path d="M5.5 4v2.2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>',
};

function drawInto(canvas: HTMLCanvasElement, bitmap: ImageBitmap | null, w: number, h: number) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d')!;
  if (bitmap) ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
}

/**
 * Swipe left to reveal a Delete button (touch and mouse). A swipe never counts as a tap on the row.
 */
function swipeToDelete(content: HTMLElement, label: string, onDelete: () => void): HTMLElement {
  const wrap = el('div', 'swipe');
  const del = el('button', 'swipedel', 'Delete');
  del.setAttribute('aria-label', label);
  del.tabIndex = -1;
  del.onclick = (e) => { e.stopPropagation(); onDelete(); };
  wrap.append(del, content);
  const OPEN = -88;
  let startX = 0, startY = 0, base = 0, dx = 0, tracking = false, swiping = false, moved = false;
  const set = (x: number, animate: boolean) => {
    content.style.transition = animate ? 'transform 160ms ease-out' : 'none';
    content.style.transform = x ? `translateX(${x}px)` : '';
    wrap.classList.toggle('open', x !== 0);
    del.tabIndex = x ? 0 : -1;
  };
  content.addEventListener('pointerdown', (e) => {
    tracking = true; swiping = false; moved = false;
    startX = e.clientX; startY = e.clientY;
    base = wrap.classList.contains('open') ? OPEN : 0;
    dx = base;
  });
  content.addEventListener('pointermove', (e) => {
    if (!tracking) return;
    const mx = e.clientX - startX, my = e.clientY - startY;
    if (!swiping) {
      if (Math.abs(mx) < 8 || Math.abs(mx) < Math.abs(my)) { if (Math.abs(my) > 10) tracking = false; return; }
      swiping = true; moved = true;
      content.setPointerCapture(e.pointerId);
    }
    dx = Math.min(0, Math.max(OPEN * 1.4, base + mx));
    set(dx, false);
  });
  const end = () => {
    if (!tracking) return;
    tracking = false;
    if (swiping) set(dx < OPEN / 2 ? OPEN : 0, true);
  };
  content.addEventListener('pointerup', end);
  content.addEventListener('pointercancel', end);
  // A finished swipe or a tap on an open row only closes it.
  content.addEventListener('click', (e) => {
    if (moved || (wrap.classList.contains('open') && !swiping)) {
      e.stopImmediatePropagation(); e.preventDefault();
      if (!moved) set(0, true);
    }
    moved = false;
  }, { capture: true });
  return wrap;
}

function cropCanvas(bitmap: ImageBitmap | null, w: number, h: number, cls = ''): HTMLCanvasElement {
  const c = el('canvas', cls);
  drawInto(c, bitmap, w, h);
  return c;
}

/** Cards the review walk-through visits: unresolved, or a matched card flagged to check. */
export function needsReview(card: CardInstance): boolean {
  if (card.userChoice) return false;
  return card.state === 'needs-attention' || (card.state === 'confirmed' && !!card.value?.cue && card.value.cue.kind === 'check-variant');
}

export function isCue(card: CardInstance): boolean {
  // A fully unpriced card is "Value unknown", not a variant to double-check.
  return card.state === 'confirmed' && (!!card.value?.cue || (!!card.value?.rangeIncomplete && card.value.priced > 0));
}

/** Short status of a saved card (Collection and batch review). */
export function savedLabel(c: SavedCard): string {
  if (c.state === 'queued') return 'Waiting to check';
  if (c.state === 'needs-attention') return REASON[c.reason as AttentionReason] ?? 'Needs attention';
  if (c.state === 'confirmed' && c.checkVariant && !c.checkedByUser) return 'Check variant';
  if (c.state === 'skipped') return 'Not identified';
  if (c.state === 'not-in-catalog') return 'Not in the catalog';
  return c.name ?? 'Unresolved';
}

export function stateLabel(card: CardInstance): string {
  switch (card.state) {
    case 'queued': return 'Waiting to check';
    case 'recognizing': return 'Checking card';
    case 'verifying': return 'Confirming match';
    case 'skipped': return 'Decided later';
    case 'back': return 'Card back · not counted';
    case 'needs-attention': return 'Needs attention · ' + REASON[card.reason ?? 'error'];
    case 'confirmed': {
      const m = card.decision as MatchResult;
      const parts = ['Matched'];
      if (m.languageAssumed) parts.push(`${languageName(m.preferredLanguage)} assumed`);
      return parts.join(' · ');
    }
  }
}

export class View {
  handlers!: Handlers;
  readonly video: HTMLVideoElement;
  hasFrameCallback = false;
  /** performance.now() of the last presented-frame callback. */
  lastFrameAt = 0;
  private readonly app: HTMLElement;
  private readonly title = el('h1', 'title');
  private readonly sub = el('div', 'sub');
  private readonly cam = el('div', 'cam');
  private readonly still = el('canvas', 'still');
  private readonly svg = document.createElementNS(SVG, 'svg');
  private readonly hits = el('div');
  private readonly chip = el('div', 'chip');
  private readonly fx = el('div');
  private readonly diag = el('div', 'diag');
  private readonly expandBtn = el('button', 'expand');
  private camLayout = '';
  private readonly homeBtn = el('button', 'homebtn');
  private overlayKey = '';
  private stillZoom = PREVIEW_ZOOM;
  private viewerOpen = false;
  private dark: boolean | null = null;
  private readonly loupe = el('div', 'loupe');
  private readonly strip = el('div', 'strip');
  private readonly rescanBtn = el('button', 'icon');
  private readonly settingsBtn = el('button', 'icon');
  private readonly themeBtn = el('button', 'icon');
  private readonly primaryBtn = el('button', 'primary');
  private readonly flights = el('div');
  private readonly layer = el('div');
  private readonly announce = el('div', 'sr');
  private readonly batchBar = el('div', 'batchbar');
  private readonly duplicateBar = el('div', 'duplicatebar');
  private readonly toast = el('div', 'toast');
  /** Batch: large instruction over the live preview ("Turn the page", "Hold steady") with its timer. */
  private readonly cue = el('div', 'cuebox');
  private stillBitmap: ImageBitmap | null = null;
  private fit: ContainFit | null = null;
  private lastSweep = 0;
  private loupeKey = '';
  private stripKey = '';
  private layerKey = '';
  private slotEls = new Map<string, HTMLButtonElement>();
  private flightEls = new Map<string, HTMLDivElement>();
  private lastAnnounce = '';
  private frameCallback = 0;
  private onFrame: ((n: number) => void) | null = null;

  constructor(root: HTMLElement) {
    this.app = root;
    root.textContent = '';
    const top = el('div', 'top');
    top.append(this.title, this.sub);
    this.sub.setAttribute('aria-live', 'off');
    this.video = el('video');
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.autoplay = true;
    this.svg.classList.add('overlay');
    this.svg.setAttribute('aria-hidden', 'true');
    this.diag.hidden = true;
    this.expandBtn.innerHTML = ICONS.expand;
    this.expandBtn.onclick = () => this.handlers.expand(!this.cam.classList.contains('expanded'));
    // The processing panel floats over the photo so it never takes layout space.
    this.cam.append(this.video, this.still, this.svg, this.hits, this.chip, this.fx, this.diag, this.expandBtn, this.loupe, this.toast, this.cue);
    this.cue.setAttribute('aria-hidden', 'true'); // the title above announces the same
    this.toast.setAttribute('role', 'status');
    this.loupe.setAttribute('aria-hidden', 'true');
    const bot = el('div', 'bot');
    const actions = el('div', 'actions');
    this.settingsBtn.innerHTML = ICONS.gear;
    this.settingsBtn.setAttribute('aria-label', 'Settings');
    this.rescanBtn.innerHTML = ICONS.rescan;
    this.rescanBtn.setAttribute('aria-label', 'Scan again');
    actions.append(this.rescanBtn, this.primaryBtn, this.themeBtn);
    bot.append(actions);
    this.announce.setAttribute('aria-live', 'polite');
    // Home page with Scan / Collection / Settings; every page has a small "Home" back button.
    const nav = el('nav', 'nav');
    this.homeBtn.innerHTML = ICONS.back + ' Home';
    this.homeBtn.onclick = () => this.handlers.tab('home');
    nav.append(this.homeBtn);
    root.append(nav, top, this.cam, this.strip, this.batchBar, this.duplicateBar, bot, this.flights, this.layer, this.announce);
    this.settingsBtn.onclick = () => this.handlers.settings();
    this.rescanBtn.onclick = () => this.handlers.rescan();
    this.themeBtn.onclick = () => this.handlers.theme();
    this.primaryBtn.onclick = () => this.handlers.primary();
  }

  /** Size of the camera box in CSS pixels (the visible region of the preview). */
  cameraBox(): { width: number; height: number } {
    return { width: this.cam.clientWidth, height: this.cam.clientHeight };
  }

  /** Theme changed (setting or system): recompute on next use. */
  themeChanged(): void {
    this.dark = null;
  }

  isDark(): boolean {
    if (this.dark !== null) return this.dark;
    this.dark = this.computeDark();
    return this.dark;
  }

  private computeDark(): boolean {
    return getComputedStyle(document.documentElement).colorScheme.includes('dark');
  }

  renderBoot(message: string, failed = false) {
    this.title.textContent = failed ? 'Cannot start' : 'Starting';
    this.sub.textContent = message;
    this.primaryBtn.disabled = true;
    this.primaryBtn.textContent = 'Scan now';
    this.themeBtn.innerHTML = this.isDark() ? ICONS.sun : ICONS.moon;
  }

  async attachStream(stream: MediaStream, onFrame: (n: number) => void) {
    this.setStreamActive(false);
    this.video.srcObject = stream;
    this.onFrame = onFrame;
    this.hasFrameCallback = typeof this.video.requestVideoFrameCallback === 'function';
  }

  setStreamActive(active: boolean): void {
    if (this.frameCallback) this.video.cancelVideoFrameCallback?.(this.frameCallback);
    this.frameCallback = 0;
    this.lastFrameAt = 0;
    if (!active) { this.video.pause(); return; }
    void this.video.play().catch(() => {});
    if (this.hasFrameCallback) {
      const tick = (_now: number, meta: VideoFrameCallbackMetadata) => {
        this.lastFrameAt = performance.now();
        this.onFrame?.(meta.presentedFrames);
        this.frameCallback = this.video.requestVideoFrameCallback(tick);
      };
      this.frameCallback = this.video.requestVideoFrameCallback(tick);
    }
  }

  detachStream(): void {
    this.setStreamActive(false);
    this.video.srcObject = null;
    this.onFrame = null;
  }

  /** Keep a display-sized copy of the immutable capture. */
  async setStill(full: ImageBitmap, maxW = 1600) {
    this.stillBitmap?.close();
    this.stillBitmap = full.width > maxW
      ? await createImageBitmap(full, { resizeWidth: maxW, resizeHeight: Math.round((full.height * maxW) / full.width), resizeQuality: 'high' })
      : await createImageBitmap(full);
    this.cam.classList.add('captured');
    this.redrawStill();
  }

  clearStill() {
    this.stillBitmap?.close();
    this.stillBitmap = null;
    // WebKit frees canvas backing stores late: shrink it now rather than keep a screen-sized buffer.
    this.still.width = this.still.height = 0;
    this.cam.classList.remove('captured');
  }

  /** Display copy of the capture held by the view (for memory breadcrumbs). */
  get stillPixels(): number {
    return this.stillBitmap ? this.stillBitmap.width * this.stillBitmap.height : 0;
  }

  redrawStill() {
    if (!this.stillBitmap) return;
    const w = this.cam.clientWidth, h = this.cam.clientHeight;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.still.width = Math.round(w * dpr);
    this.still.height = Math.round(h * dpr);
    const ctx = this.still.getContext('2d')!;
    const f = zoomedFit(this.stillBitmap.width, this.stillBitmap.height, w, h, this.stillZoom);
    ctx.drawImage(this.stillBitmap, f.ox * dpr, f.oy * dpr, this.stillBitmap.width * f.scale * dpr, this.stillBitmap.height * f.scale * dpr);
  }

  // ---------------------------------------------------------------- render

  render(s: AppSnapshot, catalog: Catalog, presenter: Presenter) {
    this.last = { s, catalog, presenter };
    const page = s.page;
    const captured = !!s.batchFrame || (page.status !== 'empty' && page.captureSize !== null && s.cyclePhase !== 'live');
    if (!captured && this.cam.classList.contains('captured') && page.status === 'empty') this.clearStill();
    // Same zoom as the live preview; full screen shows the whole capture including the hidden border.
    const zoom = s.expanded ? 1 : PREVIEW_ZOOM;
    this.stillZoom = zoom;
    const size = s.batchFrame ?? page.captureSize;
    this.fit = captured && size ? zoomedFit(size.width, size.height, this.cam.clientWidth, this.cam.clientHeight, zoom) : null;
    this.renderTop(s, catalog);
    this.renderCamera(s, captured);
    const geo = this.geometry(page);
    presenter.update(page.instances.map((i) => ({
      id: i.id, working: i.state === 'recognizing' || i.state === 'verifying', startedAt: i.startedAt,
      settledAt: i.settledAt, hasCrop: i.crop !== null,
    })), s.now, geo);
    this.renderLoupe(s, catalog, presenter);
    this.renderStrip(s, presenter);
    this.renderFlights(s, presenter, geo);
    this.renderButtons(s, captured);
    this.renderBatch(s);
    this.renderLayer(s, catalog);
  }

  /** Batch mode: compact status under the preview and the capture feedback over it. */
  private renderBatch(s: AppSnapshot) {
    const b = s.batch;
    this.duplicateBar.hidden = !b?.duplicate;
    if (b?.duplicate) {
      setHtml(this.duplicateBar, '<span class="duplicate-label"></span><button class="ghost keep-duplicate">Keep anyway</button><button class="linkbtn discard-duplicate">Dismiss</button>');
      setText(this.duplicateBar.querySelector('.duplicate-label')!, `Possible repeat of page ${b.duplicate.matchedPageNumber} · not saved`);
      const keep = this.duplicateBar.querySelector<HTMLButtonElement>('.keep-duplicate')!;
      keep.disabled = b.capturing || s.sessionChanging;
      keep.onclick = () => this.handlers.keepDuplicate();
      const discard = this.duplicateBar.querySelector<HTMLButtonElement>('.discard-duplicate')!;
      discard.disabled = b.capturing;
      discard.onclick = () => this.handlers.discardDuplicate();
    }
    this.batchBar.style.display = b ? '' : 'none';
    this.strip.style.display = b ? 'none' : '';
    setText(this.toast, s.toast ?? '');
    this.toast.classList.toggle('show', !!s.toast);
    this.renderCue(s);
    if (!b) return;
    const parts = b.active ? [plural(b.pages, 'page'), plural(b.cards, 'card')] : ['No pages scanned yet'];
    if (b.waiting) parts.push(`${b.waiting} waiting`);
    if (b.needsUser) parts.push(`${b.needsUser} need${b.needsUser === 1 ? 's' : ''} you`);
    const working = !!b.running && !b.paused && !b.stalled;
    const html = `<div class="bt">${working ? '<span class="spin" aria-hidden="true"></span>' : ''}<span class="btx"></span></div><button class="ghost fin" type="button" aria-label="Review the scanned pages">Review</button>`;
    setHtml(this.batchBar, html);
    setText(this.batchBar.querySelector('.btx')!, parts.join(' · ') + (b.paused && b.waiting ? ' · paused' : ''));
    const fin = this.batchBar.querySelector<HTMLButtonElement>('.fin')!;
    fin.disabled = !b.active;
    fin.onclick = () => this.handlers.finishBatch();
  }

  /**
   * Batch auto scanning: what to do now, large over the live preview. During the page-turn window
   * a bar empties over its duration; then the steady dwell fills it ("Scan in"). Never on a still.
   */
  private renderCue(s: AppSnapshot) {
    const b = s.batch;
    let text = '', detail = '', fill: number | null = null, kind = '';
    const live = !!b && s.batchRunning && !!s.camera && s.cameraReady && !s.batchFrame && s.cyclePhase === 'live' && !b.capturing && !b.duplicate && !b.full && !b.stalled;
    if (live && s.turn) {
      kind = 'turn'; text = 'Turn the page'; detail = `${Math.ceil(s.turn.leftMs / 1000)} s`;
      fill = s.turn.leftMs / s.turn.totalMs;
    } else if (live && s.awaitingChange) {
      kind = 'turn'; text = 'Turn the page'; detail = `Waiting for page ${s.captureNumber}`;
    } else if (live && s.countdownMs !== null) {
      kind = 'steady'; text = 'Hold steady';
      detail = s.countdownMs === 0 ? 'Waiting for a sharp image' : `Scan in ${(s.countdownMs / 1000).toFixed(1)} s`;
      fill = 1 - s.countdownMs / Math.max(1, s.settings.dwellMs);
    } else if (live) {
      kind = 'steady'; text = 'Hold steady';
    } else if (b && s.batchPaused && s.camera && s.cameraReady && !s.batchFrame && !b.capturing) {
      kind = 'paused'; text = 'Paused'; detail = 'Nothing is scanned';
    }
    this.cue.className = `cuebox${kind ? ` show ${kind}` : ''}`;
    if (!kind) return;
    setHtml(this.cue, '<div class="cuet"></div><div class="cued"></div><div class="cuebar"><span></span></div>');
    setText(this.cue.querySelector('.cuet')!, text);
    setText(this.cue.querySelector('.cued')!, detail);
    const bar = this.cue.querySelector<HTMLElement>('.cuebar')!;
    bar.style.visibility = fill === null ? 'hidden' : '';
    bar.querySelector('span')!.style.width = `${Math.round(100 * Math.max(0, Math.min(1, fill ?? 0)))}%`;
  }

  private counts(page: PageState): string {
    const c = pageCounts(page);
    const parts: string[] = [];
    if (c.ready) parts.push(`${c.ready} ready`);
    if (c.checking) parts.push(`${c.checking} checking`);
    if (c.attention) parts.push(`${c.attention} need${c.attention === 1 ? 's' : ''} attention`);
    if (c.skipped) parts.push(`${c.skipped} skipped`);
    if (c.backs) parts.push(`${c.backs} back${c.backs === 1 ? '' : 's'}`);
    return parts.join(' · ');
  }

  private renderTop(s: AppSnapshot, catalog: Catalog) {
    const page = s.page;
    let title = '', sub = '';
    const counts = this.counts(page);
    if (s.runtime.stage === 'failed') { title = 'Models failed to load'; sub = s.runtime.detail; }
    // Batch: the saved capture stays up with its outlines until the live preview is back.
    else if (s.batch && s.batchSaved && s.batchFrame) { title = s.batchSaved.title; sub = s.batchSaved.sub; }
    else if (s.cameraOpening || (s.camera && !s.cameraReady && s.cyclePhase === 'live')) { title = 'Starting camera'; sub = 'Waiting for a fresh camera image'; }
    else if (!s.camera) { title = s.hiddenStop ? 'Paused' : 'Choose a camera'; sub = s.hiddenStop ? 'Camera stopped while the app was hidden' : ''; }
    else if (s.batch && s.runtime.stage === 'ready' && (s.batch.capturing || s.batch.duplicate || s.batch.full || s.batch.stalled || s.turn || s.awaitingChange)) {
      const b = s.batch;
      if (b.capturing) { title = `Saving page ${s.batchFrame?.number ?? s.captureNumber}`; sub = s.batchQuads ? `${plural(s.batchQuads.length, 'card')} found · storing them` : 'Finding the cards'; }
      else if (b.duplicate) { title = 'Possible repeat'; sub = 'Keep it or dismiss it to continue'; }
      else if (b.stalled) { title = 'Checking stopped'; sub = `${b.stalled}. Scanned pages are saved; reload to continue checking.`; }
      else if (b.full) { title = 'Batch paused'; sub = `${b.waiting} cards are still being checked. Scanning resumes when fewer are waiting.`; }
      else if (s.turn) { title = 'Turn the page'; sub = `Then hold steady · scanning resumes in ${Math.ceil(s.turn.leftMs / 1000)} s`; }
      else { title = 'Turn the page'; sub = `Waiting for page ${s.captureNumber} · then hold steady`; }
    } else if (s.batch && s.camera && s.settings.mode !== 'manual' && !s.batchRunning && s.cyclePhase === 'live' && s.runtime.stage === 'ready') {
      if (s.batchPaused) { title = 'Scanning paused'; sub = `Next: page ${s.captureNumber} · press Continue scanning when it is in view`; }
      else { title = `Ready for page ${s.captureNumber}`; sub = 'Fit the page in view, then press Start'; }
    } else if (s.cyclePhase === 'live' || page.status === 'empty') {
      if (s.runtime.stage !== 'ready') { title = 'Loading models'; sub = s.runtime.detail; }
      else if (s.settings.mode === 'manual') { title = 'Fit the page in view'; sub = 'Tap Scan now when ready'; }
      else if (s.countdownMs === 0 && !s.stability.sharp) { title = 'Hold steady'; sub = 'Waiting for a sharp image'; }
      else if (s.countdownMs !== null) { title = 'Hold steady'; sub = `Scan in ${(s.countdownMs / 1000).toFixed(1)} s`; }
      else { title = 'Fit the page in view'; sub = 'Scans automatically when the image is still'; }
    } else if (page.status === 'detecting') { title = 'Finding cards'; sub = 'Looking for cards in the captured image'; }
    else if (page.status === 'processing') {
      const active = page.instances.find((i) => i.state === 'recognizing' || i.state === 'verifying');
      title = active ? `Checking ${active.number} of ${page.instances.length}` : 'Checking cards';
      sub = counts;
    } else if (page.status === 'failed') { title = 'Detection failed'; sub = page.error ?? ''; }
    else if (page.instances.length === 0) { title = 'No cards found'; sub = 'Adjust the page and scan again.'; }
    else {
      title = 'Review page';
      const toCheck = page.instances.filter(isCue).length;
      sub = counts + (toCheck ? ` · ${toCheck} to double-check` : '') + (page.overLimit ? ` · ${page.overLimit} more not processed (limit)` : '');
      if (s.cyclePhase === 'review' && s.countdownMs !== null) sub += ` · Next scan in ${(s.countdownMs / 1000).toFixed(1)} s`;
    }
    setText(this.title, title);
    setText(this.sub, sub);
    // Announce meaningful state changes only.
    const announce = page.status === 'settled' ? `${title}. ${sub}` : page.status === 'detecting' ? 'Capturing' : '';
    if (announce && announce !== this.lastAnnounce) { this.announce.textContent = announce; this.lastAnnounce = announce; }
    void catalog;
  }

  private renderCamera(s: AppSnapshot, captured: boolean) {
    const page = s.page;
    const expanded = captured && s.expanded;
    // Full screen: rotate the whole photo panel (overlays included) when the photo's
    // orientation differs from the screen's, so a landscape capture fills a portrait phone.
    const size = s.batchFrame ?? s.page.captureSize;
    const appW = this.app.clientWidth, appH = this.app.clientHeight;
    const rotate = expanded && !!size && (size.width > size.height) !== (appW > appH);
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    // One box for live preview and captured photo. Portrait camera (phone held upright): a 3:4
    // box that the preview fills (cropping the edges); the capture is exactly that visible part.
    // Landscape video (desktop/test videos) keeps its full frame.
    const boxAspect = vw > 0 && vh > 0 ? (vh > vw ? 4 / 3 : vh / vw) : 0;
    const aspect = captured && size ? size.height / size.width : boxAspect;
    const layout = expanded ? (rotate ? `r${appW}x${appH}` : 'e') : aspect ? `${captured ? 'c' : 'l'}${Math.round(aspect * 1000)}x${appW}x${appH}` : '';
    if (layout !== this.camLayout) {
      this.camLayout = layout;
      this.cam.classList.toggle('expanded', expanded);
      this.cam.classList.toggle('rotated', rotate);
      this.cam.classList.toggle('fit', !!aspect && !expanded);
      // Same height live and captured, so the framing does not jump when the photo is taken.
      const maxH = appH * 0.62;
      this.cam.style.cssText = rotate
        ? `width:${appH}px;height:${appW}px;left:${(appW - appH) / 2}px;top:${(appH - appW) / 2}px;transform:rotate(90deg)`
        : aspect && !expanded
          ? `height:${Math.round(Math.min(maxH, Math.max(200, (appW - 2 * parseFloat(getComputedStyle(this.cam).marginLeft || '0')) * aspect)))}px`
          : '';
      requestAnimationFrame(() => this.redrawStill());
    }
    this.expandBtn.style.display = captured ? '' : 'none';
    setHtml(this.expandBtn, expanded ? ICONS.shrink : ICONS.expand);
    this.expandBtn.setAttribute('aria-label', expanded ? 'Shrink captured image' : 'Show captured image full screen');
    const chipText = `${s.scanSession?.name ?? 'Scan'} · Page ${s.batchFrame?.number ?? s.captureNumber} · ` + (!s.camera ? 'No camera' : captured ? 'Captured' : s.camera.mock ? 'Test video' : 'Live');
    setHtml(this.chip, `<span class="${captured ? 'still-mark' : 'live'}"></span><span class="t"></span>`);
    setText(this.chip.querySelector('.t')!, chipText + (captured && s.camera?.mock ? ' · test video' : ''));
    if (s.sweepAt !== this.lastSweep) {
      this.lastSweep = s.sweepAt;
      this.fx.innerHTML = '<div class="flash"></div><div class="sweep"></div>';
      setTimeout(() => { if (this.lastSweep === s.sweepAt) this.fx.textContent = ''; }, 800);
    }
    // Overlays live only on the captured image; rebuild them only when what they show changed.
    const camW = this.cam.clientWidth, camH = this.cam.clientHeight;
    const overlayKey = !captured || !this.fit ? '' : s.batchFrame ? `batch|${s.batchFrame.number}|${s.batchQuads?.length}|${camW}x${camH}|${s.batchSaved?.title}`
      : `${page.revision}|${camW}x${camH}|${s.selected}|` + page.instances.map((c) => `${c.state}${isCue(c) ? 'c' : ''}`).join(',');
    if (overlayKey === this.overlayKey) return;
    this.overlayKey = overlayKey;
    this.svg.textContent = '';
    this.hits.textContent = '';
    if (!captured || !this.fit) return;
    this.svg.setAttribute('viewBox', `0 0 ${this.cam.clientWidth} ${this.cam.clientHeight}`);
    // Batch: what was found in this capture, so a missed card is visible before the page is turned.
    for (const quad of s.batchFrame ? s.batchQuads ?? [] : []) {
      // A full outline (not corner brackets): a missing outline must stand out at a glance.
      const path = document.createElementNS(SVG, 'polygon');
      path.setAttribute('class', 'br');
      path.setAttribute('points', mapQuad(quad, this.fit).map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
      const g = document.createElementNS(SVG, 'g');
      g.setAttribute('class', s.batchSaved && !s.batchSaved.repeat ? 'st-found saved' : 'st-found');
      g.append(path);
      this.svg.append(g);
    }
    for (const card of s.batchFrame ? [] : page.instances) {
      const q = mapQuad(card.quad, this.fit);
      const g = document.createElementNS(SVG, 'g');
      g.setAttribute('class', `st-${card.state}${isCue(card) ? ' cue' : ''}${s.selected === card.id ? ' sel' : ''}`);
      if (needsReview(card)) {
        const fill = document.createElementNS(SVG, 'polygon');
        fill.setAttribute('class', 'need');
        fill.setAttribute('points', q.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '));
        g.append(fill);
      }
      const path = document.createElementNS(SVG, 'path');
      path.setAttribute('class', 'br');
      path.setAttribute('d', bracketPath(q));
      g.append(path);
      const [cx, cy] = centroid(q);
      const badge = document.createElementNS(SVG, 'g');
      badge.setAttribute('class', 'badge');
      badge.setAttribute('transform', `translate(${cx.toFixed(1)},${cy.toFixed(1)})`);
      const mark = document.createElementNS(SVG, 'path');
      if (card.state === 'confirmed' && !isCue(card)) mark.setAttribute('d', 'M-4.5 0.4 -1.5 3.4 4.5 -3.4');
      else if (card.state === 'confirmed') mark.setAttribute('d', 'M0 -5 5 0 0 5 -5 0z');
      else if (card.state === 'needs-attention') mark.setAttribute('d', 'M0 -5 5.2 4H-5.2z M0 -1.5v2.6');
      const t = document.createElementNS(SVG, 'text');
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('dy', '3.5');
      if (card.state === 'back') {
        // Compact text pill: a non-colour cue that this outline is a card back, not a card.
        const pill = document.createElementNS(SVG, 'rect');
        for (const [k, v] of [['x', '-17'], ['y', '-9'], ['width', '34'], ['height', '18'], ['rx', '9']]) pill.setAttribute(k, v);
        t.textContent = 'Back';
        badge.append(pill, t);
      } else {
        const circle = document.createElementNS(SVG, 'circle');
        circle.setAttribute('r', '11');
        badge.append(circle);
        if (mark.getAttribute('d')) badge.append(mark);
        else {
          t.textContent = String(card.number);
          badge.append(t);
        }
      }
      g.append(badge);
      this.svg.append(g);
      const xs = q.map((p) => p[0]), ys = q.map((p) => p[1]);
      const w = Math.max(44, Math.max(...xs) - Math.min(...xs)), h = Math.max(44, Math.max(...ys) - Math.min(...ys));
      const hit = el('button', 'hit');
      hit.style.cssText = `left:${cx - w / 2}px;top:${cy - h / 2}px;width:${w}px;height:${h}px;border-radius:4px`;
      hit.setAttribute('aria-label', `Card ${card.number}, ${stateLabel(card)}`);
      hit.onclick = () => this.handlers.select(card.id);
      this.hits.append(hit);
    }
    this.diag.hidden = !s.settings.diagnostics;
    if (s.settings.diagnostics) {
      const times = page.instances.filter((i) => i.ms).map((i) => i.ms!.embed);
      this.diag.textContent = [
        `runtime ${s.runtime.provider ?? '-'}${s.runtime.loadMs ? ` · load ${Math.round(s.runtime.loadMs)} ms` : ''}`,
        s.runtime.fallbackReason ? `fallback: ${s.runtime.fallbackReason}` : '',
        `capture ${page.captureSize?.width}×${page.captureSize?.height}`,
        page.detectMs !== null ? `detect ${Math.round(page.detectMs)} ms · ${page.instances.length} found` : '',
        times.length ? `embed median ${Math.round(times.sort((a, b) => a - b)[times.length >> 1])} ms` : '',
        `thresholds ${s.manifest.thresholds_calibrated ? 'calibrated' : 'uncalibrated'} · bundle ${s.manifest.version}`,
      ].filter(Boolean).join('\n');
    }
  }

  private geometry(page: PageState): Geometry {
    // Measured lazily: reading layout on every render forces a synchronous reflow.
    let rects: { app: DOMRect; cam: DOMRect; crop?: DOMRect } | null = null;
    const measure = () => (rects ??= { app: this.app.getBoundingClientRect(), cam: this.cam.getBoundingClientRect(),
      crop: this.loupe.querySelector('.lcrop')?.getBoundingClientRect() });
    return {
      photo: (id) => {
        const { app, cam } = measure();
        const card = page.instances.find((i) => i.id === id);
        return card && this.fit ? mapQuad(card.quad, this.fit, cam.left - app.left, cam.top - app.top) : null;
      },
      loupe: () => {
        const { app, cam, crop } = measure();
        const r = crop ?? { left: cam.left + 20, top: cam.bottom - 121, width: 72, height: 101 };
        return rectQuad({ left: r.left - app.left, top: r.top - app.top, width: r.width, height: r.height });
      },
      slot: (id) => {
        const { app } = measure();
        const r = this.slotEls.get(id)?.getBoundingClientRect();
        return r ? rectQuad({ left: r.left - app.left, top: r.top - app.top, width: r.width, height: r.height }) : null;
      },
    };
  }

  /** Processing panel: shown only while a card is slow enough to be in focus; otherwise hidden. */
  private renderLoupe(s: AppSnapshot, catalog: Catalog, presenter: Presenter) {
    const page = s.page;
    const focus = presenter.focus ? page.instances.find((i) => i.id === presenter.focus) : undefined;
    const working = !!focus && (focus.state === 'recognizing' || focus.state === 'verifying');
    this.loupe.classList.toggle('show', working && !s.expanded);
    if (!focus || !working) return;
    const steps = ['rectify', 'embed', 'search', 'verify'] as const;
    const k = steps.indexOf(focus.step ?? 'rectify');
    const key = `focus|${focus.id}|${focus.revision}|${focus.step}|${presenter.focusLanded}`;
    if (key === this.loupeKey) return;
    this.loupeKey = key;
    this.loupe.textContent = '';
    const crop = el('div', 'lcrop');
    crop.append(cropCanvas(focus.crop, 72, 101, presenter.focusLanded ? '' : 'pending'));
    if (focus.step === 'embed') crop.append(el('div', 'grid'), el('div', 'scanb'));
    const body = el('div', 'lbody');
    body.append(el('div', 'lk', `Card ${focus.number} of ${page.instances.length} · taking longer`));
    body.append(el('div', 'lname', 'Checking…'));
    const list = el('div', 'sub4');
    void catalog;
    steps.forEach((step, i) => {
      const row = el('div', `s4 ${i < k ? 'done' : i === k ? 'on' : ''}`);
      row.append(el('span', 'm'), document.createTextNode(step[0].toUpperCase() + step.slice(1)));
      list.append(row);
    });
    body.append(list);
    this.loupe.append(crop, body);
  }

  /** Progress bar: one bucket per detected card; crops fly in, state shows by colour and outline. */
  private renderStrip(s: AppSnapshot, presenter: Presenter) {
    const list = s.page.instances;
    const key = list.map((i) => `${i.id}:${i.state}:${needsReview(i)}:${presenter.landed.has(i.id)}:${s.selected === i.id}:${presenter.focus === i.id}`).join('|');
    if (key === this.stripKey) return;
    this.stripKey = key;
    this.strip.textContent = '';
    this.slotEls.clear();
    for (const card of list) {
      const filled = presenter.landed.has(card.id) && card.crop !== null;
      const working = card.state === 'recognizing' || card.state === 'verifying';
      const cls = ['slot', filled ? 'filled' : '', working ? 'work' : '', `st-${card.state}`,
        needsReview(card) ? 'need' : '', s.selected === card.id ? 'on' : ''].filter(Boolean).join(' ');
      const slot = el('button', cls);
      slot.setAttribute('aria-label', `Card ${card.number}, ${stateLabel(card)}`);
      slot.onclick = () => this.handlers.select(card.id);
      if (filled) slot.append(cropCanvas(card.crop, 24, 34));
      this.strip.append(slot);
      this.slotEls.set(card.id, slot);
    }
  }

  private renderFlights(s: AppSnapshot, presenter: Presenter, geo: Geometry) {
    for (const [id, node] of this.flightEls) {
      if (!presenter.flights.has(id)) { node.remove(); this.flightEls.delete(id); }
    }
    for (const flight of presenter.flights.values()) {
      const card = s.page.instances.find((i) => i.id === flight.id);
      const q: Quad | null = presenter.quadAt(flight, s.now, geo);
      if (!card || !q) continue;
      let node = this.flightEls.get(flight.id);
      if (!node) {
        node = el('div', 'fly');
        node.append(cropCanvas(card.crop, 224, 312));
        this.flights.append(node);
        this.flightEls.set(flight.id, node);
      }
      node.style.transform = quadMatrix3d(224, 312, q);
    }
  }

  private renderButtons(s: AppSnapshot, captured: boolean) {
    // Right of the main button: the page's card list (after a capture).
    const listMode = captured && s.page.instances.length > 0;
    setHtml(this.themeBtn, ICONS.list);
    this.themeBtn.setAttribute('aria-label', 'All cards on this page');
    this.themeBtn.style.visibility = listMode ? '' : 'hidden';
    this.themeBtn.onclick = () => this.handlers.review();
    // The page review has its own back link: one back control per screen.
    this.homeBtn.style.visibility = s.overlay === 'start' || s.overlay === 'review' ? 'hidden' : '';
    // Side buttons keep their space (visibility), so the main button stays centred.
    this.rescanBtn.style.visibility = captured ? '' : 'hidden';
    const page = s.page;
    let label = 'Scan now', disabled = false;
    const batchAuto = !!s.batch && s.settings.mode !== 'manual';
    if (s.batch) {
      this.themeBtn.style.visibility = 'hidden';
      // Auto scanning: a manual "Scan now" stays available on the left while it runs.
      const manual = batchAuto && s.batchRunning && !!s.camera;
      this.rescanBtn.style.visibility = manual ? '' : 'hidden';
      setHtml(this.rescanBtn, manual ? ICONS.shutter : ICONS.rescan);
      this.rescanBtn.setAttribute('aria-label', manual ? 'Scan now' : 'Scan again');
      const b = s.batch;
      this.rescanBtn.onclick = manual ? () => this.handlers.captureNow() : () => this.handlers.rescan();
      this.rescanBtn.disabled = manual && (!s.cameraReady || s.runtime.stage !== 'ready' || b.capturing || !!b.duplicate || b.full || !!b.stalled || s.cyclePhase !== 'live');
    } else {
      setHtml(this.rescanBtn, ICONS.rescan);
      this.rescanBtn.setAttribute('aria-label', 'Scan again');
      this.rescanBtn.disabled = false;
      this.rescanBtn.onclick = () => this.handlers.rescan();
    }
    if (!s.camera) label = s.hiddenStop ? 'Resume camera' : 'Choose camera';
    else if (s.batch) {
      const b = s.batch;
      if (b.capturing) { label = 'Saving page…'; disabled = true; }
      else if (batchAuto) {
        // Automatic scanning starts only on Start; Pause stops it (the preview stays live).
        label = s.batchRunning ? 'Pause scanning' : s.batchPaused ? 'Continue scanning' : 'Start scanning';
        disabled = !s.batchRunning && (!s.cameraReady || s.runtime.stage !== 'ready' || !!b.duplicate || b.full || !!b.stalled);
      } else {
        disabled = !s.cameraReady || s.runtime.stage !== 'ready' || !!b.duplicate || b.full || !!b.stalled || s.cyclePhase !== 'live';
        label = b.full ? 'Waiting for the backlog' : 'Scan now';
      }
    }
    else if (s.cyclePhase === 'live') { disabled = !s.cameraReady || s.runtime.stage !== 'ready'; }
    else if (page.status === 'detecting') { label = 'Finding cards…'; disabled = true; }
    else if (page.status === 'processing') label = 'Review so far';
    else if (page.status === 'failed' || page.instances.length === 0) label = 'Scan again';
    else {
      const n = page.instances.filter(needsReview).length;
      label = n ? `Check ${n} card${n === 1 ? '' : 's'}` : 'Next scan';
    }
    this.primaryBtn.classList.toggle('attn', label.startsWith('Check'));
    this.primaryBtn.classList.toggle('paused', label === 'Pause scanning');
    setText(this.primaryBtn, label);
    this.primaryBtn.disabled = disabled || s.cameraOpening || (s.sessionChanging && page.status === 'settled');
    this.primaryBtn.onclick = () => (!s.camera && s.hiddenStop ? this.handlers.resume() : this.handlers.primary());
  }

  // ---------------------------------------------------------------- overlays

  private renderLayer(s: AppSnapshot, catalog: Catalog) {
    const card = s.selected ? s.page.instances.find((i) => i.id === s.selected) : undefined;
    const viewerKey = card ? `${this.cardKey(card)}|${s.viewerMode}|${s.compare}|${!!s.bigCrop}|${s.settings.reviewMode}|${s.settings.preferredLanguage}|${s.viewerSearch.query}|${s.viewerSearch.results.length}` : '';
    const key = JSON.stringify([s.scanSession, s.scanSessions, s.sessionChanging, s.error]) + (s.overlay === 'review' ? `review|${this.reviewKey(s)}|${viewerKey}`
      : s.overlay === 'settings' ? `settings|${JSON.stringify(s.settings)}|${s.runtime.provider}|${this.advancedOpen}`
      : s.overlay === 'collection' ? `collection|${s.collection.rev}|${s.collection.open}|${s.collection.view}|${s.collection.filter}|${s.settings.priceMarket}`
      : s.overlay === 'batch' ? `batch|${s.collection.rev}|${JSON.stringify(s.batch)}|${s.settings.priceMarket}`
        : s.overlay === 'start' ? `start|${s.videos.length}|${s.error}|${s.runtime.stage}|${!!s.camera}|${s.collection.rev}|${s.settings.workflow}|${this.newBinderOpen}`
          : card ? `viewer|${viewerKey}` : '');
    if (key === this.layerKey) {
      this.updateProgress(s);
      return;
    }
    this.layerKey = key;
    // Redrawing the same screen keeps its scroll position and the search field's focus and caret.
    const screenId = s.overlay === 'collection' || s.overlay === 'batch' ? `${s.overlay}|${s.collection.view}|${s.collection.open}|${s.collection.card?.pageKey}#${s.collection.card?.number}` : s.overlay;
    const sameScreen = screenId === this.screenId;
    // Fade in only over the scanner; from another screen (or a redraw) the switch is instant.
    const overScanner = this.screenId === 'none' || this.screenId === '';
    const prevScroll = sameScreen ? (this.layer.querySelector('.screen') as HTMLElement | null)?.scrollTop ?? 0 : 0;
    const input = this.searchInput;
    const focused = !!input && document.activeElement === input;
    const caret = focused ? [input!.selectionStart, input!.selectionEnd] as const : null;
    this.screenId = screenId;
    this.layer.textContent = '';
    if (s.overlay === 'start') this.layer.append(this.startScreen(s));
    else if (s.overlay === 'settings') this.layer.append(this.settingsScreen(s));
    else if (s.overlay === 'collection') this.layer.append(this.collectionScreen(s, catalog));
    else if (s.overlay === 'batch') this.layer.append(this.batchScreen(s, catalog));
    else if (s.overlay === 'review') this.layer.append(this.reviewPage(s, catalog));
    // A fade-in from opacity 0 lets the scanner underneath flash through between two screens
    // (Home → Collection → Home, or redrawing Home when switching the workflow).
    if (!overScanner) this.layer.firstElementChild?.classList.add('swap');
    const screen = this.layer.querySelector('.screen') as HTMLElement | null;
    if (screen && prevScroll) screen.scrollTop = prevScroll;
    if (focused && input!.isConnected) {
      input!.focus({ preventScroll: true });
      if (caret) input!.setSelectionRange(caret[0], caret[1]);
    }
    const viewer = !!card && (s.overlay === 'none' || s.overlay === 'review');
    if (card && viewer) {
      const v = this.viewer(card, s, catalog);
      if (this.viewerOpen) v.classList.add('swap'); // already open: switch content without the open animation
      this.layer.append(v);
    }
    this.viewerOpen = viewer;
  }

  private cardKey(c: CardInstance): string {
    return `${c.id}|${c.revision}|${c.state}|${c.step}|${c.valueState}|${c.decision?.kind}|${c.decision?.kind === 'matched' ? c.decision.preferred?.id : ''}|${c.userChoice?.identity}|${!!c.crop}`;
  }

  private reviewKey(s: AppSnapshot): string {
    return `${s.page.revision}|${s.page.savedRevision}|${s.page.status}|${s.page.instances.map((i) => this.cardKey(i)).join(',')}`;
  }

  private updateProgress(s: AppSnapshot) {
    const bar = this.layer.querySelector<HTMLElement>('.pbar span');
    const label = this.layer.querySelector('.plabel');
    if (!bar || !label) return;
    const p = s.runtime.progress;
    bar.style.width = `${Math.round(100 * (p ?? 0))}%`;
    setText(label, s.runtime.detail);
  }

  /** Home view state: the "New binder" form is open. */
  private newBinderOpen = false;
  /** Settings view state: the Advanced section is open. */
  private advancedOpen = false;
  private last: { s: AppSnapshot; catalog: Catalog; presenter: Presenter } | null = null;

  /** Redraw after a change of view-only state (no app state changed). */
  private rerender() {
    this.layerKey = '';
    if (this.last) this.render(this.last.s, this.last.catalog, this.last.presenter);
  }

  /** The game most saved cards of a binder resolved to (candidate display order). */
  private binderGame(s: AppSnapshot, sessionId: string | null | undefined, catalog: Catalog): string | null {
    if (!sessionId) return null;
    return dominantGame(s.collection.pages.filter((p) => p.sessionId === sessionId)
      .flatMap((p) => p.cards.map((c) => (c.identity ? catalog.get(c.identity)?.game : null))));
  }

  /** Expected game for the open capture: its confirmed cards, else the binder's saved cards. */
  private scannerGame(s: AppSnapshot, catalog: Catalog): string | null {
    const games = s.page.instances.map((i) => (i.state === 'confirmed' && i.decision?.kind === 'matched' ? i.decision.top && catalog.get(i.decision.top.identity)?.game : null));
    return dominantGame(games) ?? this.binderGame(s, s.scanSession?.id, catalog);
  }

  /** Home: scanning, Collection (with its value) and Settings. Scanning waits for the models. */
  private startScreen(s: AppSnapshot): HTMLElement {
    const wrap = el('div', 'start home');
    wrap.append(el('h1', '', 'Card scanner'));
    const pages = s.collection.pages;
    const scan = el('button', 'hbtn primary');
    scan.innerHTML = `${ICONS.scan}<span>Scan</span><small></small>`;
    if (s.scanSession) scan.querySelector('span')!.textContent = `Continue ${s.scanSession.name}`;
    scan.onclick = () => this.handlers.tab('scan');
    const small = scan.querySelector('small')!;
    if (s.runtime.stage === 'loading') {
      scan.disabled = true;
      small.textContent = 'Loading models…';
      const bar = el('div', 'pbar');
      bar.append(el('span'));
      scan.append(bar, el('span', 'plabel', s.runtime.detail));
      requestAnimationFrame(() => this.updateProgress(s));
    } else if (s.runtime.stage === 'failed') {
      scan.disabled = true;
      small.textContent = `Models failed to load: ${s.runtime.detail}`;
    } else {
      small.textContent = s.scanSession ? `${s.page.status === 'empty' ? 'Next: page' : 'Review page'} ${s.captureNumber}` : 'One card or a whole binder page';
    }
    scan.disabled ||= s.sessionChanging || s.cameraOpening;
    wrap.append(scan);
    // How to scan: chosen here, where scanning starts (not hidden in Settings).
    const mode = el('div', 'seg modes');
    mode.setAttribute('role', 'group');
    mode.setAttribute('aria-label', 'Scan workflow');
    for (const [value, label] of [['page', 'Single review'], ['batch', 'Batch review']] as const) {
      const b = el('button', s.settings.workflow === value ? 'on' : '', label);
      b.setAttribute('aria-pressed', String(s.settings.workflow === value));
      b.disabled = s.sessionChanging;
      b.onclick = () => { if (s.settings.workflow !== value) this.handlers.changeSetting({ workflow: value }); };
      mode.append(b);
    }
    wrap.append(mode);
    wrap.append(el('p', 'modehint', s.settings.workflow === 'batch'
      ? 'Scan page after page; cards are checked in the background and reviewed at the end.'
      : 'Check each page right after it is scanned.'));
    const binders = new Set(pages.map((p) => p.sessionId)).size;
    const t = totals(pages);
    const coll = el('button', 'hbtn');
    coll.innerHTML = `${ICONS.stack}<span>Collection</span><small></small>`;
    // Value first (today's prices; unknown prices are not counted), then what it is made of.
    const worth = t.confirmed ? formatMoney(t.confirmed, t.currency) : null;
    coll.querySelector('small')!.textContent = t.cards ? [worth, plural(t.cards, 'card'), plural(binders, 'binder')].filter(Boolean).join(' · ') : 'Nothing saved yet';
    coll.onclick = () => this.handlers.tab('collection');
    const set = el('button', 'hbtn');
    set.innerHTML = `${ICONS.gear}<span>Settings</span><small>Language, prices, trigger</small>`;
    set.onclick = () => this.handlers.tab('settings');
    wrap.append(coll, set);
    // Binders: start a new one (optional name), or finish the current one. Rename lives in Collection.
    const binder = el('div', 'binder-actions');
    if (this.newBinderOpen) {
      const form = el('form', 'new-session');
      const name = el('input');
      name.type = 'text';
      name.placeholder = 'e.g. Red Binder';
      name.maxLength = 80;
      name.setAttribute('aria-label', 'New binder name (optional)');
      const create = el('button', 'ghost', 'Start');
      create.type = 'submit';
      create.disabled = s.sessionChanging || s.cameraOpening;
      form.onsubmit = (e) => { e.preventDefault(); if (create.disabled) return; this.newBinderOpen = false; this.handlers.newSession(name.value); };
      form.append(name, create);
      binder.append(form);
      requestAnimationFrame(() => name.focus({ preventScroll: true }));
    } else {
      const open = el('button', 'linkbtn', 'New binder…');
      open.disabled = s.sessionChanging || s.cameraOpening;
      open.onclick = () => { this.newBinderOpen = true; this.rerender(); };
      binder.append(open);
      if (s.scanSession) {
        const finish = el('button', 'linkbtn', 'Finish binder');
        finish.setAttribute('aria-label', `Finish ${s.scanSession.name}`);
        finish.disabled = s.sessionChanging;
        finish.onclick = () => this.handlers.finishSession();
        binder.append(finish);
      }
    }
    wrap.append(binder);
    if (s.error) wrap.append(el('div', 'err', s.error));
    if (s.videos.length && s.runtime.stage === 'ready') {
      const dev = el('div', 'devlinks');
      dev.append(el('span', 'k', 'Test videos (dev):'));
      for (const v of s.videos) {
        const b = el('button', 'linkbtn', v.name);
        b.onclick = () => this.handlers.useVideo(v);
        dev.append(b);
      }
      wrap.append(dev);
    }
    wrap.append(el('p', 'appversion', `Version ${__APP_VERSION__} · bundle ${s.manifest.version.slice(0, 8)}`));
    return wrap;
  }

  private settingsScreen(s: AppSnapshot): HTMLElement {
    const sheet = el('div', 'screen settings');
    sheet.setAttribute('aria-label', 'Settings');
    sheet.append(el('h1', 'title', 'Settings'));
    const select = <K extends keyof Settings>(parent: HTMLElement, label: string, key: K, options: [Settings[K], string][], current: Settings[K] = s.settings[key],
      apply: (v: Settings[K]) => Partial<Settings> = (v) => ({ [key]: v } as Partial<Settings>)) => {
      const l = el('label', '', label);
      const sel = el('select');
      for (const [v, text] of options) {
        const o = el('option', '', text);
        o.value = String(v);
        o.selected = current === v;
        sel.append(o);
      }
      sel.onchange = () => {
        const raw = sel.value;
        this.handlers.changeSetting(apply(options.find(([o]) => String(o) === raw)![0]));
      };
      l.append(sel);
      parent.append(l);
    };
    select(sheet, 'Card language', 'preferredLanguage', [['en', 'English'], ['de', 'German']]);
    select(sheet, 'Prices', 'priceMarket', [['EUR', 'Cardmarket (€)'], ['USD', 'TCGplayer ($)']]);
    select(sheet, 'Trigger', 'mode', [['auto', 'Auto when steady'], ['manual', 'Manual']], s.settings.mode === 'repeat' ? 'auto' : s.settings.mode);
    select(sheet, 'Capture feedback', 'captureFeedback', [['vibrate', 'Vibrate or click'], ['off', 'Flash only']]);
    if (s.settings.workflow === 'page') select(sheet, 'After a scan', 'reviewMode', [['trust', 'Show a Check button'], ['review', 'Open the check automatically']]);
    select(sheet, 'Theme', 'theme', [['system', 'System'], ['dark', 'Dark'], ['light', 'Light']]);
    const adv = el('details', 'advanced');
    adv.open = this.advancedOpen;
    adv.addEventListener('toggle', () => { this.advancedOpen = adv.open; });
    adv.append(el('summary', '', 'Advanced'));
    select(adv, 'Steady duration', 'dwellMs', [[400, '0.4 s'], [800, '0.8 s'], [1200, '1.2 s'], [2000, '2 s']]);
    select(adv, 'Time to turn the page (batch)', 'batchCooldownMs', [[3000, '3 s'], [5000, '5 s'], [8000, '8 s'], [10000, '10 s']]);
    // Repeat-while-steady (single page only): rescans the same page, replacing its result.
    select(adv, 'Rescan the same page', 'cooldownMs', [[0, 'Off'], [1000, 'Every 1 s'], [3000, 'Every 3 s'], [5000, 'Every 5 s'], [10000, 'Every 10 s']],
      s.settings.mode === 'repeat' ? s.settings.cooldownMs : 0,
      (v) => (v ? { mode: 'repeat', cooldownMs: v } : { mode: s.settings.mode === 'repeat' ? 'auto' : s.settings.mode }));
    select(adv, 'Motion tolerance', 'tolerance', [['sensitive', 'Sensitive'], ['normal', 'Normal'], ['relaxed', 'Relaxed']]);
    select(adv, 'GPU inference (reload to apply)', 'preferWebGPU', [[true, 'WebGPU if available'], [false, 'WASM only']]);
    select(adv, 'Diagnostics', 'diagnostics', [[false, 'Off'], [true, 'On']]);
    select(adv, 'Share scans to improve recognition', 'saveCaptures', [[true, 'On'], [false, 'Off']]);
    adv.append(el('div', 'k', 'When on, photos of scans with cards and your corrections are sent to the scanner server to train future versions.'));
    adv.append(el('div', 'k', `Runtime: ${s.runtime.provider ?? s.runtime.detail}${s.runtime.fallbackReason ? ` (${s.runtime.fallbackReason})` : ''} · bundle ${s.manifest.version}`));
    sheet.append(adv);
    return sheet;
  }

  /** Current value of a saved card (today's snapshot), else its scan-time value. */
  private savedValue(c: SavedCard) {
    return c.current !== undefined ? c.current : c.value;
  }

  private collectionScreen(s: AppSnapshot, catalog: Catalog): HTMLElement {
    const wrap = el('div', 'screen collection');
    if (s.error) wrap.append(el('p', 'err', s.error));
    const pages = [...s.collection.pages].sort((a, b) => b.savedAt.localeCompare(a.savedAt));
    const money = (v: number, c: string) => formatMoney(v, c);
    const open = s.collection.open ? pages.find((p) => p.key === s.collection.open) : undefined;
    const sc = s.collection.card;
    const cardPage = sc ? pages.find((p) => p.key === sc.pageKey) : undefined;
    const savedCard = cardPage?.cards.find((c) => c.number === sc!.number);
    if (cardPage && savedCard && (open?.key === cardPage.key || !open)) {
      return this.savedCardScreen(s, cardPage, savedCard, catalog, open ? `Page ${cardPage.pageNumber}` : 'All cards');
    }
    const binderName = (id: string) => s.scanSessions.find((x) => x.id === id)?.name ?? 'Binder';
    if (open) {
      const back = el('button', 'back');
      back.innerHTML = ICONS.back + ' Collection';
      back.onclick = () => this.handlers.openSaved(null);
      const t = totals([open]);
      wrap.append(back, el('h1', 'title', `Page ${open.pageNumber}`),
        el('div', 'sub', `${binderName(open.sessionId)} · ${new Date(open.savedAt).toLocaleDateString()} · ${plural(open.cards.length, 'card')}${open.partial ? ' · partial' : ''}`));
      wrap.append(this.totalsRow(t, money));
      const list = el('div', 'clist');
      for (const c of open.cards) {
        list.append(swipeToDelete(this.savedRow(c, `card ${c.number}`, s, () => this.handlers.openSavedCard(open.key, c.number)),
          `Delete card ${c.number}`, () => this.handlers.deleteSavedCard(open.key, c.number)));
      }
      wrap.append(list);
      wrap.append(el('p', 'note', 'Tap a card to check it. Swipe left to delete.'));
      const del = el('button', 'linkbtn', 'Delete this page');
      del.onclick = async () => {
        if (await confirmDialog({ title: `Delete page ${open.pageNumber}?`, message: `Its ${plural(open.cards.length, 'card')} are removed from the collection.`, confirm: 'Delete page', danger: true })) this.handlers.deleteSaved(open.key);
      };
      wrap.append(del);
      return wrap;
    }
    wrap.append(el('h1', 'title', 'Collection'));
    const transfer = el('div', 'transfer-actions');
    const load = el('button', 'linkbtn', 'Import');
    load.setAttribute('aria-label', 'Import a collection file');
    load.disabled = s.sessionChanging;
    load.onclick = () => this.filePicker().click();
    if (pages.length) {
      const save = el('button', 'linkbtn', 'Export');
      save.setAttribute('aria-label', 'Export the collection as a file');
      save.disabled = s.sessionChanging;
      save.onclick = () => this.handlers.exportCollection();
      transfer.append(save);
    }
    transfer.append(load);
    wrap.append(transfer);
    if (!pages.length && !s.scanSessions.length) {
      wrap.append(el('p', 'note', 'Scanned pages appear here once they are saved. Import restores a collection exported from another browser.'));
      return wrap;
    }
    const all = totals(pages);
    wrap.append(el('div', 'sub', `${plural(all.cards, 'card')} · ${all.identified} identified · ${plural(new Set(pages.map((p) => p.sessionId)).size, 'binder')}`), this.totalsRow(all, money));
    const seg = el('div', 'seg views');
    for (const [view, label] of [['binders', 'Binders'], ['cards', 'All cards']] as const) {
      const b = el('button', s.collection.view === view ? 'on' : '', label);
      b.setAttribute('aria-pressed', String(s.collection.view === view));
      b.onclick = () => this.handlers.collectionView(view);
      seg.append(b);
    }
    wrap.append(seg);
    if (s.collection.view === 'cards') {
      const entries = pages.flatMap((p) => p.cards.map((c) => ({ p, c })));
      const check = entries.filter(({ c }) => savedNeedsUser(c));
      const chips = el('div', 'chips');
      for (const [filter, label] of [['all', `All · ${entries.length}`], ['check', `Needs a check · ${check.length}`]] as const) {
        const b = el('button', s.collection.filter === filter ? 'on' : '', label);
        b.setAttribute('aria-pressed', String(s.collection.filter === filter));
        b.onclick = () => this.handlers.collectionView('cards', filter);
        chips.append(b);
      }
      wrap.append(chips);
      const amount = (c: SavedCard) => { const v = this.savedValue(c); return v ? v.preferred ?? v.high ?? v.low ?? -1 : -1; };
      const shown = (s.collection.filter === 'check' ? check : entries).sort((a, b) => amount(b.c) - amount(a.c));
      const list = el('div', 'clist');
      for (const { p, c } of shown) {
        const where = `${binderName(p.sessionId)} · page ${p.pageNumber}` + (savedNeedsUser(c) && c.name ? ` · ${savedLabel(c)}` : '');
        list.append(this.savedRow(c, where, s, () => this.handlers.openSavedCard(p.key, c.number)));
      }
      if (!shown.length) list.append(el('p', 'note', 'Nothing needs a check.'));
      wrap.append(list);
    } else {
      const sessions = new Map<string, SavedPage[]>();
      for (const session of [...s.scanSessions].sort((a, b) => b.startedAt.localeCompare(a.startedAt))) sessions.set(session.id, []);
      for (const p of pages) (sessions.get(p.sessionId) ?? sessions.set(p.sessionId, []).get(p.sessionId)!).push(p);
      for (const [id, list] of sessions) {
        const t = totals(list);
        const head = el('div', 'shead session-head');
        const session = s.scanSessions.find((x) => x.id === id);
        const day = new Date(session?.startedAt ?? list[list.length - 1].savedAt);
        const name = session?.name ?? 'Binder';
        // Name and value on one line, details below, then the binder's actions in one row.
        const summary = el('div', 'session-summary');
        const line = el('div', 'session-line');
        line.append(el('div', 'sn', name), el('div', 'cv', t.confirmed ? money(t.confirmed, t.currency) : '—'));
        summary.append(line, el('div', 'k', `${day.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })} · ${plural(list.length, 'page')} · ${plural(t.cards, 'card')}`));
        const actions = el('div', 'session-actions');
        const resume = el('button', 'ghost', 'Continue scanning');
        resume.setAttribute('aria-label', `Continue ${name}`);
        resume.disabled = s.sessionChanging;
        resume.onclick = () => this.handlers.resumeSession(id);
        const rename = el('button', 'linkbtn', 'Rename');
        rename.setAttribute('aria-label', `Rename ${name}`);
        rename.disabled = s.sessionChanging;
        rename.onclick = async () => {
          const next = await promptDialog({ title: 'Rename binder', value: name, label: 'Binder name', confirm: 'Rename' });
          if (next !== null) this.handlers.renameSession(id, next);
        };
        const del = el('button', 'linkbtn danger', 'Delete');
        del.setAttribute('aria-label', `Delete ${name}`);
        del.disabled = s.sessionChanging;
        del.onclick = async () => {
          if (await confirmDialog({ title: `Delete “${name}”?`, message: `Its ${plural(list.length, 'saved page')} and ${plural(t.cards, 'card')} are removed from this browser. This cannot be undone.`, confirm: 'Delete binder', danger: true })) this.handlers.deleteSession(id);
        };
        actions.append(resume, rename, del);
        head.append(summary, actions);
        wrap.append(head);
        for (const p of list) {
          const b = el('button', 'prow');
          const thumbs = el('div', 'pthumbs');
          for (const c of p.cards.slice(0, 6)) {
            if (c.thumb) { const img = el('img'); img.src = c.thumb; img.alt = ''; thumbs.append(img); }
          }
          const pt = totals([p]);
          const info = el('div', 'ci');
          info.append(el('div', 'cn2', `Page ${p.pageNumber}`), el('div', 'k', `${plural(p.cards.length, 'card')}${p.partial ? ' · partial' : ''}`));
          b.append(thumbs, info, el('div', 'cv', pt.confirmed ? money(pt.confirmed, pt.currency) : '—'));
          b.onclick = () => this.handlers.openSaved(p.key);
          wrap.append(swipeToDelete(b, `Delete page ${p.pageNumber}`, () => this.handlers.deleteSaved(p.key)));
        }
      }
      wrap.append(el('p', 'note', 'Swipe a page left to delete it.'));
    }
    wrap.append(el('p', 'note', 'Prices: latest bundled snapshot in the market chosen in Settings (≈ converted from the other market). “Still to check” is the highest plausible price of cards whose version is open. Unknown prices are never counted as zero.'));
    return wrap;
  }

  /** One saved card as a list row: thumbnail, name, where/what, value (with the change since the scan). */
  private savedRow(c: SavedCard, detail: string, s: AppSnapshot, onOpen: () => void): HTMLElement {
    const row = el('button', 'crow2');
    row.setAttribute('aria-label', `${c.name ?? savedLabel(c)}, ${detail}. Open to check`);
    if (c.thumb) { const img = el('img'); img.src = c.thumb; img.alt = ''; row.append(img); } else row.append(el('div', 'ph'));
    const info = el('div', 'ci');
    const needs = savedNeedsUser(c);
    const k = el('div', `k${c.state === 'needs-attention' ? ' attn' : ''}`);
    k.innerHTML = needs ? (c.state === 'needs-attention' ? ICONS.att : ICONS.cue) + ' ' : '';
    k.append([c.cardId, detail, c.checkedByUser ? 'checked by you' : ''].filter(Boolean).join(' · '));
    info.append(el('div', 'cn2', c.name ?? savedLabel(c)), k);
    row.append(info);
    const v = this.savedValue(c);
    const cv = el('div', 'cv', !v ? (c.identity ? 'unknown' : '—') : v.preferred !== null ? formatMoney(v.preferred, v.currency, v.converted) + (v.cue ? ' *' : '') : 'unknown');
    // Scan-time value as history, in today's currency (exchange rate of the snapshot).
    const then = v && c.current ? scanAmountIn(c.value, v.currency, s.collection.usdToEur) : null;
    if (then !== null && v?.preferred != null && Math.abs(then - v.preferred) >= 0.01) {
      const d = v.preferred - then;
      cv.append(el('div', 'k', `${d > 0 ? '+' : '−'}${formatMoney(Math.abs(d), v.currency)} since scan`));
    }
    row.append(cv);
    row.onclick = onOpen;
    return row;
  }

  /** One saved card: its crop, current match and value; check again, pick another card, or search the catalog. */
  private savedCardScreen(s: AppSnapshot, page: SavedPage, saved: SavedPage['cards'][number], catalog: Catalog, backLabel: string): HTMLElement {
    const sc = s.collection.card!;
    const wrap = el('div', 'screen collection savedcard');
    const back = el('button', 'back');
    back.innerHTML = ICONS.back + ' ';
    back.append(backLabel);
    back.onclick = () => this.handlers.openSavedCard(page.key, null);
    wrap.append(back);
    const top = el('div', 'sctop');
    const image = el('div', 'scimg');
    image.append(el('div', 'k', 'Your scan'));
    if (sc.image) image.append(cropCanvas(sc.image, sc.image.width, sc.image.height, 'single'));
    else if (saved.thumb) { const img = el('img', 'single'); img.src = saved.thumb; img.alt = ''; image.append(img); }
    else image.append(el('div', 'ph single', 'No image'));
    top.append(image);
    const current = saved.identity ? catalog.get(saved.identity) : undefined;
    const pick = sc.pick ? catalog.get(sc.pick) : undefined;
    const side = el('div', 'scside');
    if (pick) side.append(el('div', 'k', 'Compare'), this.refImage(pick, 'single'));
    else if (current) side.append(el('div', 'k', 'Catalog'), this.refImage(current, 'single'));
    top.append(side);
    wrap.append(top);

    const subject = pick ?? current;
    const name = el('div', 'vname', subject?.name ?? (saved.state === 'not-in-catalog' ? 'Not in the catalog' : 'Not identified'));
    if (subject) name.append(el('span', 'vid', subject.cardId));
    wrap.append(name);
    if (pick) {
      const row = el('div', 'actions');
      const use = el('button', 'primary', 'Use this card');
      use.onclick = () => this.handlers.savedChoose(pick.id, false);
      const cancel = el('button', 'ghost', 'Back');
      cancel.onclick = () => this.handlers.savedPick(null);
      row.append(use, cancel);
      wrap.append(row);
    } else {
      wrap.append(el('div', 'k', [`Page ${page.pageNumber} · card ${saved.number}`, saved.checkedByUser ? 'checked by you' : saved.identity ? 'matched by the scanner' : savedLabel(saved),
        saved.languageAssumed ? `${languageName(s.settings.preferredLanguage)} assumed` : ''].filter(Boolean).join(' · ')));
      const v = this.savedValue(saved);
      if (v?.preferred != null) wrap.append(el('div', 'big', formatMoney(v.preferred, v.currency, v.converted)));
      // Language versions of the current artwork: one tap confirms the exact version.
      const versions = current ? catalog.family(current.id) : [];
      if (versions.length > 1) {
        const chips = el('div', 'chips');
        for (const ver of versions) {
          const on = saved.identity === ver.id && !saved.languageAssumed;
          const b = el('button', on ? 'on' : '', languageName(ver.language));
          b.setAttribute('aria-pressed', String(on));
          b.onclick = () => this.handlers.savedChoose(ver.id, true);
          chips.append(b);
        }
        wrap.append(chips);
      }
    }

    // Closest matches from searching again with the current bundle.
    const game = this.binderGame(s, page.sessionId, catalog);
    const cands = el('div', 'cands');
    if (sc.status === 'loading' || sc.status === 'checking') {
      const w = el('div', 'wait');
      w.append(el('span', 'spin'), document.createTextNode('Checking this card again…'));
      cands.append(w);
    } else if (sc.status === 'no-crop') {
      cands.append(el('div', 'k', 'Saved before full crops were kept, so it cannot be checked again. Search the catalog below.'));
    } else if (sc.status === 'failed') {
      cands.append(el('div', 'k', 'Checking again failed. Search the catalog below.'));
    } else if (sc.hits) {
      cands.append(el('div', 'k', 'Closest matches · tap to compare'));
      cands.append(this.cardRow(candidates(sc.hits, catalog, s.settings.preferredLanguage, 8, game).map((c) => c.card), sc.pick, saved.identity,
        (id) => this.handlers.savedPick(sc.pick === id ? null : id), s));
    }
    wrap.append(cands);

    // Manual search when the right card is not among the matches.
    const search = el('div', 'cands');
    search.append(el('div', 'k', 'Not there? Search by name or number'));
    search.append(this.searchField(sc.query, (q) => this.handlers.savedSearch(q)));
    if (sc.query.trim().length >= 2) {
      search.append(sc.results.length ? this.cardRow(sc.results, sc.pick, saved.identity, (id) => this.handlers.savedPick(sc.pick === id ? null : id), s)
        : el('div', 'k', 'No catalog entries found.'));
    }
    wrap.append(search);

    const foot = el('div', 'actions');
    const none = el('button', 'linkbtn', 'Not in the catalog');
    none.onclick = () => this.handlers.savedChoose(null, false);
    const del = el('button', 'linkbtn danger', 'Delete card');
    del.onclick = async () => {
      if (await confirmDialog({ title: 'Delete this card?', message: 'It is removed from the collection.', confirm: 'Delete card', danger: true })) this.handlers.deleteSavedCard(page.key, saved.number);
    };
    foot.append(none, del);
    wrap.append(foot);
    return wrap;
  }

  /** Batch review: the cards that need the user first, then every page of the batch. */
  private batchScreen(s: AppSnapshot, catalog: Catalog): HTMLElement {
    const b = s.batch;
    const keys = b?.active?.pageKeys ?? [];
    const byKey = new Map(s.collection.pages.map((p) => [p.key, p]));
    const pages = keys.map((k) => byKey.get(k)).filter((p): p is SavedPage => !!p);
    const sc = s.collection.card;
    const open = sc ? pages.find((p) => p.key === sc.pageKey) : undefined;
    const saved = open?.cards.find((c) => c.number === sc!.number);
    if (open && saved) return this.savedCardScreen(s, open, saved, catalog, 'Batch review');
    const wrap = el('div', 'screen collection batchrev');
    const back = el('button', 'back');
    back.innerHTML = ICONS.back + ' Scanner';
    back.onclick = () => this.handlers.scanMore();
    wrap.append(back, el('h1', 'title', 'Batch review'));
    const cards = pages.flatMap((p) => p.cards.map((c) => ({ p, c })));
    const identified = cards.filter(({ c }) => c.identity).length;
    wrap.append(el('div', 'sub', `${plural(pages.length, 'page')} · ${plural(cards.length, 'card')} · ${identified} identified`));
    if (b && b.waiting) {
      const w = el('div', 'wait');
      const working = !!b.running && !b.paused && !b.stalled;
      if (working) w.append(el('span', 'spin'));
      w.append(document.createTextNode(b.stalled ? `Checking stopped: ${b.stalled}. Reload to continue; nothing is lost.`
        : `${plural(b.waiting, 'card')} still being checked${working ? '' : ' (paused)'}. Results appear here as they finish.`));
      wrap.append(w);
    }
    if (b?.duplicate) {
      const held = el('div', 'held');
      held.append(el('div', '', `Possible repeat of page ${b.duplicate.matchedPageNumber} (${plural(b.duplicate.cards, 'card')}), not saved yet. Keep or dismiss it before finishing.`));
      const row = el('div', 'heldactions');
      const keep = el('button', 'ghost', 'Keep anyway');
      keep.disabled = b.capturing || s.sessionChanging;
      keep.onclick = () => this.handlers.keepDuplicate();
      const discard = el('button', 'linkbtn', 'Dismiss');
      discard.disabled = b.capturing;
      discard.onclick = () => this.handlers.discardDuplicate();
      row.append(keep, discard);
      held.append(row);
      wrap.append(held);
    }
    const need = cards.filter(({ c }) => savedNeedsUser(c));
    const head = (text: string) => { const h = el('div', 'shead'); h.append(el('div', 'sn', text)); return h; };
    wrap.append(head(`Needs you · ${need.length}`));
    if (need.length) {
      const list = el('div', 'clist');
      for (const { p, c } of need) list.append(this.savedRow(c, `page ${p.pageNumber} · card ${c.number}`, s, () => this.handlers.openSavedCard(p.key, c.number)));
      wrap.append(list);
    } else {
      wrap.append(el('p', 'note', b?.waiting ? 'Nothing so far. Cards that need you appear here while the rest is checked.' : 'Nothing needs you: every card was matched.'));
    }
    wrap.append(head('Pages'));
    for (const p of pages) {
      const waiting = p.cards.filter((c) => c.state === 'queued').length;
      const box = el('div', 'bpage');
      box.append(el('div', 'k', `Page ${p.pageNumber} · ${plural(p.cards.length, 'card')}${waiting ? ` · ${waiting} waiting` : p.partial ? ' · partial' : ''}`));
      const thumbs = el('div', 'bthumbs');
      for (const c of p.cards) {
        const t = el('button', `bthumb st-${c.state}${savedNeedsUser(c) ? ' need' : ''}`);
        t.setAttribute('aria-label', `Page ${p.pageNumber}, card ${c.number}: ${savedLabel(c)}`);
        if (c.thumb) { const img = el('img'); img.src = c.thumb; img.alt = ''; t.append(img); }
        t.onclick = () => this.handlers.openSavedCard(p.key, c.number);
        thumbs.append(t);
      }
      box.append(thumbs);
      wrap.append(box);
    }
    const foot = el('div', 'actions bfoot');
    const more = el('button', 'ghost', 'Scan more');
    more.onclick = () => this.handlers.scanMore();
    const done = el('button', 'primary', 'Done');
    done.disabled = !!b?.capturing || !!b?.duplicate || s.sessionChanging;
    done.onclick = () => this.handlers.doneBatch();
    foot.append(more, done);
    wrap.append(foot);
    wrap.append(el('p', 'note', 'Every page is already in the Collection. Cards still waiting keep being checked there.'));
    return wrap;
  }

  /** Catalog cards to tap: image, name and number. */
  private cardRow(cards: CatalogCard[], pick: string | null, currentId: string | null, onPick: (id: string) => void, s: AppSnapshot): HTMLElement {
    const row = el('div', 'crow');
    for (const c of cards) {
      const current = currentId !== null && c.id === currentId;
      const t = el('button', `cand${pick === c.id ? ' on' : ''}${current ? ' cur' : ''}`);
      t.setAttribute('aria-label', `Compare with ${c.name} ${c.cardId}${current ? ' (current)' : ''}`);
      t.append(this.refImage(c), el('div', 'cn', c.name), el('div', 'k', c.cardId + (s.settings.diagnostics ? ` · ${c.language}` : '')));
      t.onclick = () => onPick(c.id);
      row.append(t);
    }
    return row;
  }

  /** One persistent input, so typing survives redraws (focus and caret stay). */
  private searchInput: HTMLInputElement | null = null;

  private searchField(value: string, onInput: (query: string) => void): HTMLInputElement {
    if (!this.searchInput) {
      const input = el('input', 'search');
      input.type = 'search';
      input.placeholder = 'e.g. Nami, OP01-016, sv03 224';
      input.autocomplete = 'off';
      input.setAttribute('aria-label', 'Search the catalog by name or card number');
      this.searchInput = input;
    }
    const input = this.searchInput;
    input.oninput = () => onInput(input.value);
    if (input.value !== value) input.value = value;
    return input;
  }

  private totalsRow(t: Totals, money: (v: number, c: string) => string): HTMLElement {
    const tot = el('div', 'tot');
    const cell = (label: string, value: string) => { const d = el('div'); d.append(el('div', 'k', label), el('div', 'tv', value)); return d; };
    tot.append(cell('Value', t.confirmed ? money(t.confirmed, t.currency) : '—'),
      cell('Still to check', t.provisionalHigh || t.provisionalOpen ? `up to ${t.provisionalOpen ? '?' : money(t.provisionalHigh, t.currency)}` : '—'),
      cell('No price', String(t.unpriced)));
    return tot;
  }

  private valueBlock(card: CardInstance, m: MatchResult, catalog: Catalog): HTMLElement[] {
    const v = card.value;
    if (card.valueState === 'loading' || card.valueState === 'idle') {
      const w = el('div', 'wait');
      w.append(el('span', 'spin'), document.createTextNode('Loading prices…'));
      return [w];
    }
    if (card.valueState === 'failed' || !v) return [el('div', 'wait', 'Price lookup failed. Value unknown.')];
    const pref = m.preferred;
    const block = el('div', 'val');
    block.append(el('div', 'big', v.preferred ? formatQuote(v.preferred) : 'Value unknown'));
    block.append(el('div', 'k', [pref ? languageName(pref.language) + (m.languageAssumed ? ' assumed' : '') : '', v.source && v.asOf ? `${v.source.split(' (')[0]}, ${v.asOf}` : ''].filter(Boolean).join(' · ')));
    // Other versions only when they could change the value.
    if (v.plausible > 1 && (v.high === null || v.low !== v.high)) {
      const r2 = el('div', 'vrow');
      r2.style.marginTop = '10px';
      r2.append(el('span', 'k', 'Other versions'), el('span', 'mono', formatRange(v)));
      block.append(r2);
      if (v.low !== null) block.append(this.rangeBar(v));
      if (v.priced < v.plausible) block.append(el('div', 'k', `${v.priced} of ${v.plausible} versions priced`));
    }
    const conv = v.preferred?.converted;
    if (conv) block.append(el('div', 'k', `≈ No ${v.preferred!.currency === 'EUR' ? 'Cardmarket' : 'TCGplayer'} price for this version; converted from the ${conv.from === 'USD' ? 'TCGplayer $' : 'Cardmarket €'} price at 1 ${conv.from} = ${conv.rate} ${v.preferred!.currency} (${conv.source}, ${conv.date}).`));
    else if (v.converted) block.append(el('div', 'k', '≈ Some versions are priced by currency conversion.'));
    if (v.cue || v.rangeIncomplete) {
      const box = el('div', 'cuebox');
      box.innerHTML = ICONS.cue;
      const text = el('div');
      if (v.cue?.kind === 'check-variant') {
        const alt = v.cue.alternative;
        text.innerHTML = `<b>Check variant.</b> `;
        text.append(document.createTextNode(`The ${languageName(alt.language)} version is listed at ${formatMoney(v.cue.amount, v.currency ?? 'USD')}. Tap its language above if it is that one.`));
      } else if (v.cue?.kind === 'variant-unpriced') {
        text.innerHTML = `<b>Variant unpriced.</b> `;
        text.append(document.createTextNode(`The ${languageName(v.cue.alternative.language)} version has no price, so the upper bound is unknown.`));
      } else {
        text.innerHTML = '<b>Range incomplete.</b> ';
        text.append(document.createTextNode('A possible version has no price yet, so the upper bound is unknown.'));
      }
      box.append(text);
      block.append(box);
    }
    void catalog;
    return [block];
  }

  private rangeBar(v: ValueSummary): HTMLElement {
    const LO = Math.log10(0.05), HI = Math.log10(500);
    const pos = (x: number) => Math.max(0, Math.min(100, ((Math.log10(x) - LO) / (HI - LO)) * 100));
    const wrap = el('div');
    const track = el('div', 'track');
    const left = pos(v.low!), right = v.high !== null ? pos(v.high) : 100;
    const pl = el('span', `pl${v.high === null ? ' open' : ''}`);
    pl.style.cssText = `left:${left}%;width:${Math.max(1, right - left)}%`;
    track.append(pl);
    if (v.preferred) {
      const es = el('span', 'es');
      es.style.cssText = `left:calc(${pos(v.preferred.amount)}% - 2px);width:4px`;
      track.append(es);
    }
    const ticks = el('div', 'ticks');
    for (const t of [0.05, 0.5, 5, 50, 500]) ticks.append(el('span', '', formatMoney(t, v.currency ?? 'USD')));
    wrap.append(track, ticks);
    return wrap;
  }

  private screenId = '';
  private refImages = new Map<string, { img: HTMLImageElement; build: number }>();
  private build = 0;

  private picker: HTMLInputElement | null = null;

  /** One file input outside the rendered tree: a re-render while the picker is open must not drop the choice. */
  private filePicker(): HTMLInputElement {
    if (this.picker) return this.picker;
    const picker = el('input');
    picker.type = 'file';
    picker.accept = '.json,application/json';
    picker.hidden = true;
    picker.onchange = () => { const file = picker.files?.[0]; picker.value = ''; if (file) this.handlers.importCollection(file); };
    document.body.append(picker);
    return (this.picker = picker);
  }

  private refImage(card: CatalogCard | undefined, cls = ''): HTMLElement {
    if (!card?.ref) return el('div', `ph ${cls}`, 'No reference image');
    // Reuse loaded images: a fresh <img> per render flashes empty until it decodes again.
    const key = `${card.ref}|${cls}`;
    const cached = this.refImages.get(key);
    if (cached) {
      // Each node can sit in one place: a second use within the same build gets a copy.
      if (cached.build === this.build) return cached.img.cloneNode(true) as HTMLElement;
      cached.build = this.build;
      return cached.img;
    }
    if (this.refImages.size > 80) this.refImages.delete(this.refImages.keys().next().value!);
    const img = el('img', cls);
    this.refImages.set(key, { img, build: this.build });
    img.alt = `Catalog image of ${card.name}`;
    img.loading = 'lazy';
    img.decoding = 'async';
    img.src = `refs/${card.ref}`;
    img.onerror = () => { this.refImages.delete(key); img.replaceWith(el('div', `ph ${cls}`, 'Reference image not available here')); };
    return img;
  }

  /** The card's region in the captured photo, with its detected outline. */
  private photoCanvas(card: CardInstance, s: AppSnapshot): HTMLCanvasElement {
    const c = el('canvas', 'photo');
    const bmp = this.stillBitmap, size = s.page.captureSize;
    if (!bmp || !size) return c;
    const k = bmp.width / size.width;
    const xs = card.quad.map((p) => p[0] * k), ys = card.quad.map((p) => p[1] * k);
    const w = Math.max(...xs) - Math.min(...xs), h = Math.max(...ys) - Math.min(...ys);
    const pad = 0.35 * Math.max(w, h);
    const x0 = Math.max(0, Math.min(...xs) - pad), y0 = Math.max(0, Math.min(...ys) - pad);
    const x1 = Math.min(bmp.width, Math.max(...xs) + pad), y1 = Math.min(bmp.height, Math.max(...ys) + pad);
    const scale = Math.min(3, 1200 / Math.max(x1 - x0, y1 - y0));
    c.width = Math.round((x1 - x0) * scale);
    c.height = Math.round((y1 - y0) * scale);
    const ctx = c.getContext('2d')!;
    ctx.drawImage(bmp, x0, y0, x1 - x0, y1 - y0, 0, 0, c.width, c.height);
    ctx.strokeStyle = '#EDE9E3';
    ctx.lineWidth = Math.max(2, c.width / 300);
    ctx.beginPath();
    card.quad.forEach((p, i) => {
      const x = (p[0] * k - x0) * scale, y = (p[1] * k - y0) * scale;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
    return c;
  }

  /** The user's crop beside a catalog image (matched card, or a candidate being compared). */
  private panes(card: CardInstance, s: AppSnapshot, ref: CatalogCard, refLabel: string): HTMLElement[] {
    const mine = el('div', 'pane');
    mine.append(el('div', 'k', 'Your scan'), s.bigCrop ? cropCanvas(s.bigCrop, 300, 419) : cropCanvas(card.crop, 300, 419));
    const theirs = el('div', 'pane');
    theirs.append(el('div', 'k', refLabel), this.refImage(ref));
    return [mine, theirs];
  }

  private viewer(card: CardInstance, s: AppSnapshot, catalog: Catalog): HTMLElement {
    this.build++;
    const page = s.page;
    const index = page.instances.findIndex((i) => i.id === card.id);
    const review = s.settings.reviewMode === 'review' || !!s.checking;
    const settled = card.state === 'confirmed' || card.state === 'needs-attention' || card.state === 'skipped' || card.state === 'back';
    const m = card.decision?.kind === 'matched' ? card.decision : null;
    const shown = card.state === 'confirmed' && m ? displayCard(m, catalog) : undefined;
    const game = this.scannerGame(s, catalog);
    const list = settled && card.hits ? candidates(card.hits, catalog, s.settings.preferredLanguage, 6, game) : [];
    const searched: Candidate[] = s.viewerSearch.results.map((c) => ({ family: c.family, card: c, versions: catalog.family(c.id), score: NaN }));
    const comparing = s.compare ? list.find((c) => c.family === s.compare) ?? searched.find((c) => c.family === s.compare) : undefined;

    // Unresolved cards: smaller image so the candidates to pick from are visible without scrolling.
    const wrap = el('div', `viewer${card.state === 'needs-attention' || card.state === 'skipped' ? ' pick' : ''}`);
    wrap.setAttribute('role', 'dialog');
    wrap.setAttribute('aria-label', `Card ${card.number} details`);
    const top = el('div', 'vtop');
    const btn = (html: string, label: string, fn: () => void, disabled = false) => {
      const b = el('button', 'vbtn');
      b.innerHTML = html;
      b.setAttribute('aria-label', label);
      b.disabled = disabled;
      b.onclick = fn;
      return b;
    };
    top.append(btn(ICONS.close, 'Close', () => this.handlers.select(null)),
      el('div', 't', s.checking ? `CHECK ${s.checking.index + 1} OF ${s.checking.total}` : `CARD ${card.number} OF ${page.instances.length}${card.userChoice ? ' · CHECKED BY YOU' : ''}`),
      btn(ICONS.prev, 'Previous card', () => this.handlers.step(-1), index <= 0),
      btn(ICONS.next, 'Next card', () => this.handlers.step(1), index >= page.instances.length - 1));
    wrap.append(top);

    const body = el('div', 'vbody');
    const image = el('div', 'vimg');
    if (comparing) image.append(...this.panes(card, s, comparing.card, 'Compare'));
    else if (s.viewerMode === 'photo') image.append(this.photoCanvas(card, s));
    else if (shown) image.append(...this.panes(card, s, shown, 'Catalog'));
    else image.append(s.bigCrop ? cropCanvas(s.bigCrop, 630, 880, 'single') : card.crop ? cropCanvas(card.crop, 224, 312, 'single') : el('div', 'ph single', 'Not straightened yet'));
    body.append(image);
    if (!comparing) {
      const seg = el('div', 'seg');
      for (const [mode, label] of [['card', 'Card'], ['photo', 'In photo']] as const) {
        const b = el('button', s.viewerMode === mode ? 'on' : '', label);
        b.onclick = () => this.handlers.viewerMode(mode);
        seg.append(b);
      }
      body.append(seg);
    }

    // Result
    const subject = comparing?.card ?? shown;
    const name = el('div', 'vname', subject?.name ?? (card.state === 'back' ? 'Card back'
      : card.state === 'needs-attention' || card.state === 'skipped' ? 'Not identified' : 'Checking…'));
    if (subject) name.append(el('span', 'vid', subject.cardId));
    body.append(name);
    if (!comparing) body.append(el('div', 'k', stateLabel(card) + (s.settings.diagnostics && shown ? ` · ${shown.id}` : '')));
    if (!comparing && m && card.state === 'confirmed') body.append(...this.valueBlock(card, m, catalog));
    // Language versions of the artwork: one tap confirms the exact version.
    const versions = comparing?.versions ?? (shown ? catalog.family(shown.id) : []);
    if (versions.length > 1) {
      const chips = el('div', 'chips');
      for (const v of versions) {
        const on = !comparing && card.userChoice?.identity === v.id && m?.languageEvidence === 'user';
        const b = el('button', on ? 'on' : '', languageName(v.language));
        b.setAttribute('aria-label', `${languageName(v.language)} version`);
        b.setAttribute('aria-pressed', String(on));
        b.onclick = () => this.handlers.choose(card.id, v.id, true, review && !comparing);
        chips.append(b);
      }
      body.append(chips);
    }
    if (!comparing && m && m.languageFallback && shown && shown.language !== m.preferredLanguage) {
      body.append(el('div', 'k', `No ${languageName(m.preferredLanguage)} version in the catalog · ${languageName(shown.language)} card and price shown`));
    }
    if (!comparing && card.state === 'back') {
      body.append(el('div', 'k', 'The back of a card: not matched, valued or saved. If it is a card front, tap the matching card below.'));
    }
    if (!comparing && (card.state === 'needs-attention' || card.state === 'skipped')) {
      const box = el('div', 'cuebox');
      box.innerHTML = ICONS.att;
      box.append(el('div', '', this.hint(card, catalog)));
      body.append(box);
    }
    if (!settled) {
      const w = el('div', 'wait');
      if (card.state !== 'queued') w.append(el('span', 'spin'));
      w.append(document.createTextNode(card.state === 'queued' ? 'Waiting to check. Cards are checked one at a time.' : 'Checking card…'));
      body.append(w);
    }

    // Candidates and catalog search: tap one to compare it side by side with your card.
    const candRow = (items: Candidate[]) => {
      const row = el('div', 'crow');
      for (const c of items) {
        const t = el('button', `cand${comparing?.family === c.family ? ' on' : ''}`);
        t.setAttribute('aria-label', `Compare with ${c.card.name} ${c.card.cardId}`);
        t.append(this.refImage(c.card), el('div', 'cn', c.card.name), el('div', 'k', c.card.cardId + (s.settings.diagnostics && Number.isFinite(c.score) ? ` · ${c.score.toFixed(2)}` : '')));
        t.onclick = () => this.handlers.compare(comparing?.family === c.family ? null : c.family);
        row.append(t);
      }
      return row;
    };
    if (list.length) {
      const cands = el('div', 'cands');
      cands.append(el('div', 'k', card.state === 'confirmed' ? 'Closest matches' : card.state === 'back' ? 'Closest card fronts' : 'Tap the matching card'), candRow(list));
      body.append(cands);
    }
    if (settled && card.state !== 'back') {
      const search = el('div', 'cands');
      search.append(el('div', 'k', 'Not there? Search by name or number'), this.searchField(s.viewerSearch.query, (q) => this.handlers.viewerSearch(q)));
      if (s.viewerSearch.query.trim().length >= 2) search.append(searched.length ? candRow(searched) : el('div', 'k', 'No catalog entries found.'));
      body.append(search);
    }
    if (s.settings.diagnostics && card.decision?.top) {
      const d = card.decision;
      const c = el('div', 'cands k');
      c.append(el('div', '', `similarity ${d.top!.score.toFixed(3)} · margin ${d.margin.toFixed(3)} (thresholds ${d.thresholds.minScore}/${d.thresholds.minMargin}, ${d.thresholds.calibrated ? 'calibrated' : 'uncalibrated'})`));
      if (card.quality) c.append(el('div', '', `crop short side ${Math.round(card.quality.shortSide)} px · glare ${(card.quality.glare * 100).toFixed(1)}%`));
      if (card.ms) c.append(el('div', '', `rectify ${card.ms.rectify.toFixed(0)} ms · embed ${card.ms.embed.toFixed(0)} ms · search ${card.ms.search.toFixed(0)} ms`));
      body.append(c);
    }
    wrap.append(body);

    // Actions
    const act = el('div', 'vact');
    if (comparing) {
      const back = el('button', 'ghost', 'Not this');
      back.onclick = () => this.handlers.compare(null);
      const pick = el('button', 'primary', 'This is it');
      pick.onclick = () => this.handlers.choose(card.id, comparing.card.id, false, review);
      act.append(back, pick);
    } else if (settled) {
      const none = el('button', 'linkbtn', 'None of these');
      none.onclick = () => this.handlers.chooseNone(card.id);
      if (card.state === 'back') {
        const next = el('button', 'primary', index < page.instances.length - 1 ? 'Next' : 'Done');
        next.onclick = () => (index < page.instances.length - 1 ? this.handlers.step(1) : this.handlers.select(null));
        act.append(next);
      } else if (card.state === 'confirmed') {
        const ok = el('button', 'primary', review ? (index < page.instances.length - 1 ? 'Correct · next' : 'Correct · done') : 'Next');
        ok.onclick = () => (review && shown ? this.handlers.choose(card.id, shown.id, false, true) : this.handlers.step(1));
        if (!review && index >= page.instances.length - 1) ok.textContent = 'Done', ok.onclick = () => this.handlers.select(null);
        act.append(none, ok);
      } else {
        const retry = el('button', 'linkbtn', 'Retry');
        retry.onclick = () => this.handlers.retry(card.id);
        // Not in the catalog (recorded as a training example) vs. decide later.
        const notListed = el('button', 'ghost', 'None of these');
        notListed.onclick = () => this.handlers.chooseNone(card.id);
        const later = el('button', 'primary', card.state === 'skipped' ? 'Next' : 'Decide later');
        later.onclick = () => (card.state === 'skipped' ? this.handlers.step(1) : this.handlers.skip(card.id));
        // Same pixels + same model = same result: retrying only helps when the run was cut short.
        if (card.reason === 'timeout' || card.reason === 'error') act.append(retry);
        act.append(notListed, later);
      }
    } else {
      const next = el('button', 'ghost', 'Next card');
      next.onclick = () => this.handlers.step(1);
      act.append(next);
    }
    wrap.append(act);

    // Swipe between cards; arrow keys and Escape on desktop.
    let x0: number | null = null;
    image.addEventListener('touchstart', (e) => { x0 = e.touches[0].clientX; }, { passive: true });
    image.addEventListener('touchend', (e) => {
      if (x0 === null) return;
      const dx = e.changedTouches[0].clientX - x0;
      x0 = null;
      if (Math.abs(dx) > 50) this.handlers.step(dx < 0 ? 1 : -1);
    });
    wrap.tabIndex = -1;
    wrap.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === 'ArrowRight') this.handlers.step(1);
      else if (e.key === 'ArrowLeft') this.handlers.step(-1);
      else if (e.key === 'Escape') this.handlers.select(null);
    });
    requestAnimationFrame(() => { if (!wrap.contains(document.activeElement)) wrap.focus({ preventScroll: true }); });
    return wrap;
  }

  private hint(card: CardInstance, catalog: Catalog): string {
    if (card.state === 'skipped') return 'Left for later. It is not counted as a recognized card; pick the right card below at any time.';
    const d = card.decision;
    switch (card.reason) {
      case 'glare': return 'Glare covers part of the card, so no match was accepted. Tilt the page away from the light and scan again, or pick the card below.';
      case 'move-closer': return 'The card is small in the image. Move closer and scan again, or pick the card below.';
      case 'not-found': return 'No catalog card was similar enough. Pick it below or search for it.';
      case 'match-unclear': {
        const a = d?.top ? catalog.get(d.top.identity) : undefined, b = d?.competitor ? catalog.get(d.competitor.identity) : undefined;
        return `Two cards are about equally similar${a && b ? `: ${a.name} (${a.cardId}) and ${b.name} (${b.cardId})` : ''}. Pick the right one below.`;
      }
      case 'timeout': return 'Checking this card took too long, so it was stopped. The other cards were not held up.';
      default: return 'Something went wrong while checking this card.';
    }
  }

  private reviewPage(s: AppSnapshot, catalog: Catalog): HTMLElement {
    this.build++;
    const page = s.page;
    const wrap = el('div', 'review');
    const rh = el('div', 'rh');
    const back = el('button', 'back');
    back.innerHTML = ICONS.back + ' Captured image';
    back.onclick = () => this.handlers.closeOverlay();
    const c = pageCounts(page);
    const complete = page.status === 'settled';
    rh.append(back, el('h2', 'title', `Page ${page.pageNumber}`), el('div', 'sub', `${complete ? '' : 'Partial · '}${plural(c.found, 'card')} found · ${this.counts(page)}${page.overLimit ? ` · ${page.overLimit} over the limit` : ''}`));
    wrap.append(rh);
    // Totals keep confirmed values, provisional ranges and unpriced cards apart.
    let confirmed = 0, confirmedN = 0, provHigh = 0, provOpen = false, provN = 0, unpriced = 0, loading = false, currency = 'USD';
    for (const i of page.instances) {
      if (i.state !== 'confirmed') continue;
      if (i.valueState === 'loading') { loading = true; continue; }
      const v = i.value;
      if (!v || v.status === 'unknown') { unpriced++; continue; }
      currency = v.currency ?? currency;
      const m = i.decision as MatchResult;
      if (!v.cue && !v.rangeIncomplete && !languageMatters(m.languageAssumed, v) && v.preferred) { confirmed += v.preferred.amount; confirmedN++; }
      else { provN++; if (v.high === null) provOpen = true; else provHigh += v.high; if (!v.preferred) unpriced++; }
    }
    const tot = el('div', 'tot');
    const cell = (label: string, value: string) => { const d = el('div'); d.append(el('div', 'k', label), el('div', 'tv', value)); return d; };
    tot.append(
      cell('Value', confirmedN ? formatMoney(confirmed, currency) : '—'),
      cell('Still to check', provN ? `up to ${provOpen ? '?' : formatMoney(provHigh, currency)}` : '—'),
      cell('No price', loading ? '…' : String(unpriced)),
    );
    wrap.append(tot);
    const grid = el('div', 'sheetgrid');
    for (const i of page.instances) {
      const tile = el('button', 'tile');
      tile.append(i.crop ? cropCanvas(i.crop, 110, 154) : el('div', 'ph'));
      const m = i.decision?.kind === 'matched' ? i.decision : null;
      const shown = m ? displayCard(m, catalog) : undefined;
      tile.append(el('div', 'tn', `${i.number} · ${shown?.name ?? (i.state === 'needs-attention' ? REASON[i.reason ?? 'error'] : i.state === 'skipped' ? 'Decided later' : i.state === 'back' ? 'Card back' : 'Checking')}`));
      const v = i.value;
      const line = i.state !== 'confirmed' ? stateLabel(i)
        : i.valueState === 'loading' ? 'Loading price'
          : !v ? 'Value unknown'
            : v.cue?.kind === 'check-variant' ? `Check variant · ${formatRange(v)}`
              : v.rangeIncomplete ? `Incomplete · ${formatRange(v)}`
                : v.preferred ? formatQuote(v.preferred) : 'Value unknown';
      const k = el('div', 'k', line);
      k.style.marginTop = '2px';
      tile.append(k);
      tile.onclick = () => this.handlers.select(i.id);
      grid.append(tile);
    }
    wrap.append(grid);
    const rf = el('div', 'rf');
    const again = el('button', 'ghost', 'Scan again');
    again.disabled = s.sessionChanging;
    again.onclick = () => this.handlers.rescan();
    // Saving and moving on are one step: the page is saved (idempotently) before the next one starts.
    const next = el('button', 'primary', 'Save & next');
    next.disabled = s.sessionChanging || page.status === 'detecting';
    next.onclick = () => this.handlers.nextPage();
    rf.append(again, next);
    wrap.append(rf);
    return wrap;
  }
}
