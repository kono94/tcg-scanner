// Installed-app (home screen) support: standalone detection and the browser's install offer.
// Nothing here changes the page in a browser tab except the dismissible install hint on Home.

/** How Home can offer installing: the browser's own prompt (Chromium), or Share → Add to Home Screen (iOS). */
export type InstallHint = 'prompt' | 'ios' | null;

interface InstallPromptEvent extends Event {
  prompt(): Promise<void>;
  readonly userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

const DISMISSED_KEY = 'tcg-install-dismissed';
const DARK_BG = '#0F0E0D', LIGHT_BG = '#F2F1EE';

/** iPhone/iPad (every browser there is WebKit with a tight per-tab memory limit). */
export function isIOS(): boolean {
  return /iPhone|iPad|iPod/.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
}

/** Opened from the home screen: no browser UI around the page. */
export function isStandalone(): boolean {
  return matchMedia('(display-mode: standalone)').matches || matchMedia('(display-mode: fullscreen)').matches
    || (navigator as Navigator & { standalone?: boolean }).standalone === true;
}

export class InstallOffer {
  /** The installed-app layout (tab bar) applies: set as the `standalone` class on <html>. */
  standalone = isStandalone();
  onChange: () => void = () => {};
  private deferred: InstallPromptEvent | null = null;
  private installed = false;
  private dismissed = false;
  private themeColor: HTMLMetaElement | null = null;

  /** Construct early: Chromium fires beforeinstallprompt soon after load. */
  constructor() {
    try { this.dismissed = localStorage.getItem(DISMISSED_KEY) === '1'; } catch { /* storage blocked: offer again */ }
    document.documentElement.classList.toggle('standalone', this.standalone);
    window.addEventListener('beforeinstallprompt', (e) => {
      // Keep Chrome's own install bar away; Home offers it instead (and the browser menu still has it).
      e.preventDefault();
      this.deferred = e as InstallPromptEvent;
      this.onChange();
    });
    window.addEventListener('appinstalled', () => { this.deferred = null; this.installed = true; this.onChange(); });
    matchMedia('(display-mode: standalone)').addEventListener('change', () => {
      this.standalone = isStandalone();
      document.documentElement.classList.toggle('standalone', this.standalone);
      this.onChange();
    });
  }

  get hint(): InstallHint {
    if (this.standalone || this.installed || this.dismissed) return null;
    if (this.deferred) return 'prompt';
    return isIOS() ? 'ios' : null;
  }

  async install(): Promise<void> {
    const e = this.deferred;
    if (!e) return;
    this.deferred = null; // a prompt can be shown once
    await e.prompt();
    if ((await e.userChoice).outcome === 'accepted') this.installed = true;
    this.onChange();
  }

  dismiss(): void {
    this.dismissed = true;
    try { localStorage.setItem(DISMISSED_KEY, '1'); } catch { /* hidden for this visit only */ }
    this.onChange();
  }

  /** Installed app: colour the system status bar like the page. Browser tabs keep their own toolbar colour. */
  setTheme(dark: boolean): void {
    if (!this.standalone) return;
    if (!this.themeColor) {
      this.themeColor = document.createElement('meta');
      this.themeColor.name = 'theme-color';
      document.head.append(this.themeColor);
    }
    const color = dark ? DARK_BG : LIGHT_BG;
    if (this.themeColor.content !== color) this.themeColor.content = color;
  }
}
