// Browser Back/Forward between screens, the installed-app tab bar, and Home's install offer.
import { expect, test, type Page } from '@playwright/test';

const thumb = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=';
const seeded = {
  sessions: { activeId: null, sessions: [{ id: 's-nav', name: 'Nav Binder', startedAt: '2026-10-19T10:00:00Z', nextCapture: 2, named: true }] },
  pages: [{ key: 'page-nav-1', sessionId: 's-nav', pageNumber: 1, revision: 1, savedAt: '2026-10-19T10:00:00Z', partial: false, cards: [
    { number: 1, identity: null, name: null, cardId: null, state: 'needs-attention', reason: 'match-unclear', languageAssumed: false, checkedByUser: false, value: null, thumb, hasCrop: false },
  ] }],
};

async function open(page: Page, standalone: boolean) {
  await page.addInitScript(({ data, standalone }) => {
    localStorage.setItem('tcg-test-run', '1');
    localStorage.setItem('tcg-scanner-sessions-v2', JSON.stringify(data.sessions));
    localStorage.setItem('tcg-scanner-collection-v1', JSON.stringify(data.pages));
    if (standalone) {
      // Opened from the home screen (Playwright cannot emulate display-mode).
      const real = window.matchMedia.bind(window);
      window.matchMedia = (q: string) => (q === '(display-mode: standalone)' ? { ...real(q), matches: true, media: q, addEventListener() {}, removeEventListener() {} } as MediaQueryList : real(q));
    }
  }, { data: seeded, standalone });
  await page.goto('/');
  await expect(page.getByText(/^Version \S+ · bundle \w{8}$/)).toBeVisible({ timeout: 60000 });
}

const home = (page: Page) => page.locator('.start.home');
const pageRow = (page: Page) => page.locator('.prow').first();

test('browser Back and Forward move between screens; the tab bar stays hidden', async ({ page }) => {
  await open(page, false);
  await expect(page.locator('.tabbar')).toBeHidden();
  await page.locator('.hbtn', { hasText: 'Collection' }).click();
  await pageRow(page).click();
  await expect(page.locator('.crow2').first()).toBeVisible();
  await page.locator('.crow2').first().click();
  await expect(page.locator('.savedcard')).toBeVisible();

  await page.goBack();
  await expect(page.locator('.savedcard')).toHaveCount(0);
  await expect(page.locator('.crow2').first()).toBeVisible();
  await page.goBack();
  await expect(pageRow(page)).toBeVisible();
  await page.goBack();
  await expect(home(page)).toBeVisible();
  await page.goForward();
  await expect(page.locator('.screen.collection')).toBeVisible();

  // Closing a screen with its own control leaves no extra Back step.
  await page.locator('.homebtn').click();
  await expect(home(page)).toBeVisible();
  await page.locator('.hbtn', { hasText: 'Settings' }).click();
  await expect(page.locator('.screen.settings')).toBeVisible();
  await page.goBack();
  await expect(home(page)).toBeVisible();
  expect(await page.evaluate(() => history.length)).toBeLessThanOrEqual(3);
});

test('installed app: bottom tabs instead of the Home link, Back returns Home', async ({ page }) => {
  await open(page, true);
  const tabs = page.getByRole('navigation', { name: 'Main' });
  await expect(tabs).toBeVisible();
  await expect(page.locator('.homebtn')).toBeHidden();
  await expect(tabs.getByRole('button', { name: 'Home' })).toHaveAttribute('aria-current', 'page');
  await expect(page.locator('.install')).toHaveCount(0);
  await expect(page.locator('.hbtn', { hasText: 'Collection' })).toBeHidden(); // in the tab bar instead
  await expect(page.locator('.hbtn', { hasText: 'Settings' })).toBeHidden();
  const shots = process.env.SHOTS_DIR;
  if (shots) await page.screenshot({ path: `${shots}/nav-home.png` });

  await tabs.getByRole('button', { name: 'Collection' }).click();
  await expect(tabs.getByRole('button', { name: 'Collection' })).toHaveAttribute('aria-current', 'page');
  await pageRow(page).click();
  // The tab bar stays above the screen's content.
  const bar = await tabs.boundingBox(), screen = await page.locator('.screen.collection').boundingBox();
  expect(screen!.y + screen!.height).toBeLessThanOrEqual(bar!.y + 1);
  if (shots) await page.screenshot({ path: `${shots}/nav-collection.png` });
  await tabs.getByRole('button', { name: 'Settings' }).click();
  await expect(page.locator('.screen.settings')).toBeVisible();
  await page.goBack();
  await expect(home(page)).toBeVisible();
  await expect(tabs.getByRole('button', { name: 'Home' })).toHaveAttribute('aria-current', 'page');
});

test('Home offers the browser install prompt and can hide the offer', async ({ page }) => {
  await open(page, false);
  await expect(page.locator('.install')).toHaveCount(0); // desktop Chrome here: no prompt, not iOS
  const prompted = await page.evaluate(() => {
    const e = new Event('beforeinstallprompt', { cancelable: true }) as Event & { prompt(): Promise<void>; userChoice: Promise<unknown> };
    (window as unknown as { prompted: boolean }).prompted = false;
    e.prompt = async () => { (window as unknown as { prompted: boolean }).prompted = true; };
    e.userChoice = Promise.resolve({ outcome: 'dismissed' });
    dispatchEvent(e);
    return e.defaultPrevented;
  });
  expect(prompted).toBe(true); // Chrome's own install bar is suppressed
  await expect(page.locator('.install')).toBeVisible();
  await page.locator('.install').getByRole('button', { name: 'Install', exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as unknown as { prompted: boolean }).prompted)).toBe(true);
  await expect(page.locator('.install')).toHaveCount(0); // one prompt per page load

  await page.evaluate(() => {
    const e = new Event('beforeinstallprompt', { cancelable: true });
    Object.assign(e, { prompt: async () => {}, userChoice: Promise.resolve({ outcome: 'dismissed' }) });
    dispatchEvent(e);
  });
  await page.getByRole('button', { name: 'Hide install hint' }).click();
  await expect(page.locator('.install')).toHaveCount(0);
  await page.reload();
  await expect(home(page)).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem('tcg-install-dismissed'))).toBe('1');
});
