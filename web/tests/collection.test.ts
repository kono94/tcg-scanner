import { describe, expect, it } from 'vitest';
import { scanAmountIn, totals, withCurrentPrices, type SavedPage } from '../src/collection/store';
import type { PriceService } from '../src/prices/service';
import type { Quote } from '../src/recognition/value';

const q = (amount: number, currency: string, converted = false): Quote =>
  ({ amount, currency, basis: 'test', asOf: '2026-10-17', source: 'test', ...(converted ? { converted: { from: 'USD', rate: 0.9, date: '2026-10-16', source: 'test' } } : {}) });

function service(table: Record<string, Quote>, currency: string): PriceService {
  return { description: 'test', currency: () => currency, quotes: async (ids) => new Map(ids.map((id) => [id, table[id] ?? null])) };
}

const page = (cards: SavedPage['cards']): SavedPage =>
  ({ key: 'p', sessionId: 's', pageNumber: 1, revision: 1, savedAt: '2026-10-17T10:00:00Z', partial: false, cards });

describe('collection prices', () => {
  it('reprices saved cards in the current market, including old pages without version lists', async () => {
    const saved = page([
      { number: 1, identity: 'onepiece:en:OP02-117_p6', name: 'Ice Age', cardId: 'OP02-117_p6', state: 'confirmed', languageAssumed: false, checkedByUser: false,
        versions: ['onepiece:en:OP02-117_p6'], preferredId: 'onepiece:en:OP02-117_p6', checkVariant: false,
        value: { currency: 'USD', preferred: 1.0, low: 1.0, high: 1.0, cue: false }, thumb: null },
      { number: 2, identity: 'onepiece:en:OP01-120', name: 'Shanks', cardId: 'OP01-120', state: 'confirmed', languageAssumed: false, checkedByUser: false,
        value: { currency: 'USD', preferred: 10, low: 10, high: 10, cue: false }, thumb: null },
    ]);
    const [p] = await withCurrentPrices([saved], service({ 'onepiece:en:OP02-117_p6': q(1.02, 'EUR', true), 'onepiece:en:OP01-120': q(12, 'EUR') }, 'EUR'));
    expect(p.cards[0].current).toMatchObject({ currency: 'EUR', preferred: 1.02, converted: true, cue: false });
    expect(p.cards[1].current).toMatchObject({ currency: 'EUR', preferred: 12, high: 12 });
    expect(totals([p])).toMatchObject({ currency: 'EUR', confirmed: 13.02, unpriced: 0 });
    expect(scanAmountIn(saved.cards[1].value, 'EUR', 0.9)).toBeCloseTo(9);
  });

  it('counts an assumed language as confirmed only when all language versions cost the same', () => {
    const card = (number: number, low: number, high: number): SavedPage['cards'][number] => ({ number, identity: `pokemon:en:c${number}`, name: 'C', cardId: `c${number}`,
      state: 'confirmed', languageAssumed: true, checkedByUser: false, value: { currency: 'EUR', preferred: low, low, high, cue: false }, thumb: null });
    expect(totals([page([card(1, 2, 2), card(2, 1, 5)])])).toMatchObject({ confirmed: 2, provisionalHigh: 5 });
  });

  it('keeps a missing current price unknown, never zero', async () => {
    const saved = page([{ number: 1, identity: 'pokemon:de:x', name: 'X', cardId: 'x', state: 'confirmed', languageAssumed: false, checkedByUser: false,
      versions: ['pokemon:de:x'], preferredId: 'pokemon:de:x', value: null, thumb: null }]);
    const [p] = await withCurrentPrices([saved], service({}, 'USD'));
    expect(p.cards[0].current?.preferred).toBeNull();
    expect(totals([p]).unpriced).toBe(1);
  });
});
