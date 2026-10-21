import { describe, expect, it } from 'vitest';
import { Catalog, type CatalogCard } from '../src/recognition/catalog';
import { candidates, decideMatch, userChoiceMatch, type MatchResult } from '../src/recognition/policy';
import { formatQuote, formatRange, summarizeValue, type Quote } from '../src/recognition/value';

const card = (id: string, family: string, name = 'Shanks'): CatalogCard => {
  const [game, language, cardId] = id.split(':');
  return { id, game, language, cardId, name, family, finish: 'unknown', printing: 'unknown' };
};
const catalog = new Catalog([
  card('onepiece:en:OP01-120', 'onepiece:OP01-120'),
  card('onepiece:ja:OP01-120', 'onepiece:OP01-120'),
  card('onepiece:en:OP01-120_p1', 'onepiece:OP01-120_p1'),
  card('onepiece:en:OP01-078', 'onepiece:OP01-078', 'Boa Hancock'),
  card('onepiece:ja:OP01-078', 'onepiece:OP01-078', 'Boa Hancock'),
  card('pokemon:en:sv03-224', 'pokemon:en:sv03-224', 'Charizard'),
]);
const quote = (amount: number): Quote => ({ amount, currency: 'USD', basis: 'test', asOf: '2026-05-10', source: 'test' });
const hits = (...pairs: [string, number][]) => pairs.map(([identity, score], row) => ({ identity, score, row }));
const matched = (d: ReturnType<typeof decideMatch>) => { expect(d.kind).toBe('matched'); return d as MatchResult; };

describe('match policy', () => {
  it('assumes the preferred language when language is not observed', () => {
    const m = matched(decideMatch(hits(['onepiece:ja:OP01-120', 0.8], ['onepiece:en:OP01-120', 0.79], ['pokemon:en:sv03-224', 0.4]), catalog, { preferredLanguage: 'en' }));
    expect(m.preferred?.id).toBe('onepiece:en:OP01-120');
    expect(m.languageAssumed).toBe(true);
    expect(m.languageEvidence).toBe('not-read');
    expect(m.plausible.map((c) => c.language).sort()).toEqual(['en', 'ja']);
    expect(m.unresolved).toContain('language');
    // Margin is measured to the next artwork, not the other-language version.
    expect(m.competitor?.identity).toBe('pokemon:en:sv03-224');
  });

  it('keeps an observed language over the preference', () => {
    const m = matched(decideMatch(hits(['onepiece:en:OP01-120', 0.8], ['pokemon:en:sv03-224', 0.4]), catalog, { preferredLanguage: 'en', observedLanguage: 'ja' }));
    expect(m.preferred?.language).toBe('ja');
    expect(m.languageAssumed).toBe(false);
    expect(m.unresolved).not.toContain('language');
  });

  it('does not treat alternate art (_p1) as the same artwork', () => {
    const d = decideMatch(hits(['onepiece:en:OP01-120', 0.7], ['onepiece:en:OP01-120_p1', 0.68]), catalog, { preferredLanguage: 'en' });
    expect(d.kind).toBe('match-unclear');
  });

  it('reports not-found below the score floor and when there are no hits', () => {
    expect(decideMatch(hits(['pokemon:en:sv03-224', 0.3]), catalog, { preferredLanguage: 'en' }).kind).toBe('not-found');
    expect(decideMatch([], catalog, { preferredLanguage: 'en' }).kind).toBe('not-found');
  });

  it('falls back to the matched version when the preferred language has no entry', () => {
    const m = matched(decideMatch(hits(['pokemon:en:sv03-224', 0.8], ['onepiece:en:OP01-078', 0.3]), catalog, { preferredLanguage: 'ja' }));
    expect(m.preferred?.id).toBe('pokemon:en:sv03-224');
    expect(m.languageFallback).toBe(true);
  });
});

describe('value policy', () => {
  const shanks = matched(decideMatch(hits(['onepiece:en:OP01-120', 0.8], ['pokemon:en:sv03-224', 0.4]), catalog, { preferredLanguage: 'en' }));

  it('flags a valuable compatible alternative even in another language', () => {
    const v = summarizeValue(shanks, new Map([['onepiece:en:OP01-120', quote(12)], ['onepiece:ja:OP01-120', quote(160)]]));
    expect(v.preferred?.amount).toBe(12);
    expect([v.low, v.high]).toEqual([12, 160]);
    expect(v.cue).toMatchObject({ kind: 'check-variant', amount: 160 });
    expect(v.status).toBe('known');
  });

  it('keeps an unpriced version unknown, never zero', () => {
    const v = summarizeValue(shanks, new Map([['onepiece:en:OP01-120', quote(12)], ['onepiece:ja:OP01-120', null]]));
    expect(v.low).toBe(12);
    expect(v.high).toBeNull();
    expect(v.rangeIncomplete).toBe(true);
    expect(v.status).toBe('partial');
    expect(v.cue?.kind).toBe('variant-unpriced');
  });

  it('reports value unknown without quotes', () => {
    const v = summarizeValue(shanks, new Map());
    expect(v.status).toBe('unknown');
    expect(v.low).toBeNull();
  });

  it('excludes unrelated expensive search hits from the range and the cue', () => {
    const v = summarizeValue(shanks, new Map([['onepiece:en:OP01-120', quote(12)], ['onepiece:ja:OP01-120', quote(14)], ['pokemon:en:sv03-224', quote(900)]]));
    expect(v.high).toBe(14);
    expect(v.cue).toBeNull();
  });

  it('marks prices converted from the other currency as estimates', () => {
    const conv = { ...quote(1.05), currency: 'EUR', converted: { from: 'USD', rate: 0.88865, date: '2026-10-16', source: 'test' } };
    const v = summarizeValue(shanks, new Map([['onepiece:en:OP01-120', conv], ['onepiece:ja:OP01-120', { ...quote(3), currency: 'EUR' }]]));
    expect(v.converted).toBe(true);
    expect(formatQuote(v.preferred!)).toBe('≈€1.05');
    expect(formatRange(v)).toBe('≈€1.05–€3.00');
  });
});

describe('user choices', () => {
  it('confirms the artwork but keeps the language open unless picked explicitly', async () => {
    const { userChoiceMatch } = await import('../src/recognition/policy');
    const art = userChoiceMatch(catalog, 'onepiece:en:OP01-120', [], 'en', false);
    expect(art.languageAssumed).toBe(true);
    expect(art.plausible).toHaveLength(2);
    const exact = userChoiceMatch(catalog, 'onepiece:ja:OP01-120', [], 'en', true);
    expect(exact.languageAssumed).toBe(false);
    expect(exact.languageEvidence).toBe('user');
    expect(exact.plausible.map((c) => c.id)).toEqual(['onepiece:ja:OP01-120']);
  });

  it('lists distinct artworks as candidates with their language versions', async () => {
    const { candidates } = await import('../src/recognition/policy');
    const list = candidates(hits(['onepiece:ja:OP01-120', 0.8], ['onepiece:en:OP01-120', 0.79], ['onepiece:en:OP01-078', 0.5]), catalog, 'en');
    expect(list.map((c) => c.card.id)).toEqual(['onepiece:en:OP01-120', 'onepiece:en:OP01-078']);
    expect(list[0].versions).toHaveLength(2);
  });
});

describe('price service markets', () => {
  it('serves the selected market of a schema-2 snapshot and keeps missing prices unknown', async () => {
    const { SnapshotPriceService } = await import('../src/prices/service');
    const file = { schema_version: 2, snapshot_date: '2026-10-17', markets: {
      USD: { currency: 'USD', source: 'TCGplayer', quotes: { a: { amount: 2, basis: 'x' } } },
      EUR: { currency: 'EUR', source: 'Cardmarket', quotes: { a: { amount: 1.5, basis: 'y' } } } } };
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async () => ({ ok: true, json: async () => file })) as unknown as typeof fetch;
    let market = 'EUR';
    const s = new SnapshotPriceService('prices.json', () => market);
    expect((await s.quotes(['a', 'b'])).get('a')).toMatchObject({ amount: 1.5, currency: 'EUR' });
    expect((await s.quotes(['b'])).get('b')).toBeNull();
    market = 'USD';
    expect((await s.quotes(['a'])).get('a')).toMatchObject({ amount: 2, currency: 'USD' });
    globalThis.fetch = realFetch;
  });
});

describe('language scoped by the setting (EN/DE)', () => {
  it('uses the selected language version as the card, not an assumption', () => {
    const cat = new Catalog([
      card('pokemon:en:sv03-224', 'pokemon:sv03-224', 'Revavroom ex'),
      card('pokemon:de:sv03-224', 'pokemon:sv03-224', 'Knattox ex'),
      card('onepiece:en:OP01-078', 'onepiece:OP01-078', 'Boa Hancock'),
    ]);
    const m = matched(decideMatch(hits(['pokemon:en:sv03-224', 0.8], ['onepiece:en:OP01-078', 0.3]), cat, { preferredLanguage: 'de', languageFromSetting: true }));
    expect(m.preferred?.id).toBe('pokemon:de:sv03-224');
    expect(m.plausible.map((c) => c.id)).toEqual(['pokemon:de:sv03-224']);
    expect(m.languageAssumed).toBe(false);
    expect(m.languageEvidence).toBe('setting');
    // One Piece has no German version: the English card is the card, priced, nothing assumed.
    const op = matched(decideMatch(hits(['onepiece:en:OP01-078', 0.8], ['pokemon:en:sv03-224', 0.3]), cat, { preferredLanguage: 'de', languageFromSetting: true }));
    expect(op.preferred?.id).toBe('onepiece:en:OP01-078');
    expect(op.languageFallback).toBe(true);
    expect(op.languageAssumed).toBe(false);
    const v = summarizeValue(op, new Map([['onepiece:en:OP01-078', { ...quote(4), currency: 'EUR' }]]));
    expect(v.preferred?.amount).toBe(4);
  });
});

describe('card backs', () => {
  const back = (game: string): CatalogCard => ({ id: `back:${game}`, game, language: 'any', cardId: 'back', name: 'Card back', family: `back:${game}`, finish: 'n/a', printing: 'n/a', kind: 'back' });
  const withBacks = new Catalog([
    card('onepiece:en:OP01-120', 'onepiece:OP01-120'),
    card('pokemon:en:sv03-224', 'pokemon:en:sv03-224', 'Charizard'),
    back('pokemon'), back('onepiece'),
  ]);

  it('decides a back, never a card match, when a back is clearly the best hit', () => {
    const d = decideMatch(hits(['back:pokemon', 0.82], ['back:onepiece', 0.5], ['pokemon:en:sv03-224', 0.3]), withBacks, { preferredLanguage: 'en' });
    expect(d.kind).toBe('back');
    if (d.kind !== 'back') return;
    expect(d.game).toBe('pokemon');
    // Margin is measured to the best real card, not to the other game's back.
    expect(d.competitor?.identity).toBe('pokemon:en:sv03-224');
    expect(d.margin).toBeCloseTo(0.52);
    expect(d.thresholds.calibrated).toBe(false);
  });

  it('leaves a weak or ambiguous back hit unresolved instead of matching a card', () => {
    const weak = decideMatch(hits(['back:onepiece', 0.5], ['onepiece:en:OP01-120', 0.3]), withBacks, { preferredLanguage: 'en' });
    expect(weak.kind).toBe('match-unclear');
    const close = decideMatch(hits(['back:onepiece', 0.7], ['onepiece:en:OP01-120', 0.65]), withBacks, { preferredLanguage: 'en' });
    expect(close.kind).toBe('match-unclear');
    const low = decideMatch(hits(['back:onepiece', 0.3]), withBacks, { preferredLanguage: 'en' });
    expect(low.kind).toBe('not-found');
  });

  it('treats a close back hit as a competitor for a card match', () => {
    const d = decideMatch(hits(['onepiece:en:OP01-120', 0.7], ['back:onepiece', 0.68]), withBacks, { preferredLanguage: 'en' });
    expect(d.kind).toBe('match-unclear');
    const m = matched(decideMatch(hits(['onepiece:en:OP01-120', 0.8], ['back:onepiece', 0.3]), withBacks, { preferredLanguage: 'en' }));
    expect(m.plausible.map((c) => c.id)).toEqual(['onepiece:en:OP01-120']);
  });

  it('never offers a back as a candidate, a search result or a user choice', () => {
    const list = candidates(hits(['back:pokemon', 0.9], ['pokemon:en:sv03-224', 0.4]), withBacks, 'en');
    expect(list.map((c) => c.card.id)).toEqual(['pokemon:en:sv03-224']);
    expect(withBacks.search('back', 'en')).toEqual([]);
    expect(withBacks.search('card back', 'en')).toEqual([]);
    expect(() => userChoiceMatch(withBacks, 'back:pokemon', [], 'en', false)).toThrow('card back');
  });
});

describe('artwork matches (printing open)', () => {
  const art = (id: string, family: string): CatalogCard => ({ id, game: 'yugioh', language: 'en', cardId: '89631139', name: 'Blue-Eyes White Dragon', family, finish: 'unknown', printing: 'unknown', kind: 'artwork' });
  const printing = (id: string, family: string, rarity: string): CatalogCard => ({ id, game: 'yugioh', language: 'en', cardId: id.split(':')[2], name: 'Blue-Eyes White Dragon', family, finish: rarity, printing: 'Set', ref: 'x' });
  const ygo = new Catalog([
    art('yugioh:en:89631139', 'yugioh:card:89631139'),
    art('yugioh:en:89631140', 'yugioh:card:89631139'),
    printing('yugioh:en:LOB-EN001:UR', 'yugioh:card:89631139', 'Ultra Rare'),
    printing('yugioh:en:SDK-001:UR', 'yugioh:card:89631139', 'Ultra Rare'),
    printing('yugioh:en:LCKC-EN001:C', 'yugioh:card:89631139', 'Common'),
    art('yugioh:en:46986414', 'yugioh:card:46986414'),
  ]);

  it('shows the matched artwork, keeps every printing plausible and the printing open', () => {
    const m = matched(decideMatch(hits(['yugioh:en:89631140', 0.8], ['yugioh:en:89631139', 0.79], ['yugioh:en:46986414', 0.5]), ygo, { preferredLanguage: 'en' }));
    expect(m.preferred?.id).toBe('yugioh:en:89631140');
    expect(m.plausible.map((c) => c.id).sort()).toEqual(['yugioh:en:LCKC-EN001:C', 'yugioh:en:LOB-EN001:UR', 'yugioh:en:SDK-001:UR']);
    expect(m.unresolved).toEqual(['printing']);
    expect(m.languageAssumed).toBe(false);
    // The other artwork of the same card is not the competitor.
    expect(m.competitor?.identity).toBe('yugioh:en:46986414');
  });

  it('prices only the printings: the artwork itself has no quote, a valuable printing raises the cue', () => {
    const m = matched(decideMatch(hits(['yugioh:en:89631139', 0.8], ['yugioh:en:46986414', 0.5]), ygo, { preferredLanguage: 'en' }));
    const v = summarizeValue(m, new Map([['yugioh:en:LOB-EN001:UR', quote(120)], ['yugioh:en:SDK-001:UR', quote(40)], ['yugioh:en:LCKC-EN001:C', quote(0.5)]]));
    expect(v.preferred).toBeNull();
    expect([v.low, v.high, v.status]).toEqual([0.5, 120, 'known']);
    expect(v.cue).toMatchObject({ kind: 'check-variant', amount: 120 });
  });

  it('measures the spread from the cheapest printing when the artwork has no price', () => {
    const m = matched(decideMatch(hits(['yugioh:en:89631139', 0.8], ['yugioh:en:46986414', 0.5]), ygo, { preferredLanguage: 'en' }));
    const value = (prices: [string, number][]) => summarizeValue(m, new Map(prices.map(([id, a]) => [id, quote(a)])));
    // Every printing is valuable, none 3x the cheapest: no cue.
    expect(value([['yugioh:en:LOB-EN001:UR', 40], ['yugioh:en:SDK-001:UR', 30], ['yugioh:en:LCKC-EN001:C', 25]]).cue).toBeNull();
    // A cheap and a valuable printing: cue.
    expect(value([['yugioh:en:LOB-EN001:UR', 45], ['yugioh:en:LCKC-EN001:C', 0.5]]).cue).toMatchObject({ kind: 'check-variant', amount: 45 });
    // Only one price known: a valuable one still raises the cue (unpriced printings may be cheap).
    expect(value([['yugioh:en:LOB-EN001:UR', 45]]).cue).toMatchObject({ kind: 'check-variant', amount: 45 });
    // Small spreads never do.
    expect(value([['yugioh:en:LOB-EN001:UR', 3], ['yugioh:en:LCKC-EN001:C', 0.1]]).cue).toBeNull();
  });

  it('a picked printing is the exact version, with its artwork hit as evidence', () => {
    const m = userChoiceMatch(ygo, 'yugioh:en:SDK-001:UR', hits(['yugioh:en:89631139', 0.8], ['yugioh:en:46986414', 0.5]), 'en', true);
    expect(m.plausible.map((c) => c.id)).toEqual(['yugioh:en:SDK-001:UR']);
    expect(m.top.identity).toBe('yugioh:en:89631139');
    expect(m.unresolved).toEqual([]);
  });

  it('a picked artwork leaves the printing open', () => {
    const m = userChoiceMatch(ygo, 'yugioh:en:89631139', [], 'en', false);
    expect(m.preferred?.id).toBe('yugioh:en:89631139');
    expect(m.plausible).toHaveLength(3);
    expect(m.unresolved).toEqual(['printing']);
  });
});
