// Value status, preferred estimate and plausible-versions range (docs/MATCH_AND_VALUE_POLICY.md).
// Missing prices are unknown, never zero. Ranges are envelopes, not confidence intervals.
import type { CatalogCard } from './catalog';
import type { MatchResult } from './policy';

/** A quote estimated from the other market's price at a recorded exchange rate. */
export interface Conversion {
  from: string;
  rate: number;
  date: string;
  source: string;
}

export interface Quote {
  amount: number;
  currency: string;
  basis: string;
  asOf: string;
  source: string;
  /** Set when this market had no own price for the version. */
  converted?: Conversion;
}

export interface ValueSettings {
  /** An alternative at or above this amount can be materially valuable. */
  valuableAmount: number;
  /** Alternative/preferred ratio that counts as a material spread. */
  spreadRatio: number;
}

/** Defaults need evaluation; they are product placeholders. */
export const DEFAULT_VALUE_SETTINGS: ValueSettings = { valuableAmount: 20, spreadRatio: 3 };

export type ValueCue =
  | { kind: 'check-variant'; alternative: CatalogCard; amount: number }
  | { kind: 'variant-unpriced'; alternative: CatalogCard }
  | null;

export interface ValueSummary {
  status: 'known' | 'partial' | 'unknown';
  currency: string | null;
  preferred: Quote | null;
  low: number | null;
  /** Null when any plausible version is unpriced: the upper bound is unknown. */
  high: number | null;
  priced: number;
  plausible: number;
  rangeIncomplete: boolean;
  cue: ValueCue;
  asOf: string | null;
  source: string | null;
  /** Some priced version's quote is converted from the other currency. */
  converted: boolean;
}

export function summarizeValue(match: MatchResult, quotes: ReadonlyMap<string, Quote | null>, settings = DEFAULT_VALUE_SETTINGS): ValueSummary {
  const versions = match.plausible;
  const priced = versions.map((c) => ({ card: c, quote: quotes.get(c.id) ?? null })).filter((v): v is { card: CatalogCard; quote: Quote } => v.quote !== null);
  const currencies = new Set(priced.map((p) => p.quote.currency));
  if (currencies.size > 1) throw new Error('Mixed currencies need an explicit conversion');
  const amounts = priced.map((p) => p.quote.amount);
  const complete = priced.length === versions.length;
  const preferred = match.preferred ? quotes.get(match.preferred.id) ?? null : null;
  const languageOpen = match.unresolved.includes('language');
  let cue: ValueCue = null;
  if (versions.length > 1) {
    const alternatives = versions.filter((c) => c.id !== match.preferred?.id);
    const valuable = priced
      .filter((p) => p.card.id !== match.preferred?.id)
      .filter((p) => p.quote.amount >= settings.valuableAmount && (!preferred || p.quote.amount >= preferred.amount * settings.spreadRatio))
      .sort((a, b) => b.quote.amount - a.quote.amount)[0];
    if (valuable) cue = { kind: 'check-variant', alternative: valuable.card, amount: valuable.quote.amount };
    else if (languageOpen) {
      const unpriced = alternatives.find((c) => !quotes.get(c.id));
      if (unpriced) cue = { kind: 'variant-unpriced', alternative: unpriced };
    }
  }
  return {
    status: priced.length === 0 ? 'unknown' : complete ? 'known' : 'partial',
    currency: priced[0]?.quote.currency ?? null,
    preferred,
    low: amounts.length ? Math.min(...amounts) : null,
    high: complete && amounts.length ? Math.max(...amounts) : null,
    priced: priced.length,
    plausible: versions.length,
    rangeIncomplete: !complete,
    cue,
    asOf: priced[0]?.quote.asOf ?? null,
    source: priced[0]?.quote.source ?? null,
    converted: priced.some((p) => p.quote.converted),
  };
}

export function formatMoney(amount: number, currency: string, estimated = false): string {
  const symbol = currency === 'USD' ? '$' : currency === 'EUR' ? '€' : currency + ' ';
  const digits = amount < 10 ? 2 : 0;
  return (estimated ? '≈' : '') + symbol + amount.toFixed(digits);
}

/** The preferred estimate, marked with ≈ when converted from the other currency. */
export function formatQuote(q: Quote): string {
  return formatMoney(q.amount, q.currency, !!q.converted);
}

export function formatRange(v: ValueSummary): string {
  if (v.low === null || !v.currency) return 'Value unknown';
  const low = formatMoney(v.low, v.currency, v.converted);
  if (v.high === null) return `${low} – unknown`;
  return v.high === v.low ? low : `${low}–${formatMoney(v.high, v.currency)}`;
}
