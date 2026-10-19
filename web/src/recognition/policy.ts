// Match decision and preferred-language resolution (docs/MATCH_AND_VALUE_POLICY.md).
// Retrieval identifies an artwork family; language/printing stay unresolved unless observed.
import { Catalog, isBack, type CatalogCard } from './catalog';
import type { Hit } from './gallery';

export interface MatchThresholds {
  /** Minimum cosine similarity of the best hit. */
  minScore: number;
  /** Minimum gap to the best hit outside the matched artwork family. */
  minMargin: number;
  /** False until thresholds come from an independent calibration set. */
  calibrated: boolean;
}

/** Placeholder defaults. Not calibrated and not chosen from test results. */
export const DEFAULT_THRESHOLDS: MatchThresholds = { minScore: 0.45, minMargin: 0.05, calibrated: false };

export interface BackThresholds {
  /** Minimum cosine similarity of the best card-back reference. */
  minScore: number;
  /** Minimum gap from the best card-back hit to the best real-card hit. */
  minMargin: number;
  calibrated: boolean;
}

/**
 * Uncalibrated defaults for card backs (scripts/evaluate_card_backs.py, 2026-10-17): the 298
 * real front crops scored at most 0.34 against a back row and the closest card references (Poké
 * Ball trainers) 0.57, while held-out real back scans and synthetic photo-like back views scored
 * at least 0.77 with a margin of at least 0.35 to the best card. Few, partly synthetic positives
 * and no phone captures: not a calibrated operating point.
 */
export const DEFAULT_BACK_THRESHOLDS: BackThresholds = { minScore: 0.65, minMargin: 0.1, calibrated: false };

/** 'setting': the user scoped the app to one card language (EN or DE) in Settings. */
export type LanguageEvidence = 'observed' | 'not-read' | 'user' | 'setting';

export interface MatchResult {
  kind: 'matched';
  top: Hit;
  /** Best hit from a different artwork family, used for the margin. */
  competitor: Hit | null;
  margin: number;
  /** Every catalog version compatible with the matched artwork (plausible candidates). */
  plausible: CatalogCard[];
  observedLanguage: string | null;
  languageEvidence: LanguageEvidence;
  preferredLanguage: string;
  /**
   * Preferred compatible interpretation. When the wanted language has no version of this artwork
   * (e.g. no German One Piece), the matched version stands in (`languageFallback`), so a matched
   * card always has a version to price. Null only for an empty family.
   */
  preferred: CatalogCard | null;
  /** The wanted language has no version of this artwork; `preferred` is the matched one instead. */
  languageFallback: boolean;
  languageAssumed: boolean;
  unresolved: string[];
  thresholds: MatchThresholds;
}

export interface NoMatch {
  kind: 'not-found' | 'match-unclear';
  top: Hit | null;
  competitor: Hit | null;
  margin: number;
  thresholds: MatchThresholds;
}

/** The crop shows a card back: not a card match, never priced or counted as a card. */
export interface BackResult {
  kind: 'back';
  /** Best card-back hit (identity `back:<game>`). */
  top: Hit;
  game: string;
  /** Best real-card hit, used for the margin. */
  competitor: Hit | null;
  margin: number;
  thresholds: BackThresholds;
}

export type MatchDecision = MatchResult | NoMatch | BackResult;

export interface MatchOptions {
  preferredLanguage: string;
  /** Language read from the card (OCR or user); null when not established. */
  observedLanguage?: string | null;
  /** The user collects one language: that version is the card (not an assumption among alternatives). */
  languageFromSetting?: boolean;
  thresholds?: MatchThresholds;
  backThresholds?: BackThresholds;
}

export function decideMatch(hits: readonly Hit[], catalog: Catalog, options: MatchOptions): MatchDecision {
  const thresholds = options.thresholds ?? DEFAULT_THRESHOLDS;
  const top = hits[0] ?? null;
  if (!top) return { kind: 'not-found', top: null, competitor: null, margin: 0, thresholds };
  const topCard = catalog.get(top.identity);
  if (isBack(topCard)) {
    // A back is never a card match: either a confident back, or unresolved.
    const back = options.backThresholds ?? DEFAULT_BACK_THRESHOLDS;
    const competitor = hits.find((h) => !isBack(catalog.get(h.identity))) ?? null;
    const margin = competitor ? top.score - competitor.score : top.score;
    if (top.score >= back.minScore && margin >= back.minMargin) {
      return { kind: 'back', top, game: topCard!.game, competitor, margin, thresholds: back };
    }
    return { kind: top.score < thresholds.minScore ? 'not-found' : 'match-unclear', top, competitor, margin, thresholds };
  }
  const family = catalog.family(top.identity);
  const familyIds = new Set(family.map((c) => c.id));
  // Ambiguity is measured against the next distinct artwork, not another version of the same one.
  const competitor = hits.find((h) => !familyIds.has(h.identity)) ?? null;
  const margin = competitor ? top.score - competitor.score : top.score;
  if (top.score < thresholds.minScore) return { kind: 'not-found', top, competitor, margin, thresholds };
  if (margin < thresholds.minMargin) return { kind: 'match-unclear', top, competitor, margin, thresholds };

  const observed = options.observedLanguage ?? null;
  const wanted = observed ?? options.preferredLanguage;
  const inWanted = family.find((c) => c.language === wanted) ?? null;
  const languages = new Set(family.map((c) => c.language));
  const preferred = inWanted ?? catalog.get(top.identity) ?? family[0] ?? null;
  const languageFallback = inWanted === null;
  const scoped = !observed && !!options.languageFromSetting && inWanted !== null;
  const unresolved: string[] = [];
  if (!observed && !scoped && languages.size > 1) unresolved.push('language');
  if (family.some((c) => c.finish === 'unknown')) unresolved.push('finish');
  return {
    kind: 'matched', top, competitor, margin,
    plausible: scoped ? [preferred!] : family,
    observedLanguage: observed,
    languageEvidence: observed ? 'observed' : scoped ? 'setting' : 'not-read',
    preferredLanguage: options.preferredLanguage,
    preferred,
    languageFallback,
    // Without observed language, a preferred-language interpretation is an assumption, not a detection.
    // A fallback is no assumption when the artwork exists in a single language only.
    languageAssumed: !observed && !scoped && preferred !== null && (!languageFallback || languages.size > 1),
    unresolved,
    thresholds,
  };
}

/** The card shown for a match: the preferred interpretation, else the visual top hit. */
export function displayCard(match: MatchResult, catalog: Catalog): CatalogCard | undefined {
  return match.preferred ?? catalog.get(match.top.identity);
}

/**
 * A result the user picked. Picking the artwork confirms the card; the language counts as
 * confirmed only when the user picked a language version explicitly, otherwise the other
 * language versions stay plausible. Machine evidence stays separate (`top` keeps the
 * retrieval hit for that entry when there was one).
 */
export function userChoiceMatch(catalog: Catalog, identity: string, hits: readonly Hit[], preferredLanguage: string,
  languageConfirmed: boolean, thresholds: MatchThresholds = DEFAULT_THRESHOLDS): MatchResult {
  const card = catalog.get(identity);
  if (!card) throw new Error(`Unknown catalog entry ${identity}`);
  if (isBack(card)) throw new Error(`${identity} is a card back, not a card`);
  const versions = catalog.family(identity);
  const family = new Set(versions.map((c) => c.id));
  const languageOpen = !languageConfirmed && new Set(versions.map((c) => c.language)).size > 1;
  const top = hits.find((h) => h.identity === identity) ?? { identity, row: -1, score: NaN };
  const competitor = hits.find((h) => !family.has(h.identity)) ?? null;
  return {
    kind: 'matched', top, competitor, margin: competitor && Number.isFinite(top.score) ? top.score - competitor.score : NaN,
    plausible: languageOpen ? versions : [card],
    observedLanguage: languageOpen ? null : card.language, languageEvidence: languageOpen ? 'not-read' : 'user', preferredLanguage,
    preferred: card, languageFallback: false, languageAssumed: languageOpen,
    unresolved: [...(languageOpen ? ['language'] : []), ...(card.finish === 'unknown' ? ['finish'] : [])], thresholds,
  };
}

export interface Candidate {
  family: string;
  /** Representative entry shown for the artwork: preferred language when available. */
  card: CatalogCard;
  /** Every catalog entry with this artwork (language versions). */
  versions: CatalogCard[];
  score: number;
}

/** Distinct artwork candidates from a hit list, best first. */
export function candidates(hits: readonly Hit[], catalog: Catalog, preferredLanguage: string, limit = 6, preferGame: string | null = null): Candidate[] {
  const out: Candidate[] = [];
  const seen = new Set<string>();
  for (const h of hits) {
    const card = catalog.get(h.identity);
    if (!card || isBack(card) || seen.has(card.family)) continue;
    seen.add(card.family);
    const versions = catalog.family(h.identity);
    out.push({ family: card.family, card: versions.find((v) => v.language === preferredLanguage) ?? card, versions, score: h.score });
  }
  // Display order only: cards of the expected game first (score order within each group). This
  // never changes the match decision; other games stay available further along.
  if (preferGame) out.sort((a, b) => Number(b.card.game === preferGame) - Number(a.card.game === preferGame));
  return out.slice(0, limit);
}

/** The game most cards resolved to (for candidate display order); null without a clear majority. */
export function dominantGame(games: readonly (string | null | undefined)[]): string | null {
  const counts = new Map<string, number>();
  for (const g of games) if (g) counts.set(g, (counts.get(g) ?? 0) + 1);
  const ranked = [...counts].sort((a, b) => b[1] - a[1]);
  if (!ranked.length || (ranked[1] && ranked[1][1] * 2 > ranked[0][1])) return null;
  return ranked[0][0];
}
