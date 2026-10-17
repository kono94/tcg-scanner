# Match uncertainty, preferred language and value ranges

Status: agreed product direction, 2026-10-16. Candidate-family resolution, range
pricing and these controls are planned; they are not implemented in the app or
retrieval pilot.

## Language preference and candidate identity

Settings include **Preferred card language**, separately from interface language,
currency and market. Clear observed language overrides that preference. When
language is unknown, prefer a compatible candidate in the selected language for
ordering, metadata and a provisional estimate, labeled **English assumed** (or
the selected language). Do not label the fallback **English detected**.

Preserve compatible candidates in other languages and their value implications.
Do not filter them out before retrieval or hide a valuable alternative. If the
preferred language has no compatible entry or quote, show unavailable rather than
substituting another physical printing silently. A preference must not increase
recognition confidence, confirm an exact ID or become a training label.

Keep physical instance, underlying card, artwork, catalog printing, language,
finish and condition separate. Catalog adapters must establish cross-language
and printing-family relationships explicitly; a shared character name or base
code alone does not establish the same artwork. Preserve the candidate IDs,
supporting evidence, unresolved attributes and exclusion reasons. A raw top-K
list is a proposal, not automatically the plausible candidate set. Cosine
similarities are not probabilities. Price must not make a weak match credible.

Suggested fields: `observed_language`, `language_evidence_status`,
`preferred_language`, `effective_display_language`, `language_assumed`,
`preferred_candidate`, `plausible_candidates`, `unresolved_attributes`, and
`resolution_provenance`. Save assumed results as provisional. An explicit user
selection records user confirmation separately from observed evidence. Changing
preferences must not overwrite confirmed evidence or create another copy.

## Value presentation

Every card has a persistent value-status indicator. Selected-card details and
page review always expose a range or **Value unknown**:

- **Preferred estimate:** price/range for the preferred compatible interpretation,
  with language, finish and condition assumptions visible.
- **Plausible versions range:** an envelope across all supported compatible
  language/printing/finish candidates, including versions not distinguishable in
  the image. Keep it available even when one interpretation is preferred.
- **Coverage/context:** priced versus plausible candidate count, source/market,
  currency, condition/finish basis, quote basis and update time.

The versions range is not a confidence interval, guaranteed sale price or expected
value. Do not probability-weight prices using cosine scores. Do not combine
currencies, raw and graded cards, or incompatible conditions/quote bases into an
unlabeled range. Any currency conversion records its own rate/source/time.
Graded alternatives stay separate unless physical grading is established.

Missing prices are unknown, never zero. Partial coverage shows **Range incomplete**;
an unpriced plausible candidate prevents claiming a known upper bound. Stale,
missing or failed lookups remain visible without endless waits. One supported
version still shows its available estimate/range and resolution status.

Keep price retrieval behind a replaceable service, independent of recognition.
Tag quotes with physical instance, capture/candidate revision and provenance so
late responses cannot reprice another card or an obsolete candidate set.

## Valuable-variant review and single-card investigation

Show **Check variant** when a supported compatible alternative could materially
change the value and language, edition, stamp or finish remains unresolved.
Provide configurable valuable-card and material-range-spread thresholds; defaults
need evaluation. Use all plausible candidates, not just the preferred-language
estimate. A weak unrelated expensive search hit must not trigger the cue.
Unknown quotes produce **Variant unpriced**, not a fabricated high-value claim.

Explain which alternative causes the cue and the distinguishing detail. Offer
**Inspect this card** for a closer single-card capture of the language, code,
stamp or foil detail. Preserve the rest of the binder page and update that same
physical instance; never count a rescan as an additional copy or automatically
assign a definitive condition/grade.

Keep corner brackets and compact instance badges. Every instance has a small,
non-color-only value/review indicator; do not cover nine card faces with nine
range labels. Details/review expose the full range and assumptions. Price and
recognition progress are separate: recognition must not keep spinning for prices.
Page totals separate confirmed values, provisional ranges and unpriced cards;
a provisional upper bound is not the collection's confirmed value.

Illustrative UI labels, not actual market quotes:

```text
Artwork matched · English assumed · Finish unconfirmed
Preferred estimate: €6–€12
Plausible versions: €6–€180 · 3 of 4 versions priced
Check variant · Range incomplete
[Inspect this card]
```

## Card backs (implemented in the browser pilot, 2026-10-17)

Binder pages often show a card face down. Backs are enrolled as special gallery
rows (`back:pokemon`, `back:onepiece`; catalog `kind: 'back'`, own family, no
price), so the normal similarity search recognizes them:

- Best hit is a back with score ≥ 0.65 and a margin ≥ 0.10 to the best real
  card: the instance becomes **Card back** (session state `back`). Uncalibrated
  pilot values (`PILOT_BACK_THRESHOLDS`, chosen from
  `scripts/evaluate_card_backs.py`; see `web/README.md`).
- A weaker back hit is never turned into a card match: it stays
  `match-unclear`/`not-found` (needs attention). A back close behind a card hit
  counts as the competitor for the card margin.
- A back is terminal and settled, but it is not a card: it is not priced, not
  reviewed or listed in the check flow, not counted as ready or needs-attention,
  does not make a page partial, and is **not saved** to the collection (so it
  never enters collection counts or totals). Page counts show it separately,
  e.g. `7 ready · 2 backs`; the overlay badge is a small `Back` pill.
- Backs are never offered as candidates, manual-search results or user choices.
  The user can still correct a misread back to a card by picking a front.
- Not covered: Japanese/Asian Pokémon backs, One Piece DON!! backs, empty
  pockets and other non-cards (these stay unresolved as before).

## Acceptance and evaluation

- Unknown language plus English preference: English is assumed; a compatible
  Japanese higher-value alternative stays visible and can trigger review.
- Readable Japanese plus English preference: Japanese stays selected.
- Unresolved stamp/finish: show the supported range and identifying detail,
  without inventing an exact printing.
- Unrelated expensive top-K hit: exclude it from plausible versions and the cue.
- Unpriced plausible variant: incomplete range, unknown upper bound, never zero.
- Delayed/stale/missing prices: other cards finish and quote coverage stays visible.
- Closer capture, preference change or late quote: preserve physical-copy counts,
  confirmed evidence and candidate/capture revisions.

Report artwork-family recognition separately from exact language/printing
resolution, fallback usage, price coverage and review-cue precision/coverage.
Preference-assisted presentation does not improve measured exact-ID accuracy.
