# Optional OCR fast pass and identity evidence

Latest direction, 2026-10-15: **similarity search handles the normal case**. OCR is an optional fast pass when crop quality is exceptional, and selective evidence for ambiguous shortlists. It must not become a required recognition stage or a slow retry loop ahead of retrieval. This shared pipeline uses catalog-specific identity rules; no user transcription or manual dataset labeling is required.

```text
capture → detect physical cards → source-resolution crop/rectification + quality
  → ordinary usable crop: encoder → similarity shortlist → selective verification
  → exceptionally clear crop: optional bounded OCR/code shortcut
      → unique supported code: catalog candidate
      → absent/ambiguous text or exhausted budget: similarity path immediately
  → insufficient/conflicting evidence: unresolved, never a forced exact label
```

We do not need to know the printed ID's location in advance: a text detector proposes regions; recognition supplies strings and confidence; syntax and catalog lookup find plausible ID evidence. Keep the bounding polygons/boxes and crop/source transforms with every token. An OCR result anywhere on a card is not necessarily its ID: attack damage, HP, copyright years, set totals and explanatory text also contain numbers.

PaddleOCR with text detection/oriented regions is a sensible backend to benchmark. It is not installed or integrated yet. Tesseract is available locally with English/OSD/numeric models; Japanese recognition is not currently available, although Latin printed codes may still be readable. Record the backend, language/model/version, region confidence and source pixel size. OCR improvements cannot restore characters that glare, blur or low source resolution has erased.

## Identity rules

| Game | Observed tokens | Required caution |
| --- | --- | --- |
| One Piece | `OP01-001`, `ST01-001`, `EB01-001`, `PRB01-001`, `P-001` | Printed base code may be shared by alternate artwork/printings. `_p1` is a gallery variant identifier, not a code OCR can invent from the printed base number. |
| Pokémon | Collector fraction such as `001/198`, `TG01/TG30`, set code/symbol, language and visible name | Collector fraction alone can recur across sets; foil/reverse/edition may not be resolved by the same digital front. Bare numerals are insufficient. |
| Yu-Gi-Oh! | Eight-digit passcode, printing set code, language/edition markings | Passcode identifies the underlying card; exact printing needs additional evidence. One passcode can have many artworks/sets/rarities. |
| Magic/sports, later | Game-specific identifiers and visual/categorical evidence | Keep extension points; do not claim these games are supported without catalog adapters and independent evaluation. |

Preserve raw observed strings. Whitespace/case normalization may be explicit, but no silent O/0 substitutions, missing-digit completion or fuzzy nearest-code assignment. A valid code absent from the enrolled catalog stays unresolved/not-enrolled; it must not be replaced with a visually similar enrolled card. If two plausible codes disagree, try another available clear view or stronger OCR automatically, then quarantine if still unresolved.

Direct resolution means a candidate at a declared catalog granularity, not proof of a genuine physical card or every finish/printing. A source image can be a proxy or a broadcast graphic. Keep identity, printing, physical instance, reference and evidence provenance separate.

A user's preferred card language is a display/pricing fallback only; it cannot
supply missing OCR evidence or label training data. Keep observed language unknown
when unreadable, label the assumption, and let clear observed language override
the preference. See [candidate/value policy](MATCH_AND_VALUE_POLICY.md).

## Fallback and data labels

Visual retrieval normally supplies the shortlist and can disambiguate artwork variants. Exact cosine search with a shared frozen encoder is the initial baseline; score and distinct-identity margin require calibration and are not probabilities. A selective OCR/detail pass can verify ambiguous candidates. Code/visual conflict is an uncertainty signal, not a reason to choose whichever score is larger. Skip the OCR shortcut when quality is insufficient; calibrate eligibility and a strict work budget from real crops rather than inventing universal thresholds.

Offline label proposals can combine readable code, catalog constraints, visual agreement and independent nearby views of the same physical card. Short-range propagation must stop at cuts, swaps or ambiguous occlusion. Stronger vision/LLM teachers can adjudicate candidates; their output remains auditable pseudo-label evidence. Low-evidence frames can still support generic card detection or unlabeled/self-supervised training.

Acquisition prefers 2160p/4K with a hard minimum 1440p available source format. Also retain crop dimensions/pixels-per-card, blur/glare and readable-detail signals: nominal video resolution alone does not guarantee correct IDs. Keep original videos/creator/session groups together across splits; adjacent frames and reuploads cannot provide independent validation.

## Current implementation boundary

`ocr_identity.py` / `scripts/propose_card_labels.py` implement offline whole-card
Tesseract regions, bounded PSM 7 retries on observed ID-like regions, catalog lookup
and 0/180-degree orientation candidates. The retired frozen-MobileNet fallback was
removed. `identity_evidence.py` separately checks proposed visual/OCR agreement,
shared-token ambiguity and conflicts. PaddleOCR remains unimplemented. The browser
uses DINOv2/ArcFace similarity retrieval; this offline OCR tool is not its scheduler.

Evidence: region retries on 24 original One Piece reference fixtures produced seven exact high-confidence code reads, with four unique catalog candidates, two alternate-art ambiguities and one conflict. On 16 actual source-resolution GroundingDINO crop proposals, OCR resolved zero catalog identities. The observed Japanese Pokémon fraction `048/063` could not resolve against the English catalog. Preserve these failures rather than inferring that larger crops guarantee readable IDs. The browser now uses the fine-tuned retrieval encoder; no calibrated real-video exact-label accuracy or universal encoder performance has been established.
