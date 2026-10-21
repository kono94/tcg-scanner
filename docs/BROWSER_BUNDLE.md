# Current browser bundle

The bundle in `web/public/bundle` (DVC). Update this page when a new bundle ships and move
the superseded text to `archive/docs/` (the previous page is
[BROWSER_BUNDLE_c73c459e7a24eaf2.md](../archive/docs/BROWSER_BUNDLE_c73c459e7a24eaf2.md)).

Bundle `09304b039491978b` (2026-10-21) adds English Yu-Gi-Oh! to `c73c459e7a24eaf2` without
retraining: 14,777 artwork gallery rows and 44,658 printings (set code × rarity) as catalog-only
versions, from `scripts/catalog/enroll_yugioh_gallery.py`. The models (fp16 encoder on WebGPU,
int8 encoder on WASM, fp16 `corners-s640-v4-ft` detector, confidence 0.6, NMS IoU 0.7) and the
Pokémon/One Piece rows are unchanged. The detector now comes from the export folder instead of a
per-build re-export, so rebuilding with the same inputs gives the same bundle version.

| Content | Rows / entries |
| --- | --- |
| Pokémon + One Piece card rows (EN, with German Pokémon versions) | 24,038 |
| Yu-Gi-Oh! artwork rows (EN) | 14,777 |
| Card-back rows | 7 (Pokémon 3, One Piece 3, Yu-Gi-Oh! 1) |
| Catalog entries | 101,754 (`catalog.json` 27.8 MB) |
| `gallery.f16` | 38,822 rows, 29.8 MB |
| Game logos (`logos/`, from `scripts/catalog/acquire_game_logos.py`) | 3 |

## Yu-Gi-Oh!

- Retrieval evidence: [YUGIOH_ZERO_SHOT.md](research/YUGIOH_ZERO_SHOT.md) (163/179 reviewed
  video crops top-1, 35 distinct cards, heavy-foil Quarter Century Secret Rares weakest). In the
  browser (WASM, int8 encoder) a Magician of Faith crop scores 0.824 against its artwork (0.827
  fp32 in Python); scoped to Pokémon + One Piece its best hit is 0.354, below the match threshold.
- A match is an artwork with the printing open: every printing of the card is a plausible version.
  Prices are YGOPRODeck set prices in USD for 12,362 of 44,658 printings (zero or missing prices stay
  unknown); EUR values are converted from them and marked ≈. Many Yu-Gi-Oh! ranges are therefore
  incomplete.
- Users choose the games to recognize on Home/Settings (default Pokémon + One Piece), shown with
  the official logos; search uses only the selected games' rows.
- Card back `back:yugioh`: one flat TCG back image (Wikipedia, pinned in
  `scripts/catalog/acquire_card_backs.py`). Four YGOPRODeck entries whose image is a card back
  (placeholders) are excluded from the gallery (similarity ≥ 0.8 to the back, listed in
  `enrolled/yugioh.json`). Afterwards Yu-Gi-Oh! references reach at most 0.589 against the back and
  all 195 video crops at most 0.115 (back threshold 0.65, margin 0.10, uncalibrated). `scripts/release/evaluate_card_backs.py`:
  24/24 synthetic photo-like views of the Yu-Gi-Oh! back become `back:yugioh` (score ≥ 0.84, margin
  ≥ 0.27; not independent, the only reference is in the gallery); none of the 298 real Pokémon/One
  Piece front crops becomes a back. The
  Japanese OCG back has no reference.

| Path | Encoder | Detector | Model download |
| --- | --- | --- | --- |
| WebGPU | `encoder.fp16.onnx` 43.9 MB (40.4 gzip) | `detector.fp16.onnx` 23.0 MB (21.2 gzip) | 67 MB |
| WASM (all platforms) | `encoder.int8.onnx` 23.5 MB (20.7 gzip) | `detector.fp16.onnx` | 47 MB |

The fp32 `encoder.onnx`/`detector.onnx` stay in the bundle as the reference the variants were
converted from; the app does not download them when the manifest lists fp16 copies.

## Precision checks

`scripts/release/compare_model_precisions.py` writes the fp16 models and
`models/browser-export/gpu-20261017/precision_report.json`. Conversion:
`onnxruntime.transformers.float16` with float32 inputs/outputs; the encoder's final L2
normalization stays float32 (in fp16 its epsilon underflows and the embedding becomes NaN).

- Encoder, 298 AI-reviewed real query crops (188 train / 24 val / 86 test) against the
  bundle gallery: fp32, fp16 and int8 each rank the exact ID first for 292 and within the
  top five for 298. fp16 keeps the fp32 top-1 on 298/298 (min cosine to fp32 0.99999, top-1
  score change ≤ 0.0006). int8 keeps it on 296/298 (mean cosine 0.991, top-1 score change
  up to 0.056, margin change up to 0.067).
- Detector, 28 held-out real-corner frames (156 cards): fp32 and fp16 both find 139 within
  5% corner error (mean 2.38% vs 2.39% of the card diagonal). On 75 unlabeled Yu-Gi-Oh!
  video frames both return 210 detections, 209 within 1%.
- onnxruntime-web 1.30 WASM (Node): fp16 encoder cosine 0.999997 to fp32 on a parity
  fixture; fp16 detector raw outputs within 1.3 px (640 px input) of fp32.
- WASM memory (Node, single thread, one encoder): fp32 639 MB RSS, fp16 569 MB, int8
  397 MB, all about 650 ms per card. fp16 weights do not reduce WASM memory much, so the
  CPU path uses int8.

These are retrieval and corner diagnostics on repeated video views with AI-reviewed labels,
not scanner accuracy. In-browser WebGPU execution of the fp16 models, iPhone memory and phone
latency are unverified. `npm run typecheck`, 122 web unit/parity tests, 80 Python tests and
the 9 Playwright flows (desktop Chrome, test-video camera mock) pass.

The detector is re-exported from its `.pt` weights at every build, which yields new ONNX
bytes (identical outputs), so each rebuild changes the bundle version. The builder converts
that export to fp16 and requires its outputs to match the evaluated `detector.fp16.onnx` on
the parity scenes.
