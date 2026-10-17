# Browser scanner (pilot)

Vite + TypeScript scanner using the stable-frame cycle from `docs/CAPTURE_CYCLE.md`:
stability check → one capture → OBB detection → perspective warp → DINOv2/ArcFace
embedding → exact cosine search → match/value policy → review. Inference runs in a
worker with onnxruntime-web (WebGPU, falling back to WASM). Status: pilot. Thresholds
are uncalibrated and the detector was trained on synthetic scenes only.

## Setup

The fastest start is pulling the built artifacts with DVC (see the root README):
`dvc pull web/public/bundle web/public/media runs/browser_export/pilot-20261016`, then step 4.
To build them yourself:

```bash
# 1. Export models + parity fixtures (writes runs/browser_export/<name>/)
.env/bin/python scripts/export_browser_models.py --output runs/browser_export/pilot-20261016
# 2. Assemble the versioned bundle (web/public/bundle, gitignored). Card-back rows are added
#    from datasets/curated/recognition/backs/manifest.json when present (--no-backs to omit);
#    scripts/acquire_card_backs.py (re)creates those references.
.env/bin/python scripts/build_browser_bundle.py
# 3. Optional: local test videos for the camera mock (web/public/media, gitignored)
node web/scripts/link-media.mjs
# 4. Run
cd web && npm install && npm run dev
```

Open `http://localhost:5173`. Choose **Use camera** or a **Test video**. `?mock=<file>`
starts a test video directly. The video mock plays a local file and passes
`captureStream()` into the same path as `getUserMedia`. It is labeled in the UI and is
not a camera benchmark. Phones need HTTPS for camera access; a laptop's LAN HTTP
address does not count as localhost on the phone.

## Card backs

Card backs are recognized by the normal search: the bundle appends gallery rows for
`back:pokemon` (international back) and `back:onepiece` (blue main-deck back), embedded
with the same fp32 encoder and tcg-full-card-v1 contract (`manifest.json` → `backs`,
`gallery.back_rows`). A confident back hit (score ≥ 0.65, margin ≥ 0.10 to the best
card; uncalibrated) gives the card state `back`: shown as a small `Back` pill and in
the page counts (`7 ready · 2 backs`), never matched, priced, reviewed or saved. See
`docs/MATCH_AND_VALUE_POLICY.md`.

`scripts/evaluate_card_backs.py` (2026-10-17, gpu-20261017 encoder, EN/DE gallery of
24,038 card rows + 6 back rows): none of the 298 real front crops (train/val/test) is
decided as a back or changes its top-1 (0 of 86 test queries changed); their best back
score is at most 0.34 (median 0.05). Over all card reference rows the closest to a back
is 0.57 (Poké Ball trainer art). Positives: 12/12 held-out real Pokémon back scans and
432/432 synthetic photo-like back views (perspective, glare, sleeve sheen, blur, JPEG,
corner noise; variants of a gallery reference scored leave-one-reference-out) are
decided as the right game's back, minimum score 0.77. The positives are few and partly
synthetic, with no phone captures of backs; the held-out real scans share one scan
session with two gallery references. The detector's recall on backs in binder photos is
unmeasured (on the 16-back scan it returned 9 card-sized boxes at 640 px).

## Checks

```bash
npm run typecheck
npm test          # parity with Python/OpenCV/Ultralytics fixtures, policy, cycle, session, presenter
npm run e2e       # Chrome via Playwright: test-video scan end to end
```

Unit tests read fixtures from `runs/browser_export/pilot-20261016` (override with
`TCG_EXPORT_DIR`). Without them, the parity suites are skipped rather than passed.

## Layout

- `src/vision/`: tcg-full-card-v1 warp/resize/normalize (bit-exact with OpenCV on the fixtures), OBB decode/NMS.
- `src/recognition/`: gallery search, catalog families, match policy, value ranges.
- `src/capture/`: stability monitor and capture cycle (pure, timestamp-driven).
- `src/session/page.ts`: page instances, one bounded recognizer, retries, timeouts, generation guards.
- `src/inference/`: worker protocol, worker, client.
- `src/ui/`: Darkroom view, crop-flight presenter, settings.
- `src/prices/`: replaceable price service (bundled snapshot).
