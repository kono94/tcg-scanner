# Browser scanner (pilot)

Vite + TypeScript scanner using the stable-frame cycle from `docs/CAPTURE_CYCLE.md`:
stability check → one capture → four-corner detection (YOLOv8s-pose) → perspective warp →
DINOv2/ArcFace embedding → exact cosine search → match/value policy → review. Inference
runs in a worker with onnxruntime-web: WebGPU with the fp32 encoder, falling back to WASM
with the int8 encoder. Status: pilot, verified on desktop Chrome with a test video only.
Thresholds are uncalibrated. The detector combines synthetic pretraining with AI-reviewed
real-corner fine tuning; phone performance remains unverified.

Current bundle `27047332` (`web/public/bundle/manifest.json` records the full provenance):
encoder from `runs/card_recognizer/retrieval/dinov2-arcface-gpu-20261017`, detector
`runs/card_detector/obb/corners-s640-v4-ft/weights/best.pt` at 640 px (box confidence 0.6), 24,038
English/German card rows plus 6 card-back rows.

## Setup

The fastest start is pulling the built artifacts with DVC (see the root README):
`dvc pull web/public/bundle web/public/media runs/browser_export/pilot-20261016`, then step 4.
To build them yourself (the script defaults still point at the 2026-10-16 pilot run and the
synthetic OBB detector, so pass the current run and corner weights explicitly):

```bash
# 1. Export encoder + gallery + parity fixtures (writes runs/browser_export/<name>/)
.env/bin/python scripts/export_browser_models.py \
  --run runs/card_recognizer/retrieval/dinov2-arcface-gpu-20261017 \
  --output runs/browser_export/gpu-20261017
# 1b. int8 encoder for the WASM path: onnxruntime dynamic quantization (per-channel
#     MatMul/Gemm weights) of encoder.onnx -> encoder.int8.onnx. No script is committed;
#     the manifest records the top-1 parity check (24/24 validation queries).
# 2. Assemble the versioned bundle (web/public/bundle, gitignored). Card-back rows are added
#    from datasets/curated/recognition/backs/manifest.json when present (--no-backs to omit);
#    scripts/acquire_card_backs.py (re)creates those references.
.env/bin/python scripts/build_browser_bundle.py \
  --export runs/browser_export/gpu-20261017 \
  --encoder-int8 runs/browser_export/gpu-20261017/encoder.int8.onnx \
  --detector-weights runs/card_detector/obb/corners-s640-v4-ft/weights/best.pt --detector-conf 0.6 \
  --prices runs/prices/prices.json --detector-origin "<what the weights were trained on>" \
  --german-pokemon <TCGdex /v2/de/cards list>   # German Pokémon versions of enrolled art
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

## Batch workflow (Settings → Scan workflow)

"Batch (review at the end)" scans page after page (auto when steady or manual). Per capture:
detect → the worker writes each card's canonical crop (PNG, encoder-contract pixels), display
crop and thumbnail → IndexedDB → the capture is released (worker buffer dropped) → the page is saved
to the Collection with its cards `queued` → back to the live preview. `src/batch/session.ts`
then embeds + searches one stored crop at a time (FIFO across pages, yields to single-page work,
paused while hidden) and writes each result into the saved card. The queue is rebuilt from the
saved pages at boot, so a reload or killed tab loses nothing. Admission control: no capture while
90 cards wait (resumes at 60). Auto mode waits for a page turn (scene moved and differs from the
last capture; uncalibrated threshold) before the next automatic capture. "Review" (finish the batch) lists
the cards that need the user and opens the Collection card screen for each. On desktop Chrome
(WebGPU) the stored-crop path gave identical top-1 identities and scores to direct recognition on
six cards of the test video; not measured on a phone.

## Phone reload diagnostics (dev server)

`debug.ts` keeps an "alive" marker in localStorage. Each boot logs `navigation` type and, for the
previous page of the origin, `previous-session` (ended with pagehide / bfcache / hmr-full-reload)
or `previous-session-unclean` (no pagehide: crash or tab killed) with its last breadcrumbs.
`hmr-full-reload` (client) and `server-file-*` (server) lines mark dev-server reloads, which also
reload a connected phone. Capture/page breadcrumbs carry what is alive on the UI thread
(display crops, still image, batch backlog); WebKit exposes no heap size.

"Save captures for training" (Settings, dev server only, off by default) POSTs each full capture
(JPEG) with its detections to `/__capture`, stored under `runs/captures/<date>/`
(`runs/captures-tests/` for automated runs). Not present in production builds.

## Card backs

Card backs are recognized by the normal search: the bundle appends gallery rows for
`back:pokemon` (international back) and `back:onepiece` (blue main-deck back), embedded
with the same fp32 encoder and tcg-full-card-v1 contract (`manifest.json` → `backs`,
`gallery.back_rows`). A confident back hit (score ≥ 0.65, margin ≥ 0.10 to the best
card; uncalibrated) gives the card state `back`: shown as a small `Back` pill and in
the page counts (`7 ready · 2 backs`), never matched, priced, reviewed or saved (in batch mode the card is removed from its saved page). See
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

Unit tests read fixtures from `runs/browser_export/pilot-20261016` and the sibling
`pose-fixture` (override with `TCG_EXPORT_DIR`). Without them, the parity suites are skipped rather than passed.

## Layout

- `src/vision/`: tcg-full-card-v1 warp/resize/normalize (bit-exact with OpenCV on the fixtures), pose (four-corner) decode, legacy OBB decode, NMS.
- `src/recognition/`: gallery search, catalog families, match policy, value ranges.
- `src/capture/`: stability monitor and capture cycle (pure, timestamp-driven).
- `src/session/page.ts`: page instances, one bounded recognizer, retries, timeouts, generation guards.
- `src/batch/`: batch workflow (capture → stored crops → background recognition queue persisted in the Collection).
- `src/inference/`: worker protocol, worker, client.
- `src/ui/`: Darkroom view, crop-flight presenter, settings.
- `src/prices/`: replaceable price service (bundled snapshot).
