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
# 2. Assemble the versioned bundle (web/public/bundle, gitignored)
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
