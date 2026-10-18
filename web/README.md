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
`dvc pull web/public/bundle web/public/media runs/browser_export/gpu-20261017`, then step 4.
To build them yourself (defaults select the current GPU run and fine-tuned corner detector):

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

Open `http://localhost:5173`. Choose **Scan**, **New binder…**, or a **Test video**. `?mock=<file>`
starts a test video directly. The video mock plays a local file and passes
`captureStream()` into the same path as `getUserMedia`. It is labeled in the UI and is
not a camera benchmark. Phones need HTTPS for camera access; a laptop's LAN HTTP
address does not count as localhost on the phone.

## Home, binders and the Collection

Home shows **Continue <binder>**, the workflow choice (**Single review** = review each
capture before the next, the default; **Batch review** = scan many pages, review at the end)
with a one-line explanation, then Collection (its subtitle reads `€X · N cards · M binders`:
matched cards with a known price, unknown prices never counted), Settings, **New binder…**
(optional name) and **Finish binder**. Binder headings in Collection show name and value,
the details below, then **Continue scanning**, **Rename** and **Delete** in one row.
Confirmations and renaming use an in-app modal dialog (`src/ui/dialog.ts`, native
`<dialog>`), not the browser's `confirm()`/`prompt()`.

The UI says binder → page → card. In code and storage a binder is still a scan session
and a page a capture/`SavedPage`. A new unnamed binder is called `Binder N`. Rename or
continue any binder from its Collection heading. Collection has two views: **Binders**
(binder → pages) and **All cards** (every saved card sorted by current value, with a
**Needs a check** filter). Sessions persist
across reloads and days. Starting another session parks the previous one; it does
not delete it, except an unnamed session that never saved a capture. On upgrade, the
legacy session stays active only if it was used within its old three-hour window. Both scan workflows share the session's capture counter.

Saved captures appear as **Page 1**, **Page 2**, etc. A retry revises the
current capture; repeated saves are idempotent. New sessions start at 1; finishing
a batch, switching workflows, deleting a capture or reloading does not reset the
counter. Existing saved groups migrate without changing their keys, numbering,
quantities or stored crops. Existing duplicate display numbers are preserved.
Each binder in Collection also has **Delete**, including binders with no saved pages. The in-app confirmation removes that group's saved cards,
crop storage and session metadata, and retires its queue work. Other collections
and already uploaded training originals are unaffected.

## Camera lifecycle and responsiveness

One stream is reused during foreground scanning and short Home/Settings/Collection/
review visits. When camera pixels are unnecessary (including snapshot processing,
ordinary review and batch extraction), tracks are disabled, video playback/frame callbacks and
stability sampling stop. Returning to acquisition requires fresh nonblank pixels
and resets the stability dwell. Finishing a session, hiding or leaving the browser
page ends tracks and detaches the video. Model sessions remain independent.

Concurrent camera opens are coalesced; obsolete permission results are closed.
No automatic permission retry loop runs. Browser permission persistence is outside
the app's control: reuse avoids repeat requests during short navigation, but one
prompt across all reloads/restarts cannot be guaranteed. Physical Safari camera
indicator, power consumption, focus/restart timing and sustained performance still
need controlled checks. The iPhone 13 mini feedback is recorded in
[the work log](../docs/PHONE_FEEDBACK_2026-10-17.md).

Rendering follows state changes and active presenter animations instead of an
unconditional frame loop throughout inference. Diagnostics log camera-open and
first nonblank-preview timing; these do not measure autofocus completion. Repeat
mode resumes camera delivery when acquiring the next replacement capture.

## Batch workflow (Settings → Scan workflow)

"Batch review" scans page after page (auto when steady or manual). In auto mode, entering
the scanner (e.g. **Continue <binder>**) only opens the live preview: automatic scanning
starts with **Start scanning** under the preview, **Pause scanning** stops it (`Scanning
paused` with a `Paused` box over the preview; **Continue scanning** resumes), and leaving the
scanner or losing the camera stops it again (a manual shutter button stays available
while it runs). Start scans the page in view after the steady dwell, without a page-turn
gate. Per capture:
detect → the worker writes each card's canonical crop (PNG, encoder-contract pixels), display
crop and thumbnail → IndexedDB → the capture is released (worker buffer dropped) → the page is saved
to the Collection with its cards `queued` → back to the live preview. During
processing and camera warm-up, one display-sized frozen capture stays visible;
muting the stream never intentionally exposes its black output. The full worker
capture is still released after extraction. `src/batch/session.ts`
then embeds + searches one stored crop at a time (FIFO across pages, yields to single-page work,
paused while hidden) and writes each result into the saved card. The queue is rebuilt from the
saved pages at boot, so a reload or killed tab loses nothing. Admission control: no capture while
90 cards wait (resumes at 60). Auto mode waits for a page turn (scene moved and differs from the
last capture; uncalibrated threshold) before the next automatic capture. When frame
delivery was suspended during extraction, a fresh steady image that differs from
the captured reference can establish a page turn that occurred during the gap.
The unchanged page remains gated, and the scene is rechecked up to the actual
capture rather than latching a temporary change. Uniform exposure shifts are
removed from the luma comparison. After a saved (or skipped repeat) page, a page-turn window
(Settings → Advanced → Time to turn the page, default 5 s; settings revision 3 raises
stored values below 5 s once) starts when the saved photo is replaced by the live preview.
During it no capture happens and a large `Turn the page · N s` instruction with an emptying
bar sits over the preview. Then `Hold steady · Scan in X s` with a filling bar: the steady
dwell counts only from the end of the window, so steadiness while turning does not
trigger a capture. If the scene still matches the previous page, `Turn the page · Waiting
for page N` stays up. A capture that found no cards only waits 1.5 s. The window is
separate from steady dwell and single-capture repeat cooldown. Timings are uncalibrated
and not device-verified.

After detection the found cards are outlined on the frozen capture. When the page is
stored, the frozen photo shows `Page N saved · K cards · turn the page` for at least 1.5
seconds and until fresh preview pixels arrive, instead of a `Starting camera` flash, so a
missed card can be seen before the page is turned. Each capture also gives a short
vibration (Settings → Capture feedback). Browsers without the Vibration API (iOS Safari)
play a soft click instead, once audio was unlocked by a tap. Neither is device-verified.

Before admission, rectified card appearances are compared one-to-one with saved
captures in the same named session. A repeat found by an **automatic** capture is
skipped silently: its crops are removed, nothing is saved, and a note says
`Already scanned (page N) · turn the page`; auto capture waits for a page turn again.
A repeat found by a **manual** Scan now is a **possible repeat**: excluded from saved
counts and background recognition, its reduced crops held for explicit
**Keep anyway**, which also works after reload. While a repeat is held, capture (auto and manual),
finishing the batch and switching or finishing the session wait for **Keep anyway**
or **Dismiss**; only Dismiss (or deleting its collection) discards the candidate.
Overridden captures record that decision.
These are uncalibrated appearance heuristics, not physical-card identity or a
guarantee of duplicate detection. Identical-looking physical pages may need
Keep anyway. Existing saved crops acquire signatures lazily with a bounded
48-card decoding budget per attempt; missing old crops cannot establish a match.

"Review" (finish the batch) lists
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
New uploads also carry persistent scan-session, capture and source-visit IDs,
plus reported browser user agent and execution provider. A source visit is one
document load; repeated physical views across visits must still be grouped during
dataset review. Uploaded geometry is an unverified detector proposal, not a label.

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

Unit tests read fixtures from `runs/browser_export/gpu-20261017` and the sibling
`pose-fixture` (override with `TCG_EXPORT_DIR`). These fixtures exercise the current GPU encoder and pose detector. Without them, the parity suites are skipped rather than passed.

## Layout

- `src/vision/`: tcg-full-card-v1 warp/resize/normalize (bit-exact with OpenCV on the fixtures), pose (four-corner) decode, legacy OBB decode, NMS.
- `src/recognition/`: gallery search, catalog families, match policy, value ranges.
- `src/capture/`: stability monitor and capture cycle (pure, timestamp-driven).
- `src/session/page.ts`: page instances, one bounded recognizer, retries, timeouts, generation guards.
- `src/batch/`: batch workflow (capture → stored crops → background recognition queue persisted in the Collection).
- `src/inference/`: worker protocol, worker, client.
- `src/ui/`: Darkroom view, crop-flight presenter, settings.
- `src/prices/`: replaceable price service (bundled snapshot).
