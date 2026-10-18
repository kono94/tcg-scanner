# TCG Scanner Agent Guide

## Read First

- Read this guide before architectural changes and `docs/DEVELOPMENT.md` before choosing new work.
- The product is the browser scanner in `web/`; read `web/README.md` before changing it and `deploy/README.md` before hosting changes.
- Data and training are automated: synthetic scenes with known identities/corners, AI-reviewed real video crops, acquired catalog references. Do not require the user to capture or label data manually. Expand automated real-domain validation in parallel; label synthetic results honestly.
- For training on the LAN GPU machine, read `docs/GPU_TRAINING.md`.
- `docs/REPO_AUDIT_2026-10-15.md` is a historical snapshot of the repository before the retrieval/browser work.
- For scanner interaction and overlays, read `docs/SCANNER_UX.md`.
- For preferred card language, ambiguous matches, price ranges and valuable-variant review cues, read `docs/MATCH_AND_VALUE_POLICY.md`.
- For encoder training, gallery enrollment, or retrieval changes, read `docs/RECOGNITION_DESIGN.md`.
- For pretrained localization/encoder comparisons, read `docs/FOUNDATION_MODELS.md`; keep detector proposals separate from exact identities and verified labels.
- For camera triggering and batch lifecycle, read `docs/CAPTURE_CYCLE.md`.
- For collection, annotation, and dataset schemas, read `docs/DATA_COLLECTION.md`.
- For image acquisition or SAMPLE watermark removal, read `docs/IMAGE_SOURCES.md`. Preserve originals and mark generated restorations as synthetic; do not silently promote them into canonical references or independent evaluation.
- Distinguish implemented behavior, historical experiment results, proposals, and device-verified behavior in reports.

## Product Direction

Build a scanner that is easy to share through a phone-browser URL: camera capture, card localization, perspective correction, exact-card retrieval, local metadata, and a replaceable price service. The browser pilot in `web/` is the product (bundle `27047332`, 2026-10-17: YOLOv8s-pose four-corner detector `corners-s640-v4-ft/weights/best.pt`, GPU-trained DINOv2-small/ArcFace encoder `dinov2-arcface-gpu-20261017` with an int8 copy for the WASM path, 24,038 EN/DE card rows plus card-back rows). It is desktop-Chrome verified with a test-video camera mock, not yet device-verified, with uncalibrated thresholds and a detector fine-tuned on AI-reviewed real corners after synthetic pretraining. It deploys as a static site to a VPS (`deploy/`).

Target enrollment-based recognition: adding reference images and metadata should not require expanding a classifier head. This is an architectural capability, not a guarantee of accuracy on unseen cards or games. One Piece and Pokémon are enrolled; additional games require data and explicit evaluation.

Use **similarity retrieval as the normal recognition path** after localization. OCR is an optional, bounded fast pass for exceptionally clear crops and selective shortlist verification; do not require OCR to succeed or finish repeated retries before similarity runs. When attempted, detect text regions across the rectified card, parse observed game-aware ID tokens and resolve against catalog records without a fixed ID location. A clear uniquely supported code can avoid visual encoding; ambiguous/absent OCR proceeds to retrieval. Quality and time gates need calibration on real captures. PaddleOCR is a candidate backend, not installed. Never invent missing digits or infer a printing from a shared base number. See `docs/OCR_IDENTITY.md`.

Keep the architecture shared across games, establish the One Piece/Pokémon baseline first, and use Yu-Gi-Oh! as an explicit transfer evaluation. Defer Magic/sports expansion until those results; catalog adapters and printing-resolution rules remain game-specific.

Single cards and multiple cards are equally required from the first implementation. A binder page with nine cards is a primary acceptance scenario, not a later enhancement. Give each captured card an independent instance/state and show a clean instruction/progress overlay throughout capture, recognition, confirmation, and page review.

Card count is variable. Nine is a benchmark scenario, never a fixed input shape, required grid, annotation length, or maximum. Collect scenes with zero, one, and arbitrary multiple cards, including more than nine; detection emits a variable-length instance list. Recognition/gallery images represent individual cards, and encoder mini-batch size is independent of scene card count.

Prefer stable-frame capture as the first scanning mode: lightweight scene-stability check → capture → detect → recognize → review on that fixed image. Continuous per-card tracking is optional later work. Provide configurable auto-when-steady, manual, and optional repeat-while-steady triggering as specified in `docs/CAPTURE_CYCLE.md`.

Two scan workflows exist (Home → workflow choice). **Single review** recognizes and reviews each capture before the next. **Batch review** (auto scanning starts only on an explicit Start, with a page-turn window after each page) detects, stores each card's canonical crop in IndexedDB, saves the page with cards `queued`, releases the capture and returns to the preview; a persisted background queue recognizes stored crops one at a time and survives reloads. See `web/README.md`.

Card backs are enrolled as gallery rows (`back:pokemon`, `back:onepiece`). A confident back hit becomes the `back` state: shown and counted separately, never matched, priced, reviewed or saved as a card. Back thresholds are uncalibrated.

## Current Evidence Limits

- The original corpus has 1,399 One Piece reference images. The normalized acquired gallery now contains 4,373 English One Piece, 19,665 physical English Pokémon, 154 Japanese One Piece and five Japanese Pokémon references; provider gaps remain. See `docs/DATASET_CLEANUP.md` and the live manifests.
- The retired classifier manual-evaluation fixtures duplicated training images; historical metrics from them are not an independent camera benchmark.
- Do not describe existing validation scores as real-world accuracy or select deployment thresholds from them.
- The detector (YOLOv8s-pose, four card corners) now uses `corners-s640-v4-ft` at 640 px, with synthetic pretraining plus AI-reviewed real-corner fine tuning. The earlier n/s comparison used synthetic-only weights; its crop acceptance proxy is not measured recall. The new ONNX detector is 46.0 MB raw / 38.6 MB gzip; physical-phone performance remains unverified. See `docs/BROWSER_BUNDLE_2026-10-17.md`.
- Fine-tuned encoders on the full 24,197-reference catalog: the MPS pilot `dinov2-arcface-pilot-20261016` ranked the exact ID first for 19/24 validation and 84/86 test crops; the GPU run `dinov2-arcface-gpu-20261017` (6,000 classes, 4 trainable blocks, 2,100 steps) for 20/24 and 85/86. Validation is Pokémon and test is One Piece, from few videos with repeated views and AI-reviewed labels. These are retrieval diagnostics, not scanner accuracy, and unknown rejection is uncalibrated.
- Browser parity: encoder ONNX vs torch min cosine 0.9999996; the int8 encoder keeps top-1 on 24/24 validation queries (mean cosine 0.98 to fp32). The manifest marks in-browser runtime parity as unverified.

Historical results (their frozen-encoder tools and artifacts have been retired): frozen MobileNet reached 24.9% / 38.2% / 53.8% Recall@1/5/20 and frozen DINOv2-small 37.4% / 48.3% / 61.1% on 1,321 synthetic true-corner queries against a 1,431-reference pilot index. `ocr_identity.py` / `scripts/propose_card_labels.py` provide a Tesseract OCR baseline; PaddleOCR remains a proposed backend. Read `docs/FOUNDATION_MODELS.md` for their scope. Distinguish GroundingDINO localization from DINO-style image retrieval. Prefer the working `.env/bin/python`.

The six-worker expansion acquired 12 excerpts from 11 videos and retained 69 native-4K frames; 63 card-bearing frames produced 716 saved crop proposals. The completed crop review admits 298 SOL-reviewed, reference-matched single-card queries across 158 catalog-art identities, split 188 train / 24 val / 86 test by source video. All 519 candidates left after the first pass were reviewed: 262 admitted, 143 rejected, 45 identity-verified but geometry-excluded, 69 exact-identity unresolved. The cumulative cleanup removes 304 bad derived crops; 114 reviewed exclusions remain outside training. Repeated views are not distinct physical cards. These are AI-reviewed query labels, not measured recognition accuracy or independent real-corner ground truth. Pokémon catalog sync is resumable; read its live manifests rather than claiming complete coverage. Pocket digital records stay separate from physical TCG references.

Read `docs/VIDEO_DATA_AUDIT.md` for actual acquisition yield and the six-worker expansion. Distinguish source videos, retained frames, repeated physical views, raw boxes, geometry-eligible crops and exact labeled examples. Similarity is the normal path; do not require high-quality training views to yield OCR codes before they can be assessed for visual retrieval.

## Repo Map

- `web/`: the browser scanner (Vite + TypeScript, onnxruntime-web worker); see `web/README.md`.
- `deploy/`: VPS hosting (shared Caddy HTTPS proxy, static scanner stack, `deploy.sh` releases); see `deploy/README.md`.
- `datasets/curated/`, `datasets/staging/`: accepted and candidate data (DVC); see `docs/DATASET_LAYOUT.md`.
- `datasets/card_recognizer/`: original One Piece catalog corpus (Git LFS), retained for acquisition/provenance.
- `training/card_recognizer/`: retrieval trainer (`train_retrieval.py`), shared canonical preprocessing (`canonical.py`) and retrieval model/training helpers.
- `training/card_detector/`: synthetic camera effects and the Ultralytics trainer (`train_obb.py`, OBB and pose models).
- `scripts/`: acquisition, dataset curation, synthetic scenes, evaluation, browser export/bundle and price snapshots.
- `archive/`: original scraper notebook/source HTML and the dated historical roadmap.
- `resources/prices/`: price-index and snapshot inputs shared by browser bundle tooling.
- `runs/`: ignored local experiment artifacts; do not assume recorded checkpoints still exist. Selected runs, `datasets/curated`, `datasets/staging` and the web bundle are versioned with DVC (`*.dvc` pointers, store on the LAN GPU PC); see README "Data and models (DVC)". After rebuilding a tracked artifact, `dvc add` + `dvc push` and commit the pointer.
- `resources/processed/inference_example.mp4`: rendered demo with burned-in predictions, not clean accuracy-test input. `resources/benchmarks/manifest.json` records its provenance/limitations; raw replay sources and independent annotations still need to be supplied/collected.

## Implementation Rules

- Preserve unrelated work, Python tooling, notebooks and model artifacts. Do not rewrite notebooks for unrelated fixes.
- Keep capture, inference, geometry, tracking, recognition policy, metadata, prices, and UI separate.
- For the single-page snapshot mode, keep at most one active capture batch with no queued timer batches. All detections/crops/overlays refer to the same immutable capture. A repeat capture replaces candidate page results; it never appends new physical copies automatically.
- Batch mode is the sanctioned exception: each capture is reduced to stored crops and released before the next, and its background recognition queue is persisted, FIFO, yields to single-page work, pauses while hidden and has explicit admission control (no capture while 90 cards wait, resume at 60). Auto capture waits for a page turn. Keep the queue rebuildable from saved pages.
- Stability dwell and repeat cooldown are separate configurable settings. Reset the dwell on scene motion or stale/missing samples; do not infer scene stillness solely from phone sensors. Countdown expiry never bypasses busy/lifecycle/quality checks.
- Use a bounded latest-frame buffer. Admission control must happen before work is appended to a serial queue or worker message queue.
- Bound recognition work globally and per track. Coalesce pending crops; expire stale jobs; schedule tracks fairly.
- Support at least nine independently progressing card instances from the outset. A bounded worker may process them sequentially; one slow/unknown card must not block the others or cause confirmed cards to be repeatedly scheduled first.
- Derive counts from actual instances. Do not pad to nine, truncate at nine, require nine for capture/review, or equate an expected pocket layout with detections. Resource limits must be explicit and surface partial results; process larger lists in bounded chunks/lazily.
- Keep page, physical card instance, track, and catalog identity separate. Two pockets containing the same card ID are two copies. Save each page instance idempotently; do not count tracker churn or refreshes as new copies.
- Tag work with session/page generation, capture ID/timestamp, instance ID/revision, and model/index version; include track IDs/revisions only in tracking modes. Reject obsolete results; callbacks must never create retired state.
- Use monotonic time for intervals and frame age. Stop/reset/background transitions must invalidate work and release retained frames/crops.
- In tracking modes, associate detections and tracks one-to-one and specify expiry in elapsed time; do not let a successful tracker keep a lost card alive indefinitely. Snapshot mode requires no cross-frame object association.
- Confirm snapshot identities with calibrated retrieval/detail evidence and review; repeated inference on identical pixels is not independent confirmation. Tracking modes may additionally use recent independent captures. A highest-ever score must not permanently lock recognition or finalize an unconfirmed card.
- Name coordinate spaces explicitly: raw sensor pixels, oriented source pixels, detector input/letterbox, normalized detections, crop pixels, preview/view pixels.
- Carry transforms and orientation with frames. Test aspect-fill/object-fit cropping, mirroring, all supported rotations, and corner ordering outside views.
- The deployed detector predicts four ordered card corners (YOLOv8-pose) and crops by perspective warp; the earlier OBB path (rotated rectangle) remains a supported, comparable decoder. Train corners on upright scenes: sideways cards make corner names ambiguous, upside-down cards are fine because corner order is geometric. Replace the detector only when it wins `scripts/evaluate_detector_geometry.py` and `scripts/compare_detectors_real.py`.
- Preserve true physical-edge annotations; do not turn axis-aligned boxes into claimed corner ground truth or discard perspective corners after fitting an OBB. Keep geometric corner order distinct from upright artwork orientation.
- Version preprocessing: full-card policy, aspect ratio/padding, RGB order, resize/interpolation, tensor layout, and exact channel normalization. Query and gallery must use the same contract.
- Use camera-localization and known-gallery-bounds adapters feeding the same canonicalizer/encoder-input stages. Do not force verified digital references through a detector, randomly augment gallery inserts, crop away identity text, or mix trained preprocessing contracts. Validate golden crops/tensors/embeddings across runtimes and rebuild the gallery when its contract changes.
- Export parity checks must compare real outputs across runtimes, not just tensor shapes. Missing or failed checks must be reported as unverified.
- Keep price access behind a service protocol. Network requests do not belong in the camera/inference loop.
- Preferred card language is an explicit display/pricing fallback when observed language is unknown, never recognition evidence or a training label. Show `English assumed` or equivalent; preserve compatible alternatives and let observed language override the preference. A fallback must not silently confirm an exact printing.
- Every instance exposes a value status and a supported plausible-versions range in details/review. Keep preferred estimates distinct from the all-plausible-candidate range; do not hide valuable alternatives because they differ from the preferred language. Use compact `Check variant` indicators and a single-card investigation flow. Incomplete prices stay unknown, never zero; prices and recognition resolve independently. Follow `docs/MATCH_AND_VALUE_POLICY.md` for evidence, quote comparability and provisional saves.

## Scanner Overlay Rules

- Follow `docs/SCANNER_UX.md`. Use thin white acquisition outlines, a restrained white scan-line sweep on capture, compact per-card status badges, and one page-level instruction/progress area.
- Drive the UI from explicit semantic states: locating, stabilizing, queued, recognizing, verifying, confirmed, needs-attention, paused, and review. Do not infer state from confidence colors or label strings.
- A scan-line animation is capture feedback, not recognition percentage. Wait indicators must correspond to pending work; stop them on success, failure, pause, retirement, and timeout. Never add an artificial delay before showing a completed result.
- Show `Scan in X s` only for an armed, eligible capture countdown; motion resets it. Show processing/review states while busy, rather than a misleading next-scan deadline. Render snapshot boxes on the captured image, never on a moving live preview.
- Keep the card faces visible. Nine-card mode must not render nine large name/price labels; show details on selection or in review. Debug telemetry is opt-in.
- Show actual counts such as `6 ready · 2 checking · 1 needs attention`. Never invent a nine-card detection count from the expected layout or claim page completion while unresolved work remains.
- Provide targeted retry/skip and an explicit next-page action; invalidate page-generation work on page changes. Partial results remain reviewable and must not be presented as a complete page.
- Support reduced motion, non-color status cues, readable contrast, and touch targets on small phones. Geometry and recognition logic stay outside views.

## Retrieval And Data Rules

- Use the normalized layout and manifests in `docs/DATASET_LAYOUT.md`; `datasets/curated/` is the accepted dataset, while acquisition and unresolved candidates remain staging data. Recognition query folders contain complete single-card crops only. Multiple-card scenes belong in detection data, where every relevant instance must be annotated before promotion. Preserve source frames/catalog originals and hashes when deleting rejected derived crops.
- Root SOL must visually compare retained real-query crops with the actual reference artwork before assigning a catalog ID. Luna/mechanical OCR and local-feature results are proposals. A high inlier count or matching character name is insufficient: shared text/layout and alternate artwork can yield convincing false matches. Missing source pixels must not be filled to turn partial cards into complete queries.
- Source video acceptance requires actual available format metadata of at least **1440p**, with **2160p/4K preferred**, plus a per-card pixel/detail check. A 4K title, upscaled output or tiny crop in a 4K frame does not establish readable identity evidence. Unknown-resolution sources stay candidates; the inspected 1080p One Piece excerpt is an acquisition-failure example, not eligible training footage under this requirement.
- Automatic labeling must require no user transcription or screening. OCR/catalog/visual/temporal evidence produce auditable proposals; ambiguous or conflicting cards are automatically quarantined or kept unlabeled. Do not guess a label just to increase coverage, nor equate a pseudo-label with independent ground truth.
- Treat encoder choice, training objective, and search policy as separate decisions. ArcFace trains angular embeddings; deployment still uses normalized similarity search. Retain historical baseline results and current-model comparisons and measure held-out identity/game performance before selecting a loss or larger model.
- The retrieval trainer is `training/card_recognizer/train_retrieval.py`; read `docs/RETRIEVAL_TRAINING.md`. `configs/dinov2_arcface_gpu.yaml` (CUDA, LAN GPU machine per `docs/GPU_TRAINING.md`) produced the deployed encoder. The earlier Apple-MPS pilot is historical and its config/artifacts have been retired. Keep sampled-gallery validation scores separate from final full-catalog results. Never include held-out identities' references/aliases in training gradients or read test pixels for checkpoint selection. Training must not silently fall back to CPU operators.
- Direction: do not repeat frozen-encoder comparisons. Improve recognition by fine-tuning pretrained backbones with a retrieval objective, not by training from scratch. DINOv2-small/ArcFace is the deployed encoder, not a final choice; DINOv3/ConvNeXt/SigLIP-family candidates require their own availability, input, fine-tuning and deployment evidence. Rebuild gallery embeddings and the bundle after any encoder change; acquired references remain useful regardless of backbone.
- Separate catalog identity, visual family, reference image, derived descriptor, independent labeled query, and physical card instance. Preserve unresolved printing ambiguity; do not force identical visible designs into distinguishable image labels.
- Curate a bounded number of references per identity. Augmentations primarily belong in training; extra gallery templates need evidence and calibration that accounts for reference-count bias. Reserve expensive local/detail/OCR verification for ambiguous shortlists under the nine-card work budget.
- Normalize gallery and query embeddings; reject invalid/zero vectors. Cosine scores are similarities, not probabilities.
- Use a namespaced identity containing game, set, number, language, and printing/art variant when known. Preserve `_p1` and other existing variant IDs.
- Multiple reference images may map to one identity. Compute ambiguity against the next distinct identity, not another template of the same card.
- Include explicit unknown/retry outcomes, calibrated absolute and margin thresholds, and verification/review. Use temporal confirmation only when independent captures and their association are available; it is not required for the default fixed-image cycle.
- Start with exact matrix search for the present catalog. Add approximate search only after measuring its need and recall cost.
- Record encoder hash, preprocessing version, vector dimension/type, normalization, catalog hash, and identity mapping in the index manifest. Rebuild all embeddings after changing the encoder or preprocessing; reject mismatched artifacts.
- Separate training, calibration, and locked test captures by physical card/capture session/source. Check exact and near duplicates. Never put augmentations or neighboring video frames into different splits.
- For new-card generalization tests, hold identities out of encoder training but enroll their reference images at inference. Keep query photos independent of gallery images.
- Include not-enrolled cards, backs, non-cards, glare, sleeves, rotation, overlap, and visually similar variants. Do not force every crop to a known identity.
- Report counts, precision/coverage, unknown false accepts, retrieval recall, and end-to-end latency. Accuracy without dataset provenance and denominators is insufficient.

## Browser Rules

- The scanner lives in `web/`. `scripts/export_browser_models.py` writes ONNX models plus golden fixtures; `scripts/build_browser_bundle.py` assembles the content-versioned bundle (defaults select the current GPU export and fine-tuned corner detector; pass the int8 encoder and price inputs explicitly). TypeScript preprocessing must stay bit-exact with `canonical.py` on those fixtures; rerun `npm test` after any contract change. After a new bundle: `dvc add web/public/bundle && dvc push`, commit the pointer, then `deploy/deploy.sh`.
- The GPU path runs the fp32 encoder; the WASM/CPU path runs the int8 encoder when the bundle has one. Check int8 top-1 parity against fp32 before shipping it.
- Use HTTPS for physical-phone camera access; a laptop LAN HTTP address is not the phone's localhost.
- Keep inference off the main UI thread. Feature-detect capabilities, attempt the pinned runtime/model, and fall back from WebGPU to WASM on initialization or inference failure.
- Treat WASM threading as optional; it requires browser support and cross-origin isolation. Host matching runtime JS/WASM assets together.
- Stop camera tracks and invalidate pending work when hidden/stopped. Close transferred frame resources explicitly.
- Version and atomically update model/index/catalog caches. A shared URL must not serve incompatible cached artifacts.
- Device benchmarks must name phone, OS, browser, runtime version, execution provider, input size, catalog size, and warm-up/sustained duration.

## Verification

Choose checks appropriate to the change. Documentation changes need link/path and diff checks; they do not require training. Useful existing commands, after installing dependencies and restoring required artifacts (`dvc pull`):

```bash
.env/bin/python -m unittest discover -s tests -v
cd web && npm run typecheck && npm test && npm run e2e
.env/bin/python scripts/generate_synthetic_cards.py --output runs/card_detector/new-synthetic-run --scenes 40 --video-seconds 12
.env/bin/python training/card_detector/train_obb.py --data runs/card_detector/new-synthetic-run/dataset.yaml --from-scratch --epochs 1 --imgsz 320 --batch 4 --device cpu --name new-smoke-run
DEPLOY_HOST=user@VPS deploy/deploy.sh --dry-run
```

Web parity tests are skipped, not passed, when their fixtures are missing. Physical-phone testing remains mandatory for camera, orientation, latency, memory, thermal behavior and browser compatibility claims. Record unavailable tools and artifacts honestly.
