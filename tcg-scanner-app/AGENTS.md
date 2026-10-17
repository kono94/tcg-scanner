# iOS App Agent Guide

Follow the repository-level `../AGENTS.md` as well as this guide for work under `tcg-scanner-app/`. Source-file paths below are relative to `tcg-scanner-app/` inside this directory unless otherwise stated.

## Audit update — 2026-10-15

The native app remains the implemented baseline. The proposed phone-browser and retrieval direction is described in `../ROADMAP.md`; neither is implemented yet. See `../docs/REPO_AUDIT_2026-10-15.md` for source evidence.

Prioritize bounded frame admission and recognition lifecycle correctness. The current busy flag is checked only after enqueueing on the serial scanner queue. Recognition callbacks can recreate lost-track state, and stop/reset do not invalidate a session generation. Highest-ever confidence is not temporal confirmation; the existing scheduler tests preserve that behavior and need new delayed-callback/overload coverage when it changes.

`CardCropper` currently extracts an axis-aligned rectangle. There is no perspective correction, corner model, or similarity index. Exact preprocessing parity across Python and CoreML has not been verified. All ten current manual evaluation images duplicate training reference bytes; do not use their metrics as evidence of device accuracy.

## Goal

The iOS app should provide a minimal live scanner:

1. Show the back camera preview.
2. Run card detection on camera frames.
3. Draw bounding boxes exactly over visible cards.
4. Crop detected cards for recognition.
5. Resolve recognized card IDs to local metadata and prices.

Single-card scanning and nine-card binder pages are first-implementation requirements. Follow `../docs/SCANNER_UX.md` for the clean overlay and capture → recognition → confirmation → review cycle. Use compact per-card status badges with details on selection, a white capture scan-line, and truthful page-level progress. Keep debug text out of the default view. These overlay improvements are specified, not yet implemented.

Nine is a benchmark scenario, not a fixed input, required layout, or maximum. Detection/result/annotation lists have variable length; zero-card captures are valid. Follow `../docs/DATA_COLLECTION.md`, derive UI counts from actual instances, and process larger scenes in bounded chunks without silently dropping instances above nine.

Prefer the stable-frame capture cycle in `../docs/CAPTURE_CYCLE.md` for the next scanner path: configurable steady dwell → immutable capture → detect → recognize → review. Continuous `CardTracker` use is not required for this path; keep it as the existing baseline until the new cycle is verified. Show countdowns only while armed/eligible, and draw detected boxes on the captured image.

Do not defer fair multi-card scheduling, physical-copy counting, page-change invalidation, or nine-card replay tests. Snapshot instances need no cross-frame association; any future tracking mode must test it. Catalog ID is not a physical card-instance ID; identical cards in different pockets must remain separate copies.

## Current Files

- `TCP_ScannerApp.swift`: app entry point. Creates shared settings and scanner state for the Scan, Session, and Settings tabs.
- `Views/CameraView.swift`: live camera screen, loading overlay, preview layer, and bounding-box label rendering.
- `Views/HomeView.swift`: Session tab with persisted recognized cards, total price, sorting, reset, and swipe delete.
- `Views/SettingsView.swift`: persisted settings for EUR display and duplicate-card session behavior.
- `Utilities/CameraService.swift`: active camera session and pixel-buffer publisher.
- `Utilities/CardDetector.swift`: Vision/CoreML detector wrapper.
- `Utilities/CardTracker.swift`: track identity across frames.
- `Utilities/CardCropper.swift`: pixel-buffer crop extraction for recognition.
- `Utilities/CardRecognizer.swift`: exact-card CoreML recognizer wrapper.
- `Utilities/CardMetadataStore.swift`: local card metadata lookup from `card_index.json`.
- `Utilities/PriceService.swift`: bundled snapshot price lookup behind `PriceServing`.
- `Utilities/RecognitionScheduler.swift`: per-track recognition retry/refresh policy.
- `Utilities/ScannerViewModel.swift`: coordinates camera, detection, tracking, recognition, prices, overlays, and session persistence.
- `Utilities/DetectionCoordinateMapper.swift`: coordinate conversion helpers with unit tests.

## Known Problems

- Camera and overlay behavior still needs physical-device verification; simulator-only testing cannot validate real camera timing, orientation, and preview alignment.
- Detection is throttled in `ScannerViewModel` with a fixed interval. If runtime FPS drops, inspect model latency, thermal state, and frame backpressure before changing tracker logic.
- Price data is currently a bundled snapshot, not a live market API.
- Session cards are currently finalized when a track is lost or scanning stops. The current highest-ever-score policy can preserve an early wrong match; replace it with recent-evidence confirmation before saving session entries.

## Target Architecture

Current intended structure:

- `CameraService`: owns `AVCaptureSession`, authorization, lifecycle, and frame publishing.
- `CardDetector`: wraps `VNCoreMLRequest` and emits normalized detections.
- `DetectionCoordinateMapper`: converts model/Vision boxes to preview coordinates.
- `CardCropper`: currently extracts rectangular card crops from pixel buffers; perspective correction is planned.
- `CardRecognizer`: identifies the exact card from a crop.
- `CardMetadataStore`: local lookup from card ID to JSON metadata.
- `PriceService`: price lookup behind a protocol. Keep live APIs out of camera and inference code.
- `ScannerViewModel`: coordinates camera, detector, recognizer, tracking, and UI state.
- SwiftUI views: render scanner/session/settings UI only; no model math.

## Coordinate Guidance

Always name the coordinate space in variable names or comments:

- `visionNormalizedRect`
- `metadataOutputRect`
- `previewLayerRect`
- `pixelBufferRect`
- `viewRect`

For the live overlay, keep conversion logic inside `DetectionCoordinateMapper` or a preview-layer bridge. Do not embed coordinate math directly in SwiftUI views.

## Immediate Fix Order

1. Establish independent capture/calibration/test data and reproducible frame/crop debug records.
2. Implement/test the stable-dwell/countdown/cooldown policy and one-active-capture batch; reject callbacks for obsolete generations, pages, captures, or revisions in one-card/nine-card cases.
3. Finalize snapshot instances through calibrated verification/review and idempotent page saves. Fix association/expiry separately if the continuous tracking baseline remains in use.
4. Verify camera geometry, preprocessing parity, and sustained capture-to-result latency on a physical device.
5. Compare perspective-corrected retrieval against the classifier using the shared benchmark; coordinate with the browser feasibility work in the root roadmap.
6. Implement the instruction/progress overlay alongside the scheduler, then verify it on one-card and nine-card captures. Refresh prices afterward, keeping prices behind `PriceServing`.
