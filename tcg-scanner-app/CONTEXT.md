# iOS App Context

## Goal

The iOS app should provide a minimal live scanner:

1. Show the back camera preview.
2. Run card detection on camera frames.
3. Draw bounding boxes exactly over visible cards.
4. Crop detected cards for recognition.
5. Resolve recognized card IDs to local metadata and prices.

## Current Files

- `tcg-scanner-app/TCP_ScannerApp.swift`: app entry point. Creates shared settings and scanner state for the Scan, Session, and Settings tabs.
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
- Session cards are finalized when a track is lost or scanning stops. Matching can improve while the track is active; only the best per-track recognition state should be saved.

## Target Architecture

Current intended structure:

- `CameraService`: owns `AVCaptureSession`, authorization, lifecycle, and frame publishing.
- `CardDetector`: wraps `VNCoreMLRequest` and emits normalized detections.
- `DetectionCoordinateMapper`: converts model/Vision boxes to preview coordinates.
- `CardCropper`: extracts corrected card crops from pixel buffers.
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

1. Verify camera preview, boxes, recognition, and session finalization on a physical device.
2. Measure detector and recognizer latency over a longer scan, especially after the first 10-20 seconds.
3. Add still-frame debug capture so a bad frame can be saved and reproduced.
4. Improve price data refresh while keeping it behind `PriceServing`.
5. Keep expanding unit tests around coordinate mapping, recognition scheduling, session persistence, and settings.
