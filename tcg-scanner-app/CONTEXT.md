# iOS App Context

## Goal

The iOS app should provide a minimal live scanner:

1. Show the back camera preview.
2. Run card detection on camera frames.
3. Draw bounding boxes exactly over visible cards.
4. Crop detected cards for recognition.
5. Resolve recognized card IDs to local metadata and prices.

## Current Files

- `tcg-scanner-app/TCP_ScannerApp.swift`: app entry point. Currently shows a two-tab `TabView` with `CameraView()` and a placeholder History tab.
- `Views/CameraView.swift`: current live camera screen.
- `Utilities/CameraManager.swift`: current camera session and pixel-buffer publisher.
- `Utilities/YOLO.swift`: Vision/CoreML detector wrapper.
- `Utilities/VideoCapture.swift`: older camera capture implementation.
- `Views/PredictionView.swift`: older SwiftUI overlay implementation.
- `Views/BoundingBoxLayer.swift`: older CALayer overlay helper.
- `Views/GuessView.swift`: stale view that calls an outdated `CameraView(predictions:)` initializer.

## Known Problems

- There are multiple camera implementations. Prefer consolidating around one.
- `GuessView.swift` is out of sync with `CameraView`.
- Overlay boxes are scaled directly from normalized Vision coordinates to SwiftUI size. This likely breaks with `AVCaptureVideoPreviewLayer.videoGravity = .resizeAspectFill`.
- The detector is integrated, but exact-card recognizer integration is not.
- Model orientation and crop/scale assumptions are not documented in code.

## Target Architecture

Suggested structure:

- `CameraService`: owns `AVCaptureSession`, authorization, lifecycle, and frame publishing.
- `CardDetector`: wraps `VNCoreMLRequest` and emits normalized detections.
- `DetectionCoordinateMapper`: converts model/Vision boxes to preview coordinates.
- `CardCropper`: extracts corrected card crops from pixel buffers.
- `CardRecognizer`: identifies the exact card from a crop.
- `CardMetadataStore`: local lookup from card ID to JSON metadata.
- `PriceService`: async live price lookup behind a protocol.
- `ScannerViewModel`: coordinates camera, detector, recognizer, tracking, and UI state.
- `ScannerView`: SwiftUI screen only; no model math.

## Coordinate Guidance

Always name the coordinate space in variable names or comments:

- `visionNormalizedRect`
- `metadataOutputRect`
- `previewLayerRect`
- `pixelBufferRect`
- `viewRect`

For the live overlay, prefer using the preview layer conversion APIs rather than hand-scaling:

- `AVCaptureVideoPreviewLayer.layerRectConverted(fromMetadataOutputRect:)`
- `AVCaptureVideoPreviewLayer.metadataOutputRectConverted(fromLayerRect:)`

If using SwiftUI overlays, pass the preview layer bounds and converted rectangles from a UIKit bridge instead of duplicating aspect-fill math in SwiftUI.

## Immediate Fix Order

1. Make the Xcode target compile.
2. Remove or quarantine stale views that are not used by `TCP_ScannerApp`.
3. Keep only one camera frame path.
4. Move detection throttling into the view model so the app does not run inference on every frame unless intended.
5. Fix overlay conversion while showing debug labels for frame size, confidence, and orientation.
6. Add crop capture and save debug images for recognizer testing.

