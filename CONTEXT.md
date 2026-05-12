# TCG Scanner Context

## What This Project Is

TCG Scanner is a card-recognition prototype for One Piece trading cards. The intended product is a mobile app that lets a user point a phone at cards, detect the card in the camera feed, recognize the exact card, and eventually show current price information.

The repository currently contains:

- A Python/PyTorch workflow for card recognition.
- A YOLO/CoreML detector workflow for finding cards in an image.
- A local card corpus scraped from official One Piece card list pages.
- A SwiftUI iOS app that runs live detection, tracking, recognition, local metadata lookup, bundled price lookup, and session tracking.

## Current State

The project is mid-migration from Python prototypes to iOS:

- Card recognition exists in Python with a MobileNetV3-based classifier and embeddings.
- Card detection exists as a YOLO model exported to CoreML.
- The iOS app has one active scanner path built around `CameraService`, `ScannerViewModel`, `CardDetector`, `CardTracker`, `CardCropper`, `CardRecognizer`, local metadata, and bundled snapshot prices.
- The app has Scan, Session, and Settings tabs. Session cards persist on device until reset, and Settings persist price currency plus whether duplicate card IDs are allowed in the session.
- The old FastAPI upload app has been removed. The Python side is now model/data tooling only.

## Data And Models

### Card Corpus

`cards/` contains card images and JSON metadata, grouped by card prefix/set such as `OP01`, `OP02`, `ST01`, and `EB01`.

`htmls/` contains saved source HTML pages used to build that corpus.

### Detection

Detection identifies cards in a frame. Relevant files:

- `training/card_detector/export_yolo.py`
- `training/card_detector/cards.yaml`
- `tcg-scanner-app/tcg-scanner-app/Models/card_detector.mlpackage/`

`cards.yaml` defines a YOLO dataset with two classes:

- `op`
- `pokemon`

The current iOS app loads `card_detector` through the bundled CoreML package.

### Recognition

Recognition identifies the exact card. Relevant files:

- `model.py`
- `training/card_recognizer/train.py`
- `inferencer.py`
- `datasets/card_recognizer/names.txt`
- `tcg-scanner-app/tcg-scanner-app/Models/card_recognizer.mlpackage/`
- `tcg-scanner-app/tcg-scanner-app/Resources/recognizer_labels.json`
- `mobile_large_v1_state_dict.pth` if available locally

The recognizer uses MobileNetV3 Large ImageNet weights, adds a 256-dimensional embedding layer, and classifies against the labels in `names.txt`.

## Python Pipeline

`model.py` defines `CardModel`, a MobileNetV3 Large backbone with embedding and classification heads.

`training/card_recognizer/train.py`:

- Builds the dataset from `datasets/card_recognizer/cards`.
- Writes labels to `datasets/card_recognizer/names.txt`.
- Trains the classifier briefly with augmentation.
- Saves a state dict and full PyTorch model.

`inferencer.py`:

- Loads `names.txt`.
- Loads `mobile_large_v1_state_dict.pth`.
- Runs classification via `classify_image`.
- Also contains an optional template embedding search path.

`scripts/scrape_prices.py` refreshes the bundled app price snapshot at
`tcg-scanner-app/tcg-scanner-app/Resources/card_index.json`.

## iOS Pipeline

The SwiftUI app is in `tcg-scanner-app/tcg-scanner-app/`.

Important files:

- `TCP_ScannerApp.swift`: current app entry point.
- `Views/CameraView.swift`: live preview, frame subscription, overlay rendering.
- `Views/HomeView.swift`: Session tab UI and session rows.
- `Views/SettingsView.swift`: persisted app settings.
- `Utilities/CameraService.swift`: AVCaptureSession publisher for pixel buffers.
- `Utilities/CardDetector.swift`: Vision/CoreML detector wrapper.
- `Utilities/CardRecognizer.swift`: CoreML exact-card recognizer wrapper.
- `Utilities/PriceService.swift`: bundled snapshot price lookup.
- `Utilities/ScannerViewModel.swift`: scanner orchestration, tracking, recognition scheduling, session persistence.

Current app entry point creates shared `ScannerSettings` and `ScannerViewModel` instances, then passes them into the Scan, Session, and Settings tabs.

## Main Technical Problem

The hard part is not only running a model. The hard part is preserving a correct mapping across these coordinate spaces:

1. Raw camera pixel buffer.
2. Vision model input after orientation and crop/scale.
3. Vision normalized bounding boxes.
4. AVCaptureVideoPreviewLayer display geometry.
5. SwiftUI overlay coordinates.
6. Cropped card image passed to the recognizer.

The current app uses `DetectionCoordinateMapper` helpers and tests for Vision-to-metadata, metadata-to-pixel-buffer, and aspect-fill preview conversions. This should still be treated as a high-risk area and verified on a physical device, because camera orientation and preview-layer behavior are device-sensitive.
