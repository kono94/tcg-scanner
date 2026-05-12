# TCG Scanner Agent Guide

This repository is a prototype for live One Piece TCG card scanning. It has two active areas:

- Python model/data tooling at the repository root.
- A SwiftUI iOS app in `tcg-scanner-app/`.

Read `CONTEXT.md` before making architectural changes. For iOS work, also read `tcg-scanner-app/CONTEXT.md`.

## Current Product Goal

Build a simple mobile scanner that uses the phone camera to detect cards, crop or track them, recognize the exact card ID, and show live card metadata and prices.

The practical near-term goal is not a polished app. It is a reliable vertical slice:

1. Camera preview works on device.
2. Detector boxes align with the visible preview.
3. A detected card crop can be passed to recognition.
4. The recognized card ID resolves to local metadata.
5. Price lookup is isolated behind a replaceable service.

## Repo Map

- `cards/`: local card image and metadata corpus, grouped by card prefix/set.
- `htmls/`: saved One Piece card list HTML pages used by the fetcher notebook.
- `card-fetcher.ipynb`: scraper/parser notebook that created the local card corpus.
- `cardtrainer.ipynb`: exploratory notebook for detector and recognizer experiments.
- `model.py`: PyTorch MobileNetV3 recognizer model and preprocessing.
- `train.py`: recognizer training script.
- `inferencer.py`: recognizer inference plus optional embedding/template search.
- `export_yolo.py`: exports `card_detector.pt` to CoreML.
- `cards.yaml`: YOLO dataset config for card detection.
- `my.yaml`: ByteTrack tracker config.
- `tcg-scanner-app/`: SwiftUI iOS app.

## Development Rules

- Preserve the Python prototype unless intentionally migrating behavior to Swift.
- Do not rewrite notebooks as part of unrelated app fixes.
- Keep camera capture, model inference, coordinate conversion, and UI overlay code separated.
- Treat coordinate systems explicitly. Document whether a rectangle is normalized Vision coordinates, pixel-buffer coordinates, preview-layer coordinates, or SwiftUI view coordinates.
- Prefer small testable helpers for geometry conversions instead of embedding math in SwiftUI views.
- Do not assume the camera frame aspect ratio matches the preview. The current preview uses `resizeAspectFill`.
- Avoid adding live price APIs directly into camera/inference code. Put them behind a service protocol.

## Useful Verification

For Python:

```bash
python main.py
python train.py
python export_yolo.py
```

For iOS:

```bash
xcodebuild -project tcg-scanner-app/tcg-scanner-app.xcodeproj -scheme tcg-scanner-app -destination 'generic/platform=iOS' build
xcodebuild -project tcg-scanner-app/tcg-scanner-app.xcodeproj -scheme tcg-scanner-app -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.4' test
```

Camera behavior still needs physical-device testing. Simulator-only verification is not enough for this project.
