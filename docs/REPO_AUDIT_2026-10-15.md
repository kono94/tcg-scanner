# Repository audit — 2026-10-15

## Assessment

This is an implemented SwiftUI scanner plus Python classifier tooling, not yet a general TCG retrieval system or a web application. The existing component separation is useful. The biggest immediate problem is that there is no independent camera evaluation set, followed by frame backpressure and recognition lifecycle correctness. Changing the model without repairing those would make progress hard to measure.

Scope: inspected source, current corpus hashes, detector annotation ZIP, experiment manifests/reports/logs, app resources, Git state, local tool availability, and current primary browser/runtime documentation. No training, model export, phone benchmark, or iOS test run was performed. This audit changes documentation only.

## Implemented pipeline

```text
AVCapture 1080p BGRA frames
  → serial scanner queue
  → CoreML/Vision axis-aligned detector, nominally every 200 ms
  → Vision object tracking + IoU detector association
  → rectangular crop of the current pixel buffer
  → separate serial CoreML classifier queue
  → highest-score-per-track result + metadata + bundled price snapshot
  → overlays; session entry on track loss / stop
```

Evidence: `CameraService.swift`, `ScannerViewModel.swift`, `CardDetector.swift`, `CardTracker.swift`, `CardCropper.swift`, `CardRecognizer.swift`, `RecognitionScheduler.swift` under `tcg-scanner-app/tcg-scanner-app/Utilities/`.

Python uses MobileNetV3 Large, a 256-dimensional linear feature layer, and a 1,399-class head. Training optimizes cross entropy only. The export wrapper explicitly emits logits only. No deployed cosine index, enrollment workflow, ONNX export, browser frontend, or browser runtime integration was found. Python `onnxruntime` in requirements is not a browser integration.

The detector configuration has `op` and `pokemon` classes. `CardGame.supportsExactRecognition` allows only One Piece, so detecting Pokémon currently means tracking an unsupported game. Metadata and labels both have 1,399 entries. Prices are a snapshot dated 2026-05-10, not a live feed.

## Data and accuracy evidence

| Item | Observed state | Consequence |
| --- | --- | --- |
| Reference corpus | 1,399 images / 1,399 identities across 22 One Piece set groups | One reference per identity; no real-photo training samples currently present |
| `manual_eval` | 10 images, EB01-001 through EB01-010, each named `official.jpg` | Narrow fixture coverage |
| Exact duplicate check | SHA-256 of all 10 eval files matches the corresponding training image | Current evaluation is contaminated; not camera validation |
| Dataset splitting | Same manual folder serves validation and test; no duplicate check in manifest builder | Checkpoint selection, calibration, and final testing are not independent |
| Unknown evaluation | Manifest resolver requires eval identities to exist in training labels | No supported open-set benchmark in the current classifier pipeline |
| Detector labels | ZIP contains 65 train + 10 val text files, 191 + 30 rows; every row has 5 fields | Ordinary axis-aligned YOLO boxes; no corner/OBB supervision |
| Detector images | Tracked `datasets/card_detector/images.zip` is deleted locally | Detector retraining is not reproducible from this checkout alone |

Duplicate findings concern the current file bytes. Both experiment manifests reference these same eval paths, but they do not record historical image hashes; historical bytes cannot be independently reconstructed from those manifests.

Local historical results, included for provenance rather than as evidence of live accuracy:

| Experiment | Train top-1 report | Validation evidence |
| --- | --- | --- |
| `1778846379` | 97.28% on 1,399 references | Report: 40% top-1, 60% top-5 on 10 eval entries |
| `1778862585` (app manifest version) | 99.79% on 1,399 references | CSV training logs reach 100% on 10 eval entries; no separate validation classification-report JSON found |

The app manifest thresholds are confidence 0.4 and margin 0.01 with source `export-override`. These are not demonstrated open-set operating points. Softmax confidence and cosine similarity should not be presented as calibrated match probabilities.

## Prioritized source findings

### P0 — frame admission does not bound the queue

`ScannerViewModel.handle(frame:)` appends a closure to `scannerQueue` for every captured frame. `isProcessingFrame` is checked and reset inside that same serial queue. Each closure normally sees the previous closure already completed, so the flag does not discard queued frames. `alwaysDiscardsLateVideoFrames` applies to the camera callback, which returns after dispatching; it does not bound the downstream queue.

If inference/tracking takes longer than the camera interval, retained buffers and frame age can grow. Camera timestamps are available but not used for freshness or scheduling. This is a source-level failure mode; its size on a phone has not been measured.

### P0 — recognition completion can revive a retired track

The recognition callback uses `recognitionStates[track.id] ?? TrackRecognitionState()`. A completion after loss or stop can recreate state that was deliberately removed. There is no session generation, job revision, freshness check, or cancellation token. A late result can later be finalized as a session entry. Reset clears saved session cards but does not invalidate active recognition state/jobs.

The recognizer queue is serial and each track has an in-flight flag, but there is no global queue budget, crop expiry, or pending-crop replacement. Model service latency alone does not reveal capture-to-result delay.

### P1 — recognition favors a single historical score

`TrackRecognitionState.apply` keeps only a higher confidence than any previous result. A wrong early high score can prevent correction by repeated lower-scored correct evidence. Session finalization accepts any stored result, without multi-frame confirmation. Scheduling tests explicitly preserve this highest-score behavior; they are not tests of temporal correctness.

### P1 — tracking can merge identities or persist after detector absence

`CardTracker.mergeDetections` does not reserve a track after matching it, so multiple detections can update the same track in one detector pass. A successful Vision tracker resets missed-frame counts even when fresh detector passes no longer support the card. Requests returning no observation also do not increment missed counts. There is no detector-confirmed expiry in elapsed time. These are risks for ID switches and session finalization.

### P1 — crops lack rectification and preprocessing parity

`CardCropper` extracts an axis-aligned rectangle; no corner estimate, perspective warp, quality selection, or orientation canonicalization is applied there. The geometry relies on capture being physically rotated to portrait; non-up fallback orientation needs explicit end-to-end testing because the cropper does not take orientation.

Python evaluation resizes the short edge to 256 and center-crops 224 square, which can remove the card's number/text/borders. The Swift path constructs a fixed-size CoreML image input from a rectangular crop without explicitly implementing that same sequence. Exact resize behavior/parity has not been established.

The exporter uses one average standard deviation for CoreML input scale, versus per-channel standard deviations in Python. That is a known numeric approximation. Its smoke test checks output shape on a flat image and catches prediction failure without failing export. It is not a numerical equivalence test.

## Reproducibility and checkout health

- Base commit inspected: `ed68cff` (`Use 1080p capture and configurable recognition thresholds`).
- Existing unrelated work: deleted detector image ZIP, modified `datasets/card_recognizer/README.md`, untracked `scripts/import_real_card_images.py`. Left untouched.
- The local dataset README describes `scripts/crawl_google_lens_sales_images.py`, which is absent. Root guidance also referenced moved notebooks and nonexistent `main.py`; this documentation pass corrects the root guide.
- Ordinary `git status` failed because its LFS filter attempted a write in restricted `.git/lfs/tmp`. A read-only status with LFS filters disabled falsely lists hydrated recognizer weights as changed. Independent SHA-256 checks confirm both detector and recognizer weight files match their HEAD LFS object IDs; do not treat that listing as a real model modification.
- Detector weights are 3,069,888 bytes; recognizer weights are 9,764,526 bytes. These are weight sizes, not total browser download or runtime-memory estimates.
- Both `experiment.json` best-checkpoint paths are missing locally. One older checkpoint exists at another path; it must not silently substitute for the declared best model.
- Legacy `inferencer.py` expects absent root `mobile_large_v1_state_dict.pth` and `datasets/card_recognizer/names.txt`.
- `requirements.txt` includes old Linux CUDA-specific packages and does not declare `coremltools`, although the exporter imports it. It is not a reproducible cross-platform environment specification.
- The selected developer directory is Command Line Tools, and `xcodebuild -version` fails. The current `python3` lacks torch, torchvision, Lightning, Pillow, coremltools, and ONNX Runtime. No build/training success is claimed.
- Existing XCTest source covers geometry examples, retry policy, scoring, settings, and formatting. It does not exercise delayed scanner callbacks, queue overload, identity association, or actual model parity. No Python test suite was found in the inspected source tree.

## Recommended architecture

```text
camera + frame identity + monotonic timestamp + source transforms
  → bounded latest frame
  → generic card localization / tracking
  → four corners + quality gate
  → perspective warp to a full-card canonical crop
  → encoder → L2-normalized vector
  → gallery cosine top-k → optional local/OCR verification
  → unknown / candidate / temporally confirmed identity
  → metadata + price service + session UI
```

Retrieval removes the need for a classifier output for each enrolled identity. It does not remove the need for a discriminative encoder, suitable references, domain coverage, rejection, or variant resolution. Existing classifier features and a pretrained backbone are useful baselines, not proven universal embeddings. Compare them before choosing metric-learning fine tuning or a larger model.

OBB can improve rotated crops, but it represents a rotated rectangle. A tilted card projects to an arbitrary quadrilateral. Prefer four corner keypoints, a segmentation contour, or a detector plus corner-refinement stage; compare their actual downstream accuracy/latency. Existing box annotations cannot recover true corners automatically.

Browser delivery is a good product hypothesis because a URL avoids native installation. Ship a small feasibility experiment before committing to a full port. Native CoreML artifacts cannot simply be loaded by ONNX Runtime Web; restore source checkpoints, export ONNX, implement preprocessing/postprocessing, and verify equivalence first.

## Primary references checked on 2026-10-15

- [ONNX Runtime Web overview](https://onnxruntime.ai/docs/tutorials/web/): in-browser inference and execution-provider/operator coverage.
- [ONNX Runtime Web quickstart](https://onnxruntime.ai/docs/get-started/with-javascript/web.html): browser support table still lists Safari WebGPU as unsupported at audit time.
- [WebKit Safari 26.0 release notes](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/#webgpu): WebGPU shipped on macOS/iOS/iPadOS/visionOS; WebKit explicitly lists ONNX Runtime among supported frameworks. This conflicts with the older runtime table. Test the pinned runtime and actual model/device combination; do not infer a blanket Safari prohibition or blanket compatibility.
- [ONNX Runtime flags and session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html): WASM threading requires browser support plus cross-origin isolation; runtime JS/WASM builds must match; the proxy-worker option is unsuitable for WebGPU. Use an explicit application worker for the proposed shared design.
- [MDN getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia): camera permission and secure-context requirements.
- [Ultralytics OBB task](https://docs.ultralytics.com/tasks/obb/): rotated-rectangle outputs, despite four-point label storage.
- [OpenCV geometric transforms](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html): four correspondences define a perspective transform.
- [DELG paper](https://arxiv.org/abs/2001.05027): global retrieval with local-feature verification is an established image-search design; its landmark results do not establish TCG accuracy.

The recommendations above are engineering proposals derived from this inspection. See [the roadmap](../ROADMAP.md) for sequencing and acceptance criteria.
