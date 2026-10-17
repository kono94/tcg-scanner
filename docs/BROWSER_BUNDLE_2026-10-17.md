# Fine-tuned corner detector bundle

Bundle `270473320b53d5b4` replaces the detector in `535078e65aa8d649` with
`runs/card_detector/obb/corners-s640-v4-ft/weights/best.pt`, exported at 640 px.
The confidence threshold remains 0.6 and NMS IoU remains 0.7. Encoder, int8 encoder,
gallery, catalog and price files are byte-identical to the previous bundle.

The app's `plausibleCard` filter now accepts square and nearly square outlines:
perspective can foreshorten a physical card to those proportions. Convexity, the
maximum side ratio of 2.0, and the minimum box-fill check remain. A regression test
checks that these detections survive the complete decoder.

## Provenance and validation

- Source checkpoint SHA-256:
  `9cf37d1885e7e85ec3b4c3bed7e06f6e2bfe1c62b44760fffdf7803787bd5bf4`.
- ONNX SHA-256:
  `d38f1ac96320f369a7dd10d3fc49850e97fda26c0a93143522e8d0f52a756a85`.
- Exported through `scripts.build_browser_bundle.export_detector` (opset 17,
  static input, float32). Maximum absolute difference against PyTorch CPU was
  0.002228 on the seeded input and 0.001092 across five real frames. These compare
  actual raw outputs with ONNX Runtime CPU, not just tensor shapes.
- Detector size: 45,976,592 bytes raw (43.8 MiB), 38,626,265 bytes gzip. This is the
  detector alone; other models, gallery and runtime assets add download size.
- `npm run typecheck`, all 80 unit/parity tests (no skips), `npm run build`, and
  both default Playwright flows passed. Playwright uses desktop Chrome with a
  test-video camera mock, including single-page review and persisted batch review.
- Export and real-frame parity reports are DVC-versioned alongside the checkpoint
  as `browser_export.json` and `browser_real_parity.json`. Updated Ultralytics
  decoder fixtures are in `runs/browser_export/pose-fixture` (DVC).

The detector has synthetic pretraining plus AI-reviewed real-corner fine tuning.
The preceding model comparison used a small video-derived set with AI-reviewed
labels initialized from an older s model; those labels may favor the s family.
This update does not establish physical-phone latency, accuracy or numerical
browser-runtime parity. Recognition thresholds remain uncalibrated.
