# Current browser bundle

The bundle in `web/public/bundle` (DVC). Update this page when a new bundle ships and move
the superseded text to `archive/docs/` (the previous page is
[BROWSER_BUNDLE_270473320b53d5b4.md](../archive/docs/BROWSER_BUNDLE_270473320b53d5b4.md)).

Bundle `c73c459e7a24eaf2` (2026-10-21) carries half-precision copies of the encoder and
detector. Gallery, catalog, prices and card-back rows are byte-identical to `270473320b53d5b4`;
the detector weights (`corners-s640-v4-ft`, confidence 0.6, NMS IoU 0.7) are unchanged.

| Path | Encoder | Detector | Model download |
| --- | --- | --- | --- |
| WebGPU | `encoder.fp16.onnx` 43.9 MB (40.4 gzip) | `detector.fp16.onnx` 23.0 MB (21.2 gzip) | 67 MB |
| WASM (all platforms) | `encoder.int8.onnx` 23.5 MB (20.7 gzip) | `detector.fp16.onnx` | 47 MB |

The fp32 `encoder.onnx`/`detector.onnx` stay in the bundle as the reference the variants were
converted from; the app does not download them when the manifest lists fp16 copies.

## Precision checks

`scripts/release/compare_model_precisions.py` writes the fp16 models and
`models/browser-export/gpu-20261017/precision_report.json`. Conversion:
`onnxruntime.transformers.float16` with float32 inputs/outputs; the encoder's final L2
normalization stays float32 (in fp16 its epsilon underflows and the embedding becomes NaN).

- Encoder, 298 AI-reviewed real query crops (188 train / 24 val / 86 test) against the
  bundle gallery: fp32, fp16 and int8 each rank the exact ID first for 292 and within the
  top five for 298. fp16 keeps the fp32 top-1 on 298/298 (min cosine to fp32 0.99999, top-1
  score change ≤ 0.0006). int8 keeps it on 296/298 (mean cosine 0.991, top-1 score change
  up to 0.056, margin change up to 0.067).
- Detector, 28 held-out real-corner frames (156 cards): fp32 and fp16 both find 139 within
  5% corner error (mean 2.38% vs 2.39% of the card diagonal). On 75 unlabeled Yu-Gi-Oh!
  video frames both return 210 detections, 209 within 1%.
- onnxruntime-web 1.30 WASM (Node): fp16 encoder cosine 0.999997 to fp32 on a parity
  fixture; fp16 detector raw outputs within 1.3 px (640 px input) of fp32.
- WASM memory (Node, single thread, one encoder): fp32 639 MB RSS, fp16 569 MB, int8
  397 MB, all about 650 ms per card. fp16 weights do not reduce WASM memory much, so the
  CPU path uses int8.

These are retrieval and corner diagnostics on repeated video views with AI-reviewed labels,
not scanner accuracy. In-browser WebGPU execution of the fp16 models, iPhone memory and phone
latency are unverified. `npm run typecheck`, 122 web unit/parity tests, 80 Python tests and
the 9 Playwright flows (desktop Chrome, test-video camera mock) pass.

The detector is re-exported from its `.pt` weights at every build, which yields new ONNX
bytes (identical outputs), so each rebuild changes the bundle version. The builder converts
that export to fp16 and requires its outputs to match the evaluated `detector.fp16.onnx` on
the parity scenes.
