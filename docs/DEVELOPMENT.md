# Current development priorities

The browser scanner is implemented: stable-frame capture → ordered card corners →
perspective warp → fine-tuned DINOv2/ArcFace retrieval → match/value review.
Single-card and variable-card-count binder pages, single-page and persisted batch
workflows, card-back handling and local collection are supported. Evidence and
limitations are in [the browser README](../web/README.md) and
[bundle notes](BROWSER_BUNDLE_2026-10-17.md).

Next work:

The [2026-10-17 phone-feedback work log](PHONE_FEEDBACK_2026-10-17.md) records
the iPhone 13 mini report, capture inventory, implemented session/camera changes,
validation and remaining work. The phone report is not a controlled benchmark.

1. Verify physical iPhone/Safari and Android/Chrome: HTTPS camera access,
   orientation, WebGPU/WASM fallback, first/settled latency, memory, thermal
   behavior, sustained batches and background/resume. Record named devices and
   runtime/provider versions; desktop video mocks do not establish phone results.
2. Expand automated independent real-capture validation, preserving original
   pixels, source/session splits and auditable AI-reviewed corner/identity labels.
   Include single cards, binder pages, >9 cards, negatives, backs and difficult
   variants. No user transcription or manual dataset creation is required.
3. Calibrate identity/back/unknown and variant-review thresholds on permitted
   calibration data. Keep locked test captures out of selection; report precision,
   coverage, false accepts and per-scene latency with provenance and denominators.
4. Improve localization/retrieval only against declared baselines. Maintain the
   shared canonicalization contract and regenerate gallery/export fixtures after
   encoder changes. Similarity is the normal path; OCR is optional and bounded.

The former roadmap is [archived](../archive/ROADMAP_2026-10-17.md). Preserve source
images and current training/evaluation artifacts during further cleanup.
