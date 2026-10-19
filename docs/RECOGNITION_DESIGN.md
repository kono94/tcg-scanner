# Card recognition design

How the scanner turns a captured page into card identities, which contracts keep Python
and the browser consistent, and what is still open. Status 2026-10-19: implemented and
deployed in bundle `270473320b53d5b4` ([BROWSER_BUNDLE.md](BROWSER_BUNDLE.md)); physical
phone behavior and operating thresholds are unverified/uncalibrated.

## Principle: enrollment-based retrieval

The scanner recognizes a card by **similarity search against enrolled reference images**,
not by a classifier with one output per card. Adding a card, set or language means adding
references and metadata and embedding them; the encoder only needs retraining when
measured accuracy requires it (then the whole gallery is re-embedded). Accuracy on a new
game is an empirical question that needs its own held-out evaluation.

Three decisions stay separate: the encoder (pretrained backbone, fine-tuned), the training
objective (ArcFace angular margin), and the search policy (normalized embeddings, cosine
similarity, thresholds). Cosine scores are similarities, not probabilities.

## Pipeline as implemented

```text
stable-frame capture (one immutable image; docs/CAPTURE_CYCLE.md)
  → YOLOv8s-pose: variable-length list of cards, four ordered corners each
      web/src/vision/pose.ts, quad.ts
  → shape filter (convex, side ratio ≤ 2, box fill) → perspective warp per card
  → tcg-full-card-v1 canonical crop → encoder tensor           web/src/vision/preprocess.ts
  → DINOv2-small/ArcFace embedding (WebGPU fp32 or WASM int8)   web/src/inference/worker.ts
  → exact cosine search over the gallery, best template per identity
      web/src/recognition/gallery.ts
  → match policy: back / match / needs attention                web/src/recognition/policy.ts
  → value summary and review UI                                 web/src/recognition/value.ts
```

Every card is an independent instance with its own state; the count is whatever the
detector finds (zero, one, nine, more), processed in bounded chunks. Inference runs in a
worker, off the UI thread.

## Localization

The detector predicts a box plus four geometric corners (TL, TR, BR, BL in portrait
reading order) and each card is perspective-warped from them. Corner order is geometry,
not artwork orientation: upside-down cards are valid, sideways cards are excluded from
training because their corner names are ambiguous. Training combines synthetic scenes
with renderer-exact corners and AI-reviewed real corners
([training/detection/README.md](../training/detection/README.md)).

History: the project started with an oriented-box (OBB) detector and rotation-aligned
crops; a synthetic geometry comparison selected corner keypoints, and OBB decoding was
removed. Axis-aligned boxes are never expanded into claimed corner labels.

Detector outputs map back through letterbox, scale and orientation to source pixels before
cropping, so crops use full capture resolution. Tiny cards are not upscaled into
"recovered" detail; the fix is a closer or higher-resolution capture.

## Canonical preprocessing contract

Queries and gallery references share one versioned contract, `tcg-full-card-v1`
(`training/recognition/canonical.py`, mirrored bit-exactly in TypeScript and checked by
the web parity tests):

1. RGB uint8 input with ordered physical corners (camera) or verified full-image bounds
   (digital reference: never pushed through the detector).
2. Perspective warp to a 224×312 full card (OpenCV `INTER_LINEAR`); borders, name and
   card number are kept.
3. Aspect-preserving fit into 224×224 (`INTER_AREA` when shrinking), centered with
   RGB-mean padding; no center crop.
4. CHW float32, ImageNet mean/std normalization.

Any change to the contract or the encoder requires re-embedding the gallery, re-exporting
golden fixtures and rerunning parity (`npm test`). The gallery index records encoder hash,
contract version, vector dimension and reference order; mismatched artifacts are rejected.

## Encoder

DINOv2-small fine-tuned with ArcFace, 384-D L2-normalized descriptors; recipe, isolation
and results in [RETRIEVAL_TRAINING.md](RETRIEVAL_TRAINING.md). Not training from scratch
and not repeating frozen-encoder sweeps: improvements come from fine-tuning stronger
pretrained backbones under the same isolation and evaluation.

## Search and match policy

- Exact matrix search over all gallery rows (24,038 card rows plus card backs in the
  bundle); approximate search only if measured need and recall cost justify it.
- Scores are per identity (best template), and ambiguity is measured against the best hit
  from a **different artwork family**, not another language/printing of the same art.
- A match needs similarity ≥ 0.45 and margin ≥ 0.05; card backs (`back:pokemon`,
  `back:onepiece`) need ≥ 0.65 and margin ≥ 0.1. All four values are **uncalibrated**
  defaults (`calibrated: false` in `policy.ts`). Below them a card needs attention.
- Retrieval identifies an artwork family. Language and printing stay unresolved unless
  observed; the preferred language is a display/pricing fallback shown as such, and
  plausible versions keep a value range ([MATCH_AND_VALUE_POLICY.md](MATCH_AND_VALUE_POLICY.md)).
- Repeated inference on the same pixels is not independent confirmation; a retry is a new
  capture.

OCR is not part of the browser path. `training/recognition/ocr_identity.py` (Tesseract)
supports offline label proposals; an in-app OCR fast path remains optional
([OCR_IDENTITY.md](OCR_IDENTITY.md)).

## Gallery and identity records

| Record | Responsibility |
| --- | --- |
| Catalog identity | `game:language:card_id`, with art/printing suffixes kept (`OP01-001_p1`); unknown fields stay unknown |
| Artwork family | identities sharing one artwork across languages/printings; used for margin and value range |
| Reference image | original path/hash, provider, role, watermark notes (`datasets/curated` sidecars) |
| Descriptor | reference ID, encoder and contract hashes, dimension/dtype |
| Query crop | source group, source frame, corners, reviewed identity or unknown; never a gallery row |
| Physical card instance | one detected card on one captured page; two pockets with the same card are two copies |

The gallery holds curated references (one or few per identity), not augmentations;
max-over-templates scoring favors identities with many references, so extra templates
need evidence and calibration. A reference used as its own query measures self-match,
not recognition.

## Open questions and next experiments

1. **Calibration** (highest priority): match/back/unknown thresholds and the variant cue
   on independent real captures (`datasets/sources/phone`), reporting precision,
   coverage, unknown false accepts and per-page latency with denominators.
2. **Unknown and hard cases**: not-enrolled cards, backs, glare, sleeves, overlap and
   same-art variants in evaluation, with an explicit unknown outcome.
3. **Selective verification**: for ambiguous shortlists only, local features, aligned
   detail comparison or OCR, under the nine-card work budget. Not implemented.
4. **Stronger encoders**: DINOv3 (ViT/ConvNeXt) or SigLIP-family backbones, fine-tuned the
   same way, with export size, browser latency and int8 parity measured.
5. **More games**: Yu-Gi-Oh! as an explicit transfer evaluation before Magic or sports.
6. **Detector refinement**: a local corner refiner only if measured crop failures justify it.

## References (checked 2026-10-15)

- [ArcFace](https://arxiv.org/abs/1801.07698), [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
- [DINOv2](https://github.com/facebookresearch/dinov2), [DINOv3](https://github.com/facebookresearch/dinov3)
- [DELG](https://arxiv.org/abs/2001.05027): global retrieval plus local-feature verification
- [Ultralytics pose format](https://docs.ultralytics.com/datasets/pose/), [OpenCV perspective transforms](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)

Generic benchmark results from these papers do not establish card-recognition accuracy.
