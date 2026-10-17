# Enrollment-based card recognition design

Status: updated 2026-10-16. A pretrained DINOv2-small/ArcFace retrieval trainer and
its first Apple-GPU pilot now exist; see [the training guide](RETRIEVAL_TRAINING.md)
and live run status for completion/results. The curated pool has 298 reviewed real
queries. Calibrated production recognition, deployment parity and the browser
runtime remain unimplemented. Frozen baseline results below are historical.

Implemented bootstrap: `training/card_recognizer/canonical.py` is shared by synthetic query-crop generation and gallery preparation. `scripts/generate_synthetic_cards.py` retains exact renderer quads/IDs alongside OBB labels. `scripts/build_embedding_gallery.py` separately creates compatible frozen descriptors and a cosine index; image preparation alone does not. Begin detection with the simple YOLO OBB baseline, using real-video crop diagnostics to decide measured follow-ups.

Measured frozen-MobileNet baseline: 1,431 enrolled references, 1,321 renderer-true-corner synthetic queries, Recall@1/5/20 = **24.9% / 38.2% / 53.8%**. Queries derive from gallery source images and exclude predicted localization failures. This does not establish real-camera accuracy and is insufficient for deployment. The report is `runs/card_recognizer/synthetic_retrieval_report.json`.

Frozen DINOv2-small improved the exact same full-set Recall@1/5/20 to **37.4% / 48.3% / 61.1%**, with no card-specific training. A separate max-score 0°/180° query sweep performed worse (35.8% @1). The report is `runs/card_recognizer/dinov2_baseline/report.json`; 16 real detector-crop shortlists are also saved as unverified proposals, not accuracy labels. Do not turn a better synthetic baseline into a camera-readiness claim.

Pretrained models can be tested with no card-specific training. Keep their roles separate: GroundingDINO uses image/text prompts to propose localization boxes; DINOv2/DINOv3 image features are candidates for normalized gallery descriptors. Neither provides guaranteed exact-print recognition merely because an item is enrolled. Evaluate a frozen descriptor before deciding whether metric fine tuning is necessary, and measure physical-card versus broadcast-graphic errors for any localization teacher. A strong offline teacher need not be the eventual phone model.

## Objective and scope

Build a reusable visual instance-retrieval engine: enroll card references and metadata, then identify independent camera crops without adding a classifier output for each new identity. One Piece, Pokémon, and other games can share encoder/search infrastructure. Recognition accuracy on a new game is an empirical question, not guaranteed by enrollment.

The existing MobileNetV3 feature output is a useful baseline. It is trained through ordinary classifier cross entropy, exported as logits, and has not been validated for retrieval. Replacing softmax with cosine at inference alone does not establish a stronger encoder.

Separate three decisions: the encoder architecture/pretraining, the training objective, and the gallery/search/verification policy. ArcFace is an angular-margin training objective; cosine is a comparison rule for normalized embeddings. They can be used together. A larger encoder or a different loss alone cannot repair poor crops, missing detail, mislabeled references, or contaminated evaluation.

Latest decision: further frozen-DINO comparisons are not the next task. Fine-tune a pretrained encoder for card retrieval. The executed baseline used DINOv2 **ViT-S/14**, frozen 384-D CLS descriptors; no production backbone has been selected. A practical first adaptation can reuse those existing pretrained weights with a learned 256–512-D projection and an ArcFace training head. The ArcFace class weights are training machinery; deployment retains the normalized encoder/projection and a reference gallery, so enrollment does not add a deployed classifier output. Compare another accessible pretrained backbone only as a controlled fine-tuning/deployment alternative, rather than starting a broad frozen-model sweep. Training from scratch is not the default with the current data.

Keep architecture, initialization and objective explicit: `pretrained backbone → pooling/projection → L2 descriptor → cosine gallery`; during training, the descriptor also feeds the chosen angular-margin or contrastive objective. DINOv3 supplies ViT and ConvNeXt backbones; SigLIP2 is another pretrained encoder family. Their generic benchmark results do not establish TCG accuracy, and access/runtime requirements must be checked before choosing a checkpoint. See the primary references in `FOUNDATION_MODELS.md`.

## Candidate families, language preference and value uncertainty

Follow [the match and value policy](MATCH_AND_VALUE_POLICY.md). Preserve supported
candidate families and unresolved language/printing/finish attributes. Preferred
language can choose a provisional displayed candidate and pricing interpretation;
it is never observed evidence, a ground-truth label or exact-ID confirmation.
Clear language evidence overrides it. Do not filter out compatible alternatives
or let an expensive weak neighbor become a plausible match.

Details/review show the preferred estimate and the supported plausible-versions
range independently, including quote coverage and assumptions. Material value
ambiguity produces a compact `Check variant` cue and a closer single-card review
flow, preserving physical-copy identity and other binder results. Range pricing
and family resolution are planned and do not change the pilot's exact-ID scores.
Measure artwork-family retrieval and exact language/printing resolution separately.

## Proposed inference pipeline

Use the [stable-frame capture policy](CAPTURE_CYCLE.md) as the default trigger: detect and recognize one immutable scene capture after a configurable stillness dwell, with no continuous per-card tracking requirement. Every actual instance uses that capture's source geometry; count is variable, including zero and more than nine. Additional independent captures are explicit retries/replacement revisions, not repeated inference counted as new evidence. See [the data contract](DATA_COLLECTION.md) for scene annotations versus individual recognition crops and gallery references.

```text
one source frame / captured page
  → variable-length localized physical card instances (zero, one, or multiple)
  → four-corner rectification + orientation + quality assessment
  → optional bounded OCR shortcut only for exceptionally clear crops
      → unique supported code: catalog identity candidate at declared granularity
      → otherwise skip/proceed immediately to the ordinary visual path
  → encoder → normalized full-card descriptor for each usable crop
  → exact cosine search → shortlist of distinct visual identities
  → selective local/detail/OCR verification on ambiguous candidates
  → calibrated unknown / candidate / ambiguous variant / confirmed result
  → calibrated verification / review + independent per-instance UI state
```

Start with a configurable top-20 distinct-identity shortlist as an experiment parameter, not a universal optimum. Measure Recall@K before evaluating the verifier: it cannot recover a true identity omitted by retrieval. Keep quality failure, not-enrolled identity, and unresolved variant distinct in results, even if all lead to a retry/review action.

For every scene, encode rectified card crops independently. Each recognition/gallery reference represents one card, while the full scene has a variable-length instance list. A whole-scene descriptor does not supply one identity per detected card. Preserve source resolution and crop transforms; increase acquisition resolution or guide closer capture when necessary. Encoder batch sizes are resource settings independent of scene count; process any-length lists in bounded chunks, including remainders. Never pad/truncate scene labels to nine.

## Detection geometry and perspective correction

An oriented bounding box (OBB) is a rotated rectangle, generally represented by center, width, height, and angle. It reduces background around cards rotated within the image plane, but cannot represent an arbitrary perspective quadrilateral. Four stored OBB vertices are not four independently estimated physical card corners. An OBB also does not establish which direction the artwork faces.

Compare three localization paths on the same independent captures:

| Path | Use / limitation |
| --- | --- |
| Generic card detector + OBB crop | Rotation-aligned baseline; perspective remains |
| Generic detector + local four-corner refiner | Reuse coarse localization, then predict/refine the actual card boundary at higher source resolution |
| Generic detector with four corner keypoints | Direct per-card localization plus quadrilateral geometry; bootstrap with renderer-known corners, then validate on real scenes |

Start with YOLO OBB and rotation-aligned crops. Compare these with renderer-known perspective crops to quantify whether a separate corner estimator would help. Add corner refinement or direct keypoints only when measured crop failures justify the complexity. Segmentation/contour fitting is another candidate when edges are reliable, but binder seams, sleeves, rounded corners, and overlap can mislead it; require a valid-quad check.

Detect generic card fronts rather than requiring the correct game label to permit localization/recognition. Define how backs and severely occluded cards are labeled/rejected. Improve the training set with varied card counts/layouts, real binder backgrounds/reflections, orientation/perspective, hard background negatives, and small-card examples. Synthetic compositions may bootstrap localization if corner geometry is transformed consistently; real camera testing is still required.

Preserve full capture resolution. Map detector outputs back through letterbox/scale/orientation transforms before taking corners/crops from the source image. Compare detector input resolution and optional overlapping tiles for small cards. Merge tile duplicates with compatible geometric suppression, without merging adjacent physical cards or identical copies. Do not upscale a tiny crop and treat it as recovered detail.

The audited detector archive has axis-aligned labels only. Expanding those boxes into four points cannot recover true card corners or supply trustworthy OBB supervision. Preserve existing labels as baseline/proposals. The implemented compositor supplies exact transformed physical-edge labels automatically; real-image teacher proposals need separate confidence/provenance and validation. With rounded corners, use the intersections of the straight card edges consistently.

OBB and keypoint (pose) exports are not interchangeable model files: each needs its own output decoding, geometry types, source transforms, suppression and quadrilateral crop extraction (the browser has both, `web/src/vision/obb.ts` and `pose.ts`). Pin an exporter/runtime version and verify real ONNX outputs rather than copying commands for a different model generation.

## Shared canonical preprocessing for queries and gallery inserts

Use two input adapters feeding one versioned canonicalizer and encoder-input contract:

```text
camera capture → detector/corner estimator → source quad + orientation ┐
                                                                      ├→ canonicalize → encoder tensor → embedding
gallery image → known/reviewed card bounds + orientation              ┘
```

An already cropped digital reference need not pass through a detector that can introduce errors. Use its verified full-card bounds and upright orientation; a photographed reference uses the same localization/rectification as a camera query. Background margins, a sleeve boundary, or a multi-card source must not silently count as a full-card reference. Review or reject uncertain enrollment geometry.

The shared deterministic stages are:

1. **Decode consistently:** apply image orientation, a documented color-profile/RGB conversion, alpha handling, and numeric range. Keep originals unchanged. Record transforms in oriented source coordinates.
2. **Validate geometry:** ordered finite, non-self-intersecting convex corners, adequate source size, consistent winding, and acceptable clipping/visibility. Reject degenerate warps. Geometric corner order and upright artwork orientation are separate concepts; do not infer card-up from the smallest coordinate sum.
3. **Rectify the full face:** map physical card corners to the configured canonical rectangle using a perspective transform. Preserve printed borders, names, and card numbers. The target aspect policy must match the supported card formats and be identical for gallery/query; other shapes require an explicit supported policy, not unnoticed stretching.
4. **Resolve orientation:** use known gallery orientation and a tested query orientation estimator. For ambiguous queries, compare a bounded set of rotated candidates only where needed; log the selected orientation and extra cost. A rotated OBB alone does not resolve upside-down artwork.
5. **Prepare encoder pixels:** resize the full card with fixed interpolation/antialiasing, then apply the same aspect/padding policy for the chosen model input. Compare full-card padding and other explicit encoder-specific policies; never silently introduce center cropping that removes identity details. A shared policy is per encoder experiment, not one arbitrary universal tensor size for every backbone.
6. **Normalize exactly:** same RGB order, tensor layout, dtype/range, per-channel mean/std, and graph preprocessing. Avoid the current average-standard-deviation export approximation. Normalize inside the exported model where practical, with explicit input expectations to prevent double normalization.
7. **Encode consistently:** evaluation mode, the identical encoder/weights and descriptor pooling/projection for both paths, then L2-normalize nonzero finite embeddings.

Quality assessment is shared where applicable, but a clean digital reference is not expected to exhibit camera motion. Treat blur/glare/clipping as explicit quality signals; a perspective warp cannot restore information hidden by reflection or occlusion. Start without aggressive sharpening, thresholding, histogram equalization, or generated repairs. Add deterministic photometric changes only after a controlled retrieval comparison; keep random augmentation in training rather than enrollment/query inference.

Enrollment should preserve the original, preview the canonical crop with its identity, and store source/geometry/encoder/preprocessing hashes plus the derived descriptor. Stage uncertain entries for review; publish gallery changes atomically after validation. Reject query/index contract mismatches even if vector dimensions happen to agree. Any canonicalization/normalization/encoder change requires rebuilding the affected gallery and rerunning ranking/rejection calibration.

Python, browser, and native implementations may use different imaging libraries, but they must implement the same contract. Share golden inputs with source corners/orientation, canonical crops, encoder tensors, and expected embeddings/rankings. Verify numerical tolerances and decisions on actual exports, including mirrored/rotated inputs and non-square aspect-fill previews; shape-only smoke tests are insufficient. Do not switch the existing trained classifier/app to a new crop policy without regression evaluation or the corresponding retraining/export.

Measure detection recall/duplicate false positives, source-normalized corner error, valid-warp rate, upright orientation accuracy, pixels per card, retrieval Recall@K/accepted precision, unknown rejection, and latency/memory by actual scene card count. Include failed detection/corner extraction in end-to-end results. Compare annotated-corner rectification as an upper-bound experiment with predicted geometry so encoder and localization failures can be distinguished.

## Encoder and loss comparison

All rows use the same independent queries and versioned preprocessing for a given encoder. Compare per-card and all-nine-correct page metrics; do not select an encoder from generic classification benchmarks alone.

| Candidate | Purpose | What must be established |
| --- | --- | --- |
| Current classifier and its trained features | Existing accuracy/cost baseline | Retrieval ranking and rejection versus current logits; restore the actual source checkpoint |
| Pretrained lightweight backbone without random feature projection | Cheap browser-oriented descriptor baseline | Full-card fidelity, real camera invariance, export parity, nine-card latency |
| Frozen DINOv2 ViT-S/14 | Broad visual-feature baseline | Pooling choice and resolution on cards, fine text/variant sensitivity, memory and export feasibility |
| Frozen DINOv3 ViT-S/16 or a larger offline model | Stronger candidate / offline reference model | Actual TCG improvement over simpler models, weight access/terms, and deployment cost |
| SSCD image-copy descriptor | Alternative objective suited to recognizing transformed source imagery | Camera/sleeve/perspective transfer; whether needed text/art differences survive its invariances |
| Compact encoder fine-tuned with ArcFace | Explicitly discriminative angular embedding | Improvement on unseen-in-training enrolled identities and confusing variants, not just training identities |
| Compact encoder with supervised contrastive training | Compare positive-view alignment and hard-negative discrimination | Gain over ArcFace and frozen encoders at the same capture split and input budget |

DINOv2/DINOv3 provide general visual features; their published results do not demonstrate TCG correctness or mobile-browser latency. SSCD targets copy detection, not physical-card recognition, and its reference repository is archived. Treat all as experiments. Do not use pretrained face-recognition weights as a card model merely because the loss is called ArcFace.

A stronger offline reference model is useful for finding the accuracy ceiling and labeling failure modes. If too expensive for nine-card browser inference, distill its representations or rankings into a compact student and evaluate that student independently. Gallery and queries must both use the student's encoder/version; teacher and student vectors are not interchangeable. Do not make a remote inference service a hidden requirement of this design.

## ArcFace training contract

Normalize feature vectors and class-proxy weights. During supervised training, apply an additive angular margin to the target-class logit with an explicit scale. This encourages a discriminative angular representation. At deployment, discard the training proxy/classifier head and compare normalized encoder outputs with enrolled references using cosine similarity. Adding an enrollment then changes gallery data rather than the training head.

That does not mean retraining is never needed: additional domains or failures may justify encoder improvement, which requires rebuilding the gallery. An ArcFace encoder can still overfit its training identities.

- Use several positive views of an identity: independent real captures where possible plus controlled synthetic perspective, blur, exposure, compression, glare, sleeve, and background changes. One reference with synthetic views is a bootstrap, not proof of camera generalization.
- Mine hard negatives from visually similar cards and reprints after automatic evidence checks; quarantine ambiguous labels rather than requiring the user to review them. Split first, then mine only from permitted training data; never use locked-test errors to tune the model or margin.
- Preserve evidence that defines the target identity. A crop/erasing augmentation that removes the only distinguishing card number can make the task contradictory. Do not train the model to ignore all text if exact printing recognition depends on text.
- Use balanced sampling across identities/games and multiple views per identity. Compare ArcFace and supervised contrastive recipes separately before combining losses or adding training complexity.
- Visually indistinguishable prints should remain an ambiguous visual family, not be forced apart by impossible image labels. Exact metadata may require user input or additional capture evidence.
- Hold out identities and entire game domains from task-specific encoder training. Enroll their references at evaluation time, then query with independent captures. For pretrained models, report that absence from upstream pretraining cannot generally be verified.

## Gallery is curated evidence, not ground-truth queries

The reference gallery represents known appearances. Ground-truth query labels belong to the independent benchmark. Using the enrollment reference itself as its test query measures self-match, not camera recognition. The current ten duplicated `manual_eval` images must never be used to choose this design's operating point.

Use explicit logical records rather than assigning all semantics to a filename:

| Record | Fields / responsibility |
| --- | --- |
| Catalog identity | Stable namespace, game, set, card number, language, art/printing variant where known; unknown fields stay unknown |
| Visual family | Optional relation among reprints/variants that share appearance or artwork; does not merge exact catalog identities |
| Reference image | Reference ID, catalog identity, original image path/hash, source/provenance, reference role, dimensions, quality flags, capture/session ID if applicable |
| Derived descriptor | Reference ID, region/descriptor type, encoder and preprocessing hashes, dimension/dtype, normalization, artifact version |
| Query capture | Independent capture/session ID, source image/frame, corners/instance location, ground-truth identity or unknown; split role separate from reference role |
| Physical card instance | Session/page/pocket or instance ID, transient track mapping, selected query crops and recognition evidence; never the gallery reference ID |

Begin with one good canonical front reference per identity and add a small, quality-controlled set of independently sourced front views where they improve real-camera matching. Preserve originals and variant IDs, including One Piece `_p1` suffixes. Record watermarks, borders, scan artifacts, and sleeve/glare conditions; test whether the encoder is matching these nuisances instead of card content.

Do not dump every synthetic augmentation into the gallery. Use augmentation primarily for training, then measure whether a few derived templates improve retrieval. Keep the reference count controlled per identity. A maximum-over-templates scorer gives identities with many references more chances to win; compare capped reference sets, per-identity aggregation, and rejection calibration as the gallery grows. Do not blindly average distinct appearance modes into one prototype.

Use full-card descriptors for initial retrieval. Optionally store detail/region descriptors or local features for the shortlisted references. A known reference can guide aligned region comparison; do not assume that all games put their artwork, set code, or number in the same location. Store this extra evidence only when measured accuracy justifies its storage and runtime cost.

Version gallery/catalog/thresholds as a coherent bundle. A model or preprocessing change requires re-embedding references; new enrollment requires recalibrating or at least regressing unknown rejection and confusing-neighbor behavior. No vector database service is needed for the current 1,399-card corpus: exact matrix search is the correctness baseline.

## Selective verification and rejection

Retrieve distinct identities rather than spending the shortlist on many templates of one card. Use best-versus-next-distinct-identity margin and absolute similarity as calibration features, not probabilities. Measure the joint operating point against not-enrolled cards, backs, pocket seams, and non-cards.

A readable code with unique catalog support can produce the initial identity candidate without visual encoding. For ambiguous candidates, test local feature correspondences, aligned detail comparison and additional OCR/catalog evidence. Glare, tiny text and shared numbering require fallback or an unresolved result; return a visual family when exact printing cannot be established. Code/visual disagreement must not be silently overridden by the larger score.

Do not apply expensive reranking to all nine cards on every frame. Run the cheap first stage for eligible crops, reuse confirmed-instance evidence, and rerank only unresolved instances under a bounded budget. A verifier disagreement or timeout yields a reviewable uncertain result, not an automatic acceptance of the nearest reference. The UI's verifying state must correspond to actual pending verification or recent-evidence confirmation.

## Decisive experiments

1. Run the implemented synthetic bootstrap first, then expand independent real capture/calibration/locked-test data through automated ingestion and annotation proposals. Include empty scenes, singles, loose groups, partially filled pages, full nine-card pages, and scenes with more than nine. Preserve every eligible missed ground-truth instance in the denominator; annotations are variable-length lists, not nine slots. Synthetic-only results remain separately labeled.
2. Enroll current canonical references; compare frozen descriptors before fine tuning. Test a controlled full-card preprocessing contract for each model and report changes explicitly.
3. Compare single-reference versus a capped multi-reference gallery with the same queries and distinct-identity scoring. Report storage and per-identity reference counts.
4. Compare frozen features, ArcFace fine tuning, and supervised contrastive fine tuning on task-training-held-out identities. Include a second game, then a game-held-out evaluation before claiming broad domain support.
5. Compare global-only search versus selective verification at the same accepted precision/coverage point. Report Recall@1/5/20, wrong-variant rate, unknown false accepts, page-complete success, first-result/page latency, and phone memory.
6. If distilling/quantizing for deployment, rebuild the gallery and rerun the same fixed evaluation. Validate ONNX output/ranking parity and nine-card performance in Safari on iPhone and Chrome on Android.

**Decision rule:** choose the least expensive complete pipeline that meets the roadmap's independently measured accuracy, rejection, and page-scanning goals. Keep the stronger offline encoder as a reference if useful; do not choose the runtime architecture based on the word ArcFace, model size, or a perfect self-match score.

## Primary references checked 2026-10-15

- [ArcFace paper](https://arxiv.org/abs/1801.07698): additive angular-margin learning for discriminative normalized embeddings; face results do not establish card performance.
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362): supervised positive/negative representation learning.
- [DINOv2 implementation and models](https://github.com/facebookresearch/dinov2): pretrained visual backbones, including the small ViT baseline.
- [DINOv3 implementation and models](https://github.com/facebookresearch/dinov3): dense visual representations, smaller distilled models, and model-access instructions.
- [SSCD implementation](https://github.com/facebookresearch/sscd-copy-detection): normalized copy-detection descriptors and cosine comparison; an alternative retrieval baseline, with an archived reference codebase.
- [DELG paper](https://arxiv.org/abs/2001.05027): global retrieval and local-feature evidence in image search.
- [Ultralytics OBB](https://docs.ultralytics.com/tasks/obb/): rotated-rectangle outputs and angle/corner representation; four output vertices do not imply arbitrary perspective geometry.
- [Ultralytics pose dataset format](https://docs.ultralytics.com/datasets/pose/): custom keypoint shapes and visibility annotation, relevant to a four-corner card model.
- [OpenCV geometric transforms](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html): four corresponding points define a perspective warp.

The proposed TCG-specific gallery, training, and verification choices above are engineering hypotheses to test, not measured conclusions from these papers.
