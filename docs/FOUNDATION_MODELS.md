# Pretrained models without card training

Updated 2026-10-15. A pretrained model can provide localization or image descriptors without card-specific training. This is an experiment to measure before deciding whether additional training is useful. Enrollment still needs correct catalog identities, source images and compatible descriptors; model size alone does not guarantee exact-card recognition.

**Latest direction:** the frozen experiments below are retained results. Do not repeat them or expand bulk frozen-DINO enrollment. Next recognition work is card-specific fine tuning of a pretrained backbone with a retrieval objective, while finishing the source-quality/crop pipeline. The partial full-catalog frozen index was cancelled and its incomplete vectors removed; raw catalog images remain available for rebuilding with the adapted encoder.

| Role | Candidate | What it supplies | What still needs evidence |
| --- | --- | --- | --- |
| Localization teacher | GroundingDINO | Text-prompted axis-aligned boxes | Physical card versus screen graphic, complete boundaries, corners, recall on independent scenes |
| Frozen retrieval encoder | DINOv2/DINOv3 | Reusable image features for cosine search | Exact ID/variant recall, unknown rejection, capture quality and deployment speed |
| Compact baseline | ImageNet MobileNetV3 | Pooled features without card fine tuning | Current retrieval result is weak; do not select it merely because it runs |
| Optional later training | ArcFace / supervised contrastive objective | Adapted embedding geometry | Improvement on held-out identities and independent captures over frozen features |

GroundingDINO and DINO image encoders have different interfaces and objectives. A detector's “trading card” label does not identify the catalog entry. Retrieval can enroll new identities without changing a fixed classifier head, but visually indistinguishable printings and erased details remain ambiguous.

## Executed localization probe

`scripts/run_foundation_detector.py` ran public `IDEA-Research/grounding-dino-tiny` at pinned revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, with frozen weights, no card-specific training, CPU / four torch threads, shortest edge 800 and longest edge 1333. Each of two prompts used box/text thresholds 0.20. Preprocess/forward/postprocess took **3.94–5.59 seconds per prompt**; model loading is excluded. This is a desktop measurement, not a browser/phone benchmark.

- One Piece binder frame 240: both prompts matched **11/11 approximate physical-card teacher boxes at IoU ≥0.5**. These are visual-teacher proposals, not independently verified annotations. The detector also matched the excluded digital Shanks overlay at IoU about **0.965**, and produced other proposals such as the binder outline.
- Pokémon pack frame 255: both prompts matched the **single approximate held-card box at IoU 0.96–0.97**. Prompt choice changed extra detections substantially.
- The synthetic-trained YOLO OBB on the same One Piece frame produced 19 accepted crop proposals, with best physical teacher-box overlaps only **0.00–0.22**. This identifies a severe real-domain gap in the current checkpoint; it does not establish that YOLO OBB as a method cannot work.

The manifest and inspectable overlays are under `runs/foundation_teacher/run_01/`. Outputs remain proposals: no exact identity, true corner or training promotion is claimed. Two selected frames are insufficient to estimate broad detection accuracy. A prompt containing “physical” did not solve screen-graphic filtering.

`scripts/crop_foundation_proposals.py` connected the saved first-prompt predictions to **23 source-resolution AABB crops**, preserving source/model hashes and leaving physical status unknown. OCR/catalog lookup on 16 prioritized proposals returned **zero resolved identities**. It read `048/063` on the Japanese Pokémon crop, but this did not match the English catalog. The One Piece code reads remained insufficient. Results are under `runs/foundation_teacher/crop_ocr_01/`; improved localization alone has not solved recognition, and these crops were not promoted to exact-label training.

Reproduce with the existing Python/torch environment and optional dependencies from `training/card_detector/requirements-foundation.txt` installed into an isolated target. The script's `--runtime` selects that target; it never needs remote model code or a card classifier. Public weight acquisition needs network access and about 700 MB storage. The original run used:

```bash
.env/bin/python scripts/run_foundation_detector.py \
  --runtime runs/foundation_teacher/runtime \
  --annotations runs/video_mining/luna-onepiece/teacher_annotations.json \
    runs/video_mining/luna-pokemon/teacher_annotations.json \
  --output runs/foundation_teacher/new_probe
```

The output must be new/empty. These local media/annotations are ignored experiment artifacts; they are not guaranteed in another checkout. Repeated runs currently create a model cache inside the selected output, so account for disk use.

## Retrieval evidence and next decision

`scripts/benchmark_frozen_retrieval.py` completed the frozen DINOv2-small comparison on the same **1,431 references / 1,321 synthetic true-corner crops**, with no card-specific training:

| Encoder / query policy | Recall@1 | Recall@5 | Recall@20 |
| --- | --- | --- | --- |
| Frozen MobileNet, stored query orientation | 24.9% | 38.2% | 53.8% |
| Frozen DINOv2-small, same stored orientation | **37.4%** | **48.3%** | **61.1%** |
| DINOv2-small, maximum over 0°/180° query views | 35.8% | 46.0% | 59.2% |
| DINOv2-small, patch-mean pooling | 37.9% | 52.2% | 64.6% |
| DINOv2-small, normalized CLS + patch-mean concatenation | 39.7% | 52.2% | 65.9% |

The optional orientation sweep increased false competition on this set; it did not improve recall. DINOv2 used pinned `facebook/dinov2-small` revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, frozen 384-D CLS features, L2 normalization and the same full-card fit/pad 224-pixel tensor policy for gallery/query. Four-thread CPU encoding took about **55 seconds for 1,431 references** and **56 seconds for 1,321 query crops** per orientation. Model loading is excluded. This is a compact foundation-model baseline; larger backbones, higher input resolution and local-feature verification have not been compared here.

These queries derive from gallery source images, include the synthetic capture degradations and exclude detector failures. No real-camera performance or calibrated rejection is established, and these recall levels are insufficient for reliable scanning. Results and hashes are in `runs/card_recognizer/dinov2_baseline/report.json`; this evidence supports further frozen-feature experiments before choosing a training objective, not a universal recognition claim.

`scripts/benchmark_dino_pooling.py` produced the two pooling ablations with the same 1,321-query set, one forward per image and no extra weights/training. CLS parity with the earlier report was asserted. Its report is `runs/card_recognizer/dinov2_pooling/report.json`; the small gain does not establish a deployable encoder. Changing pooling changes the descriptor contract and requires rebuilding the index.

`scripts/build_raw_dino_gallery.py` streams downloaded catalog fronts through the shared canonicalizer and frozen CLS encoder without writing thousands of extra crop files. It verifies provider image hashes, language/identity mappings and official model bytes; byte-identical stale filename aliases are recorded once, while conflicting images for one catalog identity cause failure. A raw-source versus prepared-reference descriptor parity check returned cosine 1.0 on the tested fixture. The full-catalog index is a separate enrollment artifact; the synthetic benchmark denominators above remain 1,431 references.

Use the stronger offline localizer to produce source-resolution crop proposals. Similarity is the normal recognition path, with an optional OCR shortcut on exceptionally clear inputs. Next adapt a pretrained encoder with a card retrieval objective, preserve unknowns/graphic errors, and evaluate on source-held-out correctly evidenced queries. The browser need not run the same large teacher used for dataset creation.

## Primary references

- [GroundingDINO official implementation](https://github.com/IDEA-Research/GroundingDINO)
- [Transformers 4.46.3 GroundingDINO interface](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/grounding-dino)
- [Pinned public GroundingDINO model](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71)
- [DINOv2 official implementation and pretrained models](https://github.com/facebookresearch/dinov2)
- [DINOv3 official implementation](https://github.com/facebookresearch/dinov3)
- [SigLIP2 paper](https://arxiv.org/abs/2502.14786)
- [ArcFace paper](https://arxiv.org/abs/1801.07698)
- [Torchvision ConvNeXt backbones](https://docs.pytorch.org/vision/main/models/convnext.html)
