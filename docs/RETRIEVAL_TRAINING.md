# Card encoder training

The scanner identifies a card by embedding its rectified crop and searching the enrolled
gallery by cosine similarity. The encoder is a pretrained DINOv2-small backbone fine-tuned
with ArcFace; the ArcFace class proxies are discarded after training, so enrolling new
references never needs a new classifier head. Code: `training/recognition/`.

## Deployed run

`models/card-encoder/dinov2-arcface-gpu-20261017` (DVC), trained with
`training/recognition/configs/dinov2_arcface_gpu.yaml` on the RTX 4070 Super
([GPU_TRAINING.md](GPU_TRAINING.md)) in 38 minutes.

| | |
| --- | --- |
| Backbone | `facebook/dinov2-small`, revision `ed25f3a3`, weights hash-pinned (`models/pretrained/dinov2-small`) |
| Descriptor | CLS token → 384×384 projection (identity-initialized) → L2 normalization |
| Objective | ArcFace, margin 0.2 rad, scale 16 |
| Trainable | projection and proxies from the start; last 4 transformer blocks + final LayerNorm after a 1-epoch warm-up |
| Optimizer | AdamW, cosine decay; learning rates backbone 3e-5, projection 1e-4, proxies 1e-3 |
| Budget | 14 sampling epochs × 150 steps = 2,100 steps; 32 identities × 2 views = 64 images per step |
| Training pool | 6,000 visual classes (6,001 catalog identities), game-balanced, including all 188 real train crops |
| Augmentation | whole-card perspective with corners kept inside, mild exposure/color/contrast, blur, JPEG, glare; no erasing or generated SAMPLE removal |
| Input contract | `tcg-full-card-v1` (`canonical.py`): 224×312 rectified card, aspect fit into 224×224 with mean padding, ImageNet normalization |
| Selected | epoch 12 by validation Recall@1, then Recall@5, then true-vs-impostor margin |

Full-catalog results (all 24,197 references enrolled, exact matrix search):

| Split | Queries | Recall@1 | Recall@5 | Mean margin true vs best other |
| --- | --- | --- | --- | --- |
| val (Pokémon) | 24 | 20/24 | 24/24 | 0.250 |
| test (One Piece) | 86 | 85/86 | 86/86 | 0.276 |

These are retrieval diagnostics on AI-reviewed crops from few videos with repeated views,
not scanner accuracy. Unknown-card rejection is uncalibrated.

## Data isolation

Splits are by source video (source group) and fixed in `datasets/curated`. Before
sampling, the trainer removes from all gradients the 39 val/test query identities, their
One Piece base families across languages and art suffixes, and exact-byte aliases
(61 catalog identities in total), plus the known same-art pair `OP09-014`/`OP09-014_p1`.
During training, each epoch evaluates the 24 validation crops against a fixed
6,062-reference distractor gallery; these interim numbers are not full-catalog results.
Test pixels are read once, after checkpoint selection. No threshold or retraining
decision uses test results.

## Outputs and safeguards

A run folder holds `resolved_config.json`, `training_plan.json`, `encoder_contract.json`
(backbone, preprocessing and source hashes), per-epoch validation reports, `last.pt`,
`best.pt`, `full_catalog_{val,test}.json`, `gallery_embeddings.npy` and `gallery_index.json`
(checkpoint/vector hashes and reference order). The trainer refuses silent CPU fallback,
changed data or config on resume, non-finite loss or gradients, and missing or
hash-mismatched inputs. The positional-embedding grid for the fixed 224×224 input is
interpolated once on CPU (identical outputs), which also keeps Apple MPS usable for
small checks.

## Commands

```bash
# new run (on the GPU PC); output must be below runs/recognition/
.env/bin/python training/recognition/train_retrieval.py \
  --config training/recognition/configs/dinov2_arcface_gpu.yaml \
  --output runs/recognition/retrieval/<new-run>

# resume an interrupted run with the identical config and dataset
.env/bin/python training/recognition/train_retrieval.py \
  --config training/recognition/configs/dinov2_arcface_gpu.yaml \
  --output runs/recognition/retrieval/<new-run> --resume
```

A new run replaces the deployed encoder only after it beats it on the same full-catalog
val/test evaluation. Then move it to `models/card-encoder/`, `dvc add` it, re-export with
`scripts/release/export_browser_models.py` (fp32 and int8), make fp16 copies with `scripts/release/compare_model_precisions.py` and rebuild the
bundle; the gallery must always be re-embedded with the new encoder.

## History

| Run | Hardware | Classes / trainable blocks / steps | Val R@1 | Test R@1 |
| --- | --- | --- | --- | --- |
| `dinov2-arcface-pilot-20261016` (retired) | Apple M1, MPS | 2,048 / 2 / 1,000 | 19/24 | 84/86 |
| `dinov2-arcface-gpu-20261017` (deployed) | RTX 4070 Super | 6,000 / 4 / 2,100 | 20/24 | 85/86 |

Before that, a MobileNetV3 classifier and frozen-encoder baselines were used; they are
archived ([FOUNDATION_MODELS.md](../archive/docs/FOUNDATION_MODELS.md)). Candidates for
the next encoder (DINOv3, ConvNeXt, SigLIP-family) need the same fine-tuning, isolation
and evaluation; see [RECOGNITION_DESIGN.md](RECOGNITION_DESIGN.md).
