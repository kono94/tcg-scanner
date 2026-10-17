# Retrieval fine-tuning pilot

The first DINOv2-small/ArcFace pilot was launched on 2026-10-16 using the local
Apple GPU through PyTorch MPS. Read its live status rather than assuming it has
finished:

- `runs/card_recognizer/retrieval/dinov2-arcface-pilot-20261016/status.json`
- `runs/card_recognizer/retrieval/dinov2-arcface-pilot-20261016.log`
- `runs/card_recognizer/retrieval/dinov2-arcface-pilot-20261016-launch.json`

## Recipe and scope

`training/card_recognizer/train_retrieval.py` is separate from the legacy
classifier trainer. Its configuration is
`training/card_recognizer/configs/dinov2_arcface_pilot.yaml`.

- Pinned, cached `facebook/dinov2-small`, revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`; pretrained weight hash checked.
- CLS features → learned 384-dimensional projection → L2-normalized descriptor.
  The projection starts as an identity matrix. ArcFace uses a 0.2-radian margin
  and scale 16. Training proxies are discarded for deployment.
- One projection/proxy warm-up sampling epoch, then the final two transformer
  blocks and final backbone LayerNorm are fine-tuned. Separate learning rates:
  backbone 3e-5, projection 1e-4, proxies 1e-3; AdamW with cosine decay.
- Ten sampling epochs × 100 optimizer steps = **1,000 steps**. These are fixed
  sampling budgets, not full passes through every image/class.
- Eight different training identities per batch, two augmented views each:
  **16 images**. Half the sampled classes have real training views, where
  available; their positive pair uses a reference and a real query.
- **2,048 training visual classes / 2,049 catalog identities**, including all
  **188 real training crops / 119 real-query identities** and a deterministic,
  game-balanced gallery sample. One exact-byte alias pair shares a proxy.
  This pilot does not train on every acquired reference identity.
- Whole-card geometric perturbation keeps all corners inside the canvas;
  mild exposure/color/contrast, blur, JPEG and glare augmentations. No erasing,
  generated SAMPLE removal or invented missing card content.
- Query and gallery share `tcg-full-card-v1` canonicalization, 224×312 card
  geometry and mean-padded, normalized 224×224 encoder tensors.

## Isolation and evaluation

Training selection starts after source/identity isolation. The **39 validation
and test query identities** are excluded from training, including related
One Piece base IDs across languages/art suffixes and exact-image aliases.
The known same-art Limejuice printing pair is also excluded. This removes
**61 catalog identities** from the training pool; none of the 188 accepted
training queries is lost. All source videos stay in their existing split.

Each sampling epoch evaluates the 24 validation crops against a fixed
**2,110-reference distractor gallery** containing the selected training
references and excluded/held-out identity families. These interim Recall@K
values are **not full-catalog results**. Validation Recall@1, then Recall@5,
then mean true-versus-impostor similarity margin select the best checkpoint.

After selection, the trainer rebuilds embeddings for **all 24,197 enrolled
references** and evaluates validation and test crops. Test pixels are first
loaded at that final stage. It writes `full_catalog_val.json`,
`full_catalog_test.json`, `gallery_embeddings.npy` and `gallery_index.json`.
The index records checkpoint/vector hashes and reference ordering. Unknown
rejection thresholds and export/runtime parity remain uncalibrated/unverified.
No test-driven retraining or threshold selection occurs in this run.

Repeated views, few videos, AI-reviewed labels and gallery-derived synthetic
augmentation limit these diagnostics. They do not establish end-to-end
scanner accuracy or broad cross-game generalization.

## Apple GPU and checkpoints

The selected PyTorch 2.4.0 MPS backend lacks bicubic positional-grid resizing.
The trainer computes the frozen pretrained grid once on CPU for the fixed
224×224 input, then moves the model to MPS. A full-backbone parity check found
**zero maximum absolute output difference** before/after that preparation.
Training forward/backward runs on MPS with CPU operator fallback disabled.
Pixel decoding/augmentation remains CPU work. An actual MPS backward step,
backbone-weight update, checkpoint save and resume passed the smoke check.
See `runs/card_recognizer/retrieval/positional_grid_parity.json`.

`last.pt` is saved every sampling epoch; `best.pt` is selected only on validation.
The run records the resolved configuration, data selection, model/preprocessing
contract, source snapshots and hashes. It refuses silent CPU fallback, changed
resume data/config, nonfinite loss/gradients and missing/hash-mismatched inputs.

Launch in a new output directory with GPU access:

```bash
.env/bin/python training/card_recognizer/train_retrieval.py \
  --config training/card_recognizer/configs/dinov2_arcface_pilot.yaml \
  --output runs/card_recognizer/retrieval/<new-run>
```

Resume an interrupted run from its last complete sampling epoch using the
identical configuration and dataset:

```bash
.env/bin/python training/card_recognizer/train_retrieval.py \
  --config training/card_recognizer/configs/dinov2_arcface_pilot.yaml \
  --output runs/card_recognizer/retrieval/<existing-run> --resume
```

The old app models are not replaced by this experiment.
