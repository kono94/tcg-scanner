# Card recognition code

Python side of recognition: the card encoder (pinned DINOv2-small fine-tuned with ArcFace,
normalized 384-D embeddings for exact cosine gallery search) and the preprocessing contract
the browser mirrors. Enrolling references never expands a classifier head.

| File | Role |
| --- | --- |
| `canonical.py` | `tcg-full-card-v1`: perspective warp to the canonical card and encoder tensor; the TypeScript port in `web/src/vision/preprocess.ts` must stay bit-exact |
| `backbone.py` | pinned DINOv2-small id/revision, loader for `models/pretrained/dinov2-small`, file hashing |
| `retrieval_training.py` | training plan and isolation, augmentation, embedding head, ArcFace |
| `train_retrieval.py` | training/evaluation loop, checkpoint selection, full-catalog evaluation |
| `configs/dinov2_arcface_gpu.yaml` | config of the deployed encoder |
| `ocr_identity.py`, `identity_evidence.py` | offline OCR (Tesseract) card-code reading and catalog evidence for label proposals |

```bash
.env/bin/python training/recognition/train_retrieval.py \
  --config training/recognition/configs/dinov2_arcface_gpu.yaml \
  --output runs/recognition/retrieval/<new-run>
```

Train on the LAN CUDA machine ([GPU_TRAINING.md](../../docs/GPU_TRAINING.md)). Recipe,
isolation, results and promotion: [RETRIEVAL_TRAINING.md](../../docs/RETRIEVAL_TRAINING.md).
Data: `datasets/curated/recognition/manifest.json` (gallery references and source-isolated
real query crops).
