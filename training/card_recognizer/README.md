# Card retrieval training

The current recognizer fine-tunes a pinned DINOv2-small backbone with ArcFace and
exports normalized 384-dimensional embeddings for exact cosine gallery search.
Adding enrolled references does not expand a deployed classifier head.

```bash
.env/bin/python training/card_recognizer/train_retrieval.py \
  --config training/card_recognizer/configs/dinov2_arcface_gpu.yaml \
  --output runs/card_recognizer/retrieval/<new-run>
```

Use the LAN CUDA machine for training; see [GPU training](../../docs/GPU_TRAINING.md).
[Retrieval training](../../docs/RETRIEVAL_TRAINING.md) records data isolation,
checkpoint selection and evidence limits. `canonical.py` defines the shared query
and gallery contract; `retrieval_training.py` defines augmentation, embedding and
ArcFace training. `foundation_retrieval.py` loads pinned backbone files and hashes
artifacts. `train_retrieval.py` owns the training/evaluation loop.

The dataset is `datasets/curated/recognition/manifest.json`, including references
and source-isolated single-card queries. The original MobileNet classifier,
manual-evaluation fixtures and frozen descriptor tools have been removed.
