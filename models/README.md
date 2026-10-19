# Models

Weights and exports the scanner depends on, versioned with DVC. Training writes new runs
to `runs/` (local scratch); a run moves here only when it replaces the current model
(promotion rules in [AGENTS.md](../AGENTS.md)).

| Folder | What it is | Produced by |
| --- | --- | --- |
| `pretrained/dinov2-small/` | `facebook/dinov2-small` at revision `ed25f3a3`: `config.json` + `model.safetensors`; the encoder's initialization | Hugging Face download, hash-pinned in `training/recognition/backbone.py` |
| `card-encoder/dinov2-arcface-gpu-20261017/` | deployed card encoder: checkpoints, encoder contract, full-catalog evaluation, gallery embeddings | `training/recognition/train_retrieval.py` |
| `corner-detector/corners-s640-v4-ft/` | deployed YOLOv8s-pose four-corner detector with export and real-frame parity reports | `training/detection/train_detector.py` |
| `browser-export/gpu-20261017/` | ONNX encoder (fp32 + int8), ONNX detector, float16 gallery and golden fixtures for the web parity tests | `scripts/release/export_browser_models.py` |
| `browser-export/corner-decoder-fixture/` | raw detector output and Ultralytics-decoded corners for the TypeScript decoder test | `scripts/release/build_browser_bundle.py` |

`web/public/bundle` (built by `scripts/release/build_browser_bundle.py` from these exports,
the curated catalog and `datasets/prices/prices.json`) is what the browser downloads.
Training evidence: [docs/RETRIEVAL_TRAINING.md](../docs/RETRIEVAL_TRAINING.md),
[training/detection/README.md](../training/detection/README.md),
[docs/BROWSER_BUNDLE.md](../docs/BROWSER_BUNDLE.md).
