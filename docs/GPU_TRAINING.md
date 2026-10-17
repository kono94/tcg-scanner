# Training on the LAN GPU machine

Status: used 2026-10-17 for the corner detector and the second card-embedding run. The Mac
(Apple M1) stays the place for data preparation, evaluation, export and the web bundle; the GPU
machine only trains. Apple's MPS backend cannot train Ultralytics pose (keypoint) models
(Ultralytics issue 4031), and is several times slower for everything else.

## Machine

- Windows PC, WSL2 Ubuntu 24.04 with mirrored networking, reachable as `kono@192.168.178.38`
  (SSH key from the Mac; WSL OpenSSH on port 22).
- NVIDIA RTX 4070 Super (12 GB), 16 cores, 30 GB RAM.
- `~/tcg-train/.venv`: Python 3.12, torch 2.4.0+cu124, torchvision 0.19.0, ultralytics 8.2.82,
  transformers 4.46.3 (pip, not the macOS runtime folder), onnx/onnxruntime for checks.
- `~/tcg-train/repo`: a partial mirror of this repository (training code, scripts, the data below).
  `runs/foundation_teacher/runtime` exists there only as an empty folder; the trainer adds it to
  `sys.path`, and the packages come from the venv instead.

## What gets copied (rsync from the Mac)

| Path | Size | Used by |
| --- | --- | --- |
| `datasets/curated/` | 3.0 GB | card embedding (gallery + labeled query crops) |
| `runs/card_recognizer/dinov2_baseline/hf_cache` | 84 MB | pinned DINOv2-small weights |
| `runs/card_detector/corners-mix` (copied with `-L`, symlinks resolved) | ~1.2 GB | corner detector |
| `training/card_detector`, `training/card_recognizer`, corner/eval scripts | small | code |

After copying a YOLO dataset, rewrite the absolute `path:` in its `dataset.yaml` for the PC.

## Runs

Corner detector (box + four card corners, YOLOv8s-pose):

```bash
python training/card_detector/train_obb.py --data runs/card_detector/corners-mix/dataset.yaml \
  --model yolov8s-pose.pt --epochs 80 --time 0.83 --imgsz 640 --batch 32 --workers 8 --device 0 \
  --degrees 10 --close-mosaic 5 --patience 25 --cos-lr --name corners-s-gpu
```

`corners-mix` = upright renderer-v3 scenes plus renderer-v2 scenes without sideways cards
(`scripts/build_corner_dataset.py --max-tilt 50`). Sideways cards make corner names ambiguous;
upside-down cards are fine (corner order is geometric).

Card embedding (DINOv2-small + ArcFace), `training/card_recognizer/configs/dinov2_arcface_gpu.yaml`:

```bash
python training/card_recognizer/train_retrieval.py \
  --config training/card_recognizer/configs/dinov2_arcface_gpu.yaml \
  --output runs/card_recognizer/retrieval/dinov2-arcface-gpu-20261017
```

Both fit on the 12 GB GPU at the same time (about 11.7 GB together). Logs go to `~/tcg-train/logs/`.

## Back on the Mac

Copy the finished run folders back into `runs/`, then evaluate and build the bundle here:
`scripts/evaluate_detector_geometry.py`, `scripts/compare_detectors_real.py`,
`scripts/export_browser_models.py`, int8 quantization, `scripts/build_browser_bundle.py`, and the
web unit/browser tests. A model only replaces the current one when it wins these checks.
