# Training on the LAN GPU machine

The Mac (Apple M1) prepares data, evaluates, exports and builds the web bundle; the GPU PC
only trains. Apple's MPS backend cannot train Ultralytics pose (keypoint) models
(Ultralytics issue 4031) and is several times slower for everything else. The GPU PC also
hosts the DVC remote (`gpu-pc` in `.dvc/config`).

## Machine

- Windows PC, WSL2 Ubuntu 24.04 with mirrored networking, `kono@192.168.178.38`
  (SSH key from the Mac; WSL OpenSSH on port 22). DVC store: `/mnt/d/tcg-scanner-dvc`.
- NVIDIA RTX 4070 Super (12 GB), 16 cores, 30 GB RAM.
- `~/tcg-train/.venv`: Python 3.12 with `requirements.txt` (torch 2.4.0+cu124 instead of
  the CPU wheel).
- `~/tcg-train/repo`: a partial mirror of this repository: code plus the inputs below.

## Inputs on the PC

| Path | Size | Used by |
| --- | --- | --- |
| `datasets/curated/` | 3.0 GB | encoder training (gallery + labeled query crops) |
| `models/pretrained/dinov2-small/` | 84 MB | encoder initialization |
| `models/corner-detector/corners-s640-v4-ft/` | 23 MB | detector fine-tuning start point |
| `runs/detection/<corner dataset>/` | ~1.2 GB | detector training (built by `build_corner_dataset.py`) |
| `training/`, `scripts/`, `requirements.txt` | small | code |

Get data with `dvc pull <pointer>` in the mirror, or rsync from the Mac. YOLO datasets
built with symlinks must be copied with `rsync -L`; afterwards fix the absolute `path:` in
their `dataset.yaml`.

The mirror predates the 2026-10-19 relayout. After syncing code, run
`python scripts/maintenance/relayout_2026_10.py` there once: it moves the old `runs/...`
inputs to the new paths (missing ones are skipped), rewrites paths stored inside them,
and `dvc add`s the results. That includes the training set of the deployed detector,
`runs/card_detector/corners-v4mix`, which exists only on the PC and becomes
`datasets/detection/corners-v4mix`; `dvc push` it and commit its pointer.

## Card encoder

```bash
python training/recognition/train_retrieval.py \
  --config training/recognition/configs/dinov2_arcface_gpu.yaml \
  --output runs/recognition/retrieval/<new-run>
```

The deployed run took 38 minutes. Recipe and evaluation:
[RETRIEVAL_TRAINING.md](RETRIEVAL_TRAINING.md).

## Corner detector

The deployed `corners-s640-v4-ft` is YOLOv8s-pose at 640 px. Its saved training arguments
record a five-epoch fine tune from `corners-s640-v4/weights/last.pt` on `corners-v4mix`
(synthetic renderer-v4 scenes plus AI-reviewed real corners), batch 24, seven workers,
learning rate 0.002, no warm-up, mosaic off. Parent and intermediate checkpoints are
retired; new fine tunes start from the deployed checkpoint.

```bash
# on the Mac: scenes and the YOLO dataset
.env/bin/python scripts/detection/generate_detection_scenes.py --output runs/detection/scenes-v4
.env/bin/python scripts/detection/build_corner_dataset.py --scenes runs/detection/scenes-v4 \
  --real datasets/detection/real-corners/train_labels_v1.json --max-tilt 50 \
  --output runs/detection/corners-v5mix
# on the PC
python training/detection/train_detector.py --data runs/detection/corners-v5mix/dataset.yaml \
  --model models/corner-detector/corners-s640-v4-ft/weights/best.pt --imgsz 640 --batch 24
```

Encoder and detector training fit on the GPU together (about 11.7 GB). Logs go to
`~/tcg-train/logs/`.

## Back on the Mac

Copy finished runs into `runs/`, then evaluate and build here:
`scripts/detection/evaluate_corners_real.py` and `compare_detectors_real.py` for the
detector, full-catalog val/test for the encoder, then
`scripts/release/export_browser_models.py`, int8 quantization,
`scripts/release/build_browser_bundle.py` and the web unit and browser tests. A model
replaces the deployed one only when it wins these checks; then move it into `models/`,
`dvc add`, `dvc push` and commit the pointer.
