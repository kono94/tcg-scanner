# Card-corner detector

The browser uses YOLOv8s-pose at 640 px from
`runs/card_detector/pose/corners-s640-v4-ft/weights/best.pt` (DVC). The model predicts
a box and four geometric corners; the browser perspective-warps each card from them.

## Dataset format

Training uses Ultralytics **YOLO pose**, not a Hugging Face dataset:

```text
images/train/*.jpg
images/val/*.jpg
labels/train/*.txt
labels/val/*.txt
dataset.yaml
```

Each label row is `class cx cy w h x1 y1 v1 ... x4 y4 v4`, with normalized image
coordinates and four visibility values. Class `0` is `card`. The dataset YAML has
`kpt_shape: [4, 3]`, `flip_idx: [1, 0, 3, 2]` and `names: {0: card}`. Scene card
counts are variable; an empty scene has an empty label file. Keypoints follow
geometric TL, TR, BR, BL portrait order; artwork orientation is separate.

`scripts/generate_detection_scenes.py` and `scene_renderer.py` generate synthetic
scenes while retaining exact source-pixel corners in JSON.
`scripts/make_scene_replay.py` encodes a few validation scenes as the camera-mock
test video `synthetic_replay.mp4` (copied into `web/public/media` by
`web/scripts/link-media.mjs`).
`scripts/build_corner_dataset.py --max-tilt 50` converts them and reviewed real
corners to YOLO pose labels. Sideways scenes are excluded to avoid ambiguous
corner names; upside-down artwork is valid. `real_corner_labels.py` produces
reviewable proposals/labels; `evaluate_corners_real.py` compares complete reviewed
frames with the browser-equivalent filter, square letterbox and NMS.

The current checkpoint combines synthetic pretraining with AI-reviewed real-corner
fine tuning. Its saved training arguments record five final epochs at 640 px,
batch 24, initial learning rate 0.002 and mosaic disabled. Earlier parent weights
and intermediate runs are retired; the final checkpoint remains usable for export
and further fine tuning. See [GPU setup](../../docs/GPU_TRAINING.md) and
[bundle evidence](../../docs/BROWSER_BUNDLE_2026-10-17.md).

`train_detector.py` is the Ultralytics YOLO-pose trainer (`--data` from
`build_corner_dataset.py`; `--model <weights.pt>` fine-tunes a checkpoint). The earlier
OBB detector path and its synthetic comparison tools are archived.
Apple MPS cannot train this pose implementation; use the LAN CUDA machine.

Evaluate changes with `evaluate_corners_real.py` (AI-reviewed real labels) and
`compare_detectors_real.py` (retrieval-acceptance proxy, not labeled accuracy).
Real labels originate from few videos and older s-model proposals; phone results
and real-world operating thresholds remain unverified.
