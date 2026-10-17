# TCG Scanner

A trading-card scanner that runs in a phone browser: open a URL, hold a single card or a
binder page steady, and the page detects every card in one captured image, straightens
each one, identifies the exact card by similarity search against a reference gallery and
shows a value estimate. All inference runs on the device; the server only delivers files.

## Current status (2026-10-17)

- **Browser scanner (pilot)** in [`web/`](web/README.md): stable-frame capture →
  YOLOv8s-pose four-corner detector → perspective warp → DINOv2-small/ArcFace embedding →
  exact cosine search over 24,038 English/German One Piece and Pokémon gallery rows plus
  card-back rows → match/value policy → per-card review. Single-page and batch (review at
  the end) workflows, local collection, bundled price snapshot. Inference runs in a worker
  with onnxruntime-web (WebGPU, falling back to WASM with an int8 encoder).
- **Verified:** desktop Chrome with a test-video camera mock (unit, parity and Playwright
  tests). **Not verified:** physical phones, real-camera accuracy, calibrated thresholds.
  The detector has synthetic pretraining and AI-reviewed real-corner fine tuning.
- **Encoder:** `dinov2-arcface-gpu-20261017`, trained on the LAN GPU machine
  ([GPU training](docs/GPU_TRAINING.md), [retrieval training](docs/RETRIEVAL_TRAINING.md)).
  On the full 24,197-reference catalog it ranks the correct exact ID first for 20/24
  validation and 85/86 test crops. These are AI-reviewed crops from few videos with
  repeated views; they are not scanner accuracy.
- **Hosting:** static site behind a shared Caddy HTTPS proxy on a VPS
  ([deploy/README.md](deploy/README.md)). The deploy flow is tested against Docker on the
  LAN; the real VPS is not yet.

Design documents: [scanner UX](docs/SCANNER_UX.md), [capture cycle](docs/CAPTURE_CYCLE.md),
[match and value policy](docs/MATCH_AND_VALUE_POLICY.md),
[recognition design](docs/RECOGNITION_DESIGN.md), [data layout](docs/DATASET_LAYOUT.md).
Read the [roadmap](ROADMAP.md) and the [agent guide](AGENTS.md) before new work.

## Quick start

```bash
pip install "dvc[ssh]"                       # into .env
dvc pull web/public/bundle web/public/media runs/browser_export/pilot-20261016
cd web && npm install && npm run dev         # http://localhost:5173
```

Choose **Use camera** or a **Test video**. Phones need HTTPS for camera access; see
[web/README.md](web/README.md) for the app, its checks and how the bundle is built.

## Data and models (DVC)

Large data is versioned with [DVC](https://dvc.org): git holds small `*.dvc` pointer files;
the content lives in a DVC store on the LAN GPU PC (`/mnt/d/tcg-scanner-dvc`, reached over
SSH; see `.dvc/config`). Git LFS still holds the older files: the original One Piece card
images under `datasets/card_recognizer/`, the detector annotation archive and a demo video.

| DVC-tracked path | Size | Contents |
| --- | --- | --- |
| `datasets/curated` | 3.0 GB | accepted gallery references, labeled query crops, card backs, detection data |
| `datasets/staging` | 50 MB | acquisition candidates not yet accepted |
| `runs/card_recognizer/retrieval/dinov2-arcface-gpu-20261017` | 377 MB | current card-embedding run (checkpoint, gallery, evaluations) |
| `runs/card_detector/pretrained` | 180 MB | earlier detector weights, including `corners-n-gpu.pt` |
| `runs/browser_export/{gpu-20261017,pilot-20261016,pose-fixture}` | 350 MB | ONNX exports and the parity fixtures the web unit tests read |
| `runs/prices/prices.json` | 5 MB | latest price snapshot (with exchange rate) |
| `web/public/bundle`, `web/public/media` | 240 MB | the built browser bundle; local test videos |

```bash
dvc pull                               # everything (~4 GB), or name paths:
dvc pull web/public/bundle runs/browser_export/pilot-20261016
# after changing tracked data:
dvc add web/public/bundle && dvc push && git add web/public/bundle.dvc
```

Everything else under `runs/` is local experiment output and stays out of git and DVC.
On the GPU PC itself, point the remote at the local folder once:
`dvc remote modify --local gpu-pc url /mnt/d/tcg-scanner-dvc`. The store is a single copy on
one disk: back it up (or add a second remote) if the data cannot be rebuilt.

To clone without downloading the LFS card images:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kono94/tcg-scanner.git
```

## Python tooling

Use the repository's `.env/bin/python` (PyTorch, OpenCV, Ultralytics, onnxruntime);
`pip install -r requirements.txt` for a new environment. Training runs on the LAN GPU
machine; the Mac prepares data, evaluates, exports and builds the bundle.

| Stage | Entry points | Docs |
| --- | --- | --- |
| Reference acquisition | `scripts/sync_onepiece_gallery.py`, `scripts/sync_pokemon_gallery.py`, `scripts/fill_pokemon_gallery_gaps.py`, `scripts/acquire_card_backs.py` | [image sources](docs/IMAGE_SOURCES.md), [cleanup](docs/DATASET_CLEANUP.md) |
| Real query crops | `scripts/mine_video_frames.py`, `scripts/review_real_detections.py` | [video data audit](docs/VIDEO_DATA_AUDIT.md), [video mining](docs/VIDEO_MINING.md) |
| Synthetic scenes and detector | `scripts/generate_detection_scenes.py`, `scripts/build_corner_dataset.py`, `training/card_detector/train_obb.py` | [detector tooling](training/card_detector/README.md) |
| Card embedding | `training/card_recognizer/train_retrieval.py` | [retrieval training](docs/RETRIEVAL_TRAINING.md) |
| Evaluation | `scripts/evaluate_detector_geometry.py`, `scripts/compare_detectors_real.py`, `scripts/evaluate_card_backs.py` | |
| Browser export | `scripts/export_browser_models.py`, `scripts/build_browser_bundle.py`, `scripts/update_prices.py` | [web/README.md](web/README.md) |

```bash
.env/bin/python -m unittest discover -s tests -v
```

Synthetic and AI-reviewed results are reported separately from real-camera accuracy, which
has not been measured yet. The older MobileNet classifier, the OCR identity baseline and the
frozen-encoder comparisons are kept as historical baselines
([training/card_recognizer/README.md](training/card_recognizer/README.md),
[OCR identity](docs/OCR_IDENTITY.md), [foundation models](docs/FOUNDATION_MODELS.md)).
