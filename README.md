# TCG Scanner

## Current status and next direction

The implemented phone scanner is the SwiftUI iOS app with its original One Piece classifier. New offline tools generate synthetic scenes, train YOLO OBB, prepare gallery images, compare frozen-encoder similarity search, and test OCR/catalog label proposals. The intended recognition path uses similarity normally, with an optional OCR fast pass for exceptionally clear crops. These experimental tools have not replaced the app models. A shareable browser client remains planned.

The first implementation must support both single cards and multi-card binder pages, including nine-card pages, with a clean instruction/progress overlay. See the [scanner interaction specification](docs/SCANNER_UX.md).

Card count is variable: nine-card pages are a benchmark, not a required input or maximum. Collect scenes with any count (including zero); annotate actual instances, then derive individual recognition crops. See the [data collection contract](docs/DATA_COLLECTION.md).

The planned default is [stable-frame scanning](docs/CAPTURE_CYCLE.md): wait until steady, capture once, then detect and recognize all cards on that image. Continuous card tracking is optional; manual capture and configurable repeat-while-steady cycles are also specified.

The [dataset cleanup](docs/DATASET_CLEANUP.md) separates acquired gallery references, reviewed single-card recognition queries, and multi-card OBB scenes in a [uniform layout](docs/DATASET_LAYOUT.md). The accepted data contains 24,197 gallery references, 298 SOL-reviewed real queries across 158 catalog-art identities and 200 synthetic detector scenes. All 519 remaining candidates have been reviewed; 114 geometry/identity exclusions remain outside training. The first Apple-GPU DINOv2-small/ArcFace fine-tuning pilot is launched; see [training status and reproduction](docs/RETRIEVAL_TRAINING.md). Further frozen-DINO comparisons are stopped.

Read the [repository audit](docs/REPO_AUDIT_2026-10-15.md), [roadmap](ROADMAP.md), and [agent guide](AGENTS.md) before new work. The current ten `manual_eval` images duplicate training references and do not measure phone-camera accuracy. Some local source checkpoints and the detector image archive are missing; see the audit before attempting reproduction.

## Setup

The new automatic bootstrap needs no manual labels. Using the existing Python environment:

```bash
.env/bin/python scripts/generate_synthetic_cards.py --scenes 200 --max-cards 14 --video-seconds 16
.env/bin/python scripts/prepare_card_gallery.py
.env/bin/python training/card_detector/train_obb.py --from-scratch --epochs 20 --imgsz 320 --batch 8 --device cpu
.env/bin/python -m unittest discover -s tests -v
```

Generated scenes, OBB labels, exact corner/identity annotations, recognition crops, replay video and gallery crops are saved under ignored `runs/` directories. Output directories must be empty/new. The gallery preparation step creates canonical images and metadata, not embeddings yet. `--from-scratch` avoids pretrained-weight downloads; normal training can use the supported YOLOv8 OBB weights. See [detector tooling](training/card_detector/README.md). Synthetic validation is separate from real-camera accuracy.

For the next stages, see [optional OCR identity evidence](docs/OCR_IDENTITY.md), [pretrained model experiments](docs/FOUNDATION_MODELS.md), [4K video mining](docs/VIDEO_MINING.md), and [cross-game scope](docs/CROSS_GAME_RESEARCH.md). `scripts/sync_pokemon_gallery.py` acquires provider catalog IDs and physical-reference images resumably; `scripts/build_embedding_gallery.py` builds their descriptors after canonical preparation. Acquisition/proposal artifacts are not automatically promoted into exact-label training or independent validation.

Clone the repository without automatically downloading all Git LFS files.

```bash
sudo apt install git-lfs # if not already installed
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kono94/tcg-scanner.git
cd tcg-scanner
```

Pull only the app model artifacts needed for iOS builds:

```bash
git lfs pull --include "tcg-scanner-app/tcg-scanner-app/Models/**"
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Data and models (DVC)

Large data is versioned with [DVC](https://dvc.org): git holds small `*.dvc` pointer files;
the content lives in a DVC store on the LAN GPU PC (`/mnt/d/tcg-scanner-dvc`, reached over
SSH; see `.dvc/config`). Git LFS stays in use for the older card images and the iOS models.

| DVC-tracked path | Size | Contents |
| --- | --- | --- |
| `datasets/curated` | 3.0 GB | accepted gallery references, labeled query crops, detection data |
| `datasets/staging` | 50 MB | acquisition candidates not yet accepted |
| `runs/card_recognizer/retrieval/dinov2-arcface-gpu-20261017` | 377 MB | current card-embedding run (checkpoint, gallery, evaluations) |
| `runs/card_detector/pretrained` | 180 MB | detector weights, including the bundled `corners-n-gpu.pt` |
| `runs/browser_export/{gpu-20261017,pilot-20261016,pose-fixture}` | 350 MB | ONNX exports and the parity fixtures the web unit tests read |
| `runs/prices/prices.json` | 5 MB | latest price snapshot (with exchange rate) |
| `web/public/bundle`, `web/public/media` | 240 MB | the built browser bundle; local test videos |

```bash
pip install "dvc[ssh]"                 # in .env
dvc pull                               # everything (~4 GB), or name paths:
dvc pull web/public/bundle runs/browser_export/pilot-20261016
# after changing tracked data:
dvc add web/public/bundle && dvc push && git add web/public/bundle.dvc
```

Everything else under `runs/` is local experiment output and stays out of git and DVC.
On the GPU PC itself, point the remote at the local folder once:
`dvc remote modify --local gpu-pc url /mnt/d/tcg-scanner-dvc`. The store is a single copy on
one disk: back it up (or add a second remote) if the data cannot be rebuilt.

Hosting the browser scanner on a VPS: [deploy/README.md](deploy/README.md).

## Usage

Launch the guided top-level CLI:

```bash
python tcg.py
```

It walks through the main workflows: training a One Piece classifier, adding a new set and retraining, evaluating reports, exporting the app model, refreshing app resources, and running iOS build/test commands.
The CLI prints each subprocess command before it starts, streams progress/log output, reports success or failure with elapsed time, and pauses before returning to the main menu.

Refresh the bundled app price snapshot:

```bash
python scripts/scrape_prices.py
```

Build the iOS app:

```bash
xcodebuild -project tcg-scanner-app/tcg-scanner-app.xcodeproj -scheme tcg-scanner-app -destination 'generic/platform=iOS' build
```

Export a recognizer CoreML package after training:

```bash
python scripts/export_recognizer_coreml.py --experiment-id <experiment_id>
```

## Recognizer Training

The notebook prototype has a script-based training/evaluation path under `training/card_recognizer/`.

Train with PyTorch Lightning and TorchMetrics:

```bash
python training/card_recognizer/train_lightning.py \
  --config training/card_recognizer/configs/mobilenetv3_classifier.yaml
```

The command trains on every scraped card image under `datasets/card_recognizer/cards/`, evaluates against the manual phone-image folder from the YAML config, prints an experiment ID, and writes all run-local artifacts under:

```text
runs/card_recognizer/experiments/<experiment_id>/
```

Evaluate the classifier and compare experiments by experiment ID:

```bash
python training/card_recognizer/evaluate_checkpoint.py \
  --experiment-id <experiment_id>

python training/card_recognizer/regression_report.py \
  --old-experiment-id <baseline_experiment_id> \
  --new-experiment-id <candidate_experiment_id>
```

See `training/card_recognizer/README.md` for the full workflow and caveats about the current one-image-per-card corpus.

https://github.com/user-attachments/assets/992716b5-3d6d-4835-84aa-8eb74fbf4293
