# TCG Scanner

A phone-browser trading-card scanner for One Piece and Pokémon: capture a steady single
card or binder page, detect each card's four corners, straighten it, identify it by
similarity search with a fine-tuned DINOv2 encoder, and review the results with prices.

## Repository map

| Folder | What it is |
| --- | --- |
| [`web/`](web/README.md) | **the product**: Vite + TypeScript scanner, ONNX inference in a worker |
| [`deploy/`](deploy/README.md) | static hosting on the VPS (Caddy, release scripts, GitHub deploy hook) |
| [`training/`](training/) | model code: `recognition/` (card encoder, preprocessing contract, OCR helpers) and `detection/` (synthetic scene renderer, corner detector trainer) |
| [`scripts/`](scripts/README.md) | pipeline tools by stage: `catalog/`, `video/`, `curation/`, `detection/`, `release/` |
| [`datasets/`](datasets/README.md) | versioned data (DVC): curated dataset, raw sources, review records, detector labels, prices |
| [`models/`](models/README.md) | versioned weights (DVC): pretrained backbone, deployed encoder and detector, browser exports |
| `runs/` | local scratch for training and experiments; never versioned, safe to delete |
| [`docs/`](docs/) | design, policies and procedures (start with [DEVELOPMENT.md](docs/DEVELOPMENT.md)) |
| `tests/` | Python unit tests (`web/tests`, `web/e2e` hold the web tests) |
| `resources/` | video-source candidate lists and a video conversion helper |
| [`archive/`](archive/README.md) | retired code and dated logs, not maintained |

## Current state

Bundle `09304b039491978b`: YOLOv8s-pose corner detector `corners-s640-v4-ft` (640 px) and
DINOv2-small/ArcFace encoder `dinov2-arcface-gpu-20261017`, 24,038 English/German Pokémon and One
Piece card rows, 14,777 English Yu-Gi-Oh! artwork rows (enrolled without retraining) and seven
card-back rows. Home has a game picker; cards are only matched within the selected games. Inference runs in a worker: WebGPU with the fp16 encoder, falling
back to WASM with the int8 encoder; both use the fp16 detector. The scanner is in use on phones. It can be installed
from the browser as a home-screen app (web app manifest, no app store, no offline mode): opened
that way it runs full screen with a bottom tab bar ([details](web/README.md#navigation-back-and-installing-as-an-app)).

Automated checks run on desktop Chrome with a test-video camera mock. No controlled phone
benchmark exists, and match thresholds are uncalibrated. Encoder retrieval diagnostics
(20/24 validation, 85/86 test crops) come from few AI-reviewed videos and are not scanner
accuracy. Details: [bundle](docs/BROWSER_BUNDLE.md), [encoder training](docs/RETRIEVAL_TRAINING.md),
[recognition design](docs/RECOGNITION_DESIGN.md).

## Run and check

Python 3.12: use `.env` or install `requirements.txt`. DVC pulls from the LAN GPU PC.

```bash
.env/bin/dvc pull web/public/bundle.dvc web/public/media.dvc \
  models/browser-export/gpu-20261017.dvc models/browser-export/corner-decoder-fixture.dvc
cd web && npm ci && npm run dev        # phones need HTTPS, see deploy/README.md
npm run typecheck && npm test && npm run e2e
cd .. && .env/bin/python -m unittest discover -s tests -v
```

## Data and models

Git holds code and small `.dvc` pointers; DVC holds `datasets/`, `models/` and the web
bundle/media (remote: the LAN GPU PC; the VPS store holds only deploy artifacts). After
changing a tracked folder: `dvc add <folder>`, `dvc push`, commit the pointer.

Detector datasets are Ultralytics YOLO pose (`kpt_shape: [4, 3]`, four ordered corners);
recognition uses the JSON manifest in `datasets/curated/recognition/` with source-isolated
splits. The price snapshot `datasets/prices/prices.json` comes from
`scripts/release/update_prices.py` and is copied into each bundle.

Before changing behavior read [AGENTS.md](AGENTS.md), [development priorities](docs/DEVELOPMENT.md),
[scanner UX](docs/SCANNER_UX.md), [capture cycle](docs/CAPTURE_CYCLE.md) and the
[match/value policy](docs/MATCH_AND_VALUE_POLICY.md). To add real data from videos:
[VIDEO_MINING.md](docs/VIDEO_MINING.md).
