# Scripts

Command-line tools grouped by pipeline stage. Run them from the repository root with the
project environment, for example `.env/bin/python scripts/release/update_prices.py --help`.

| Stage | Script | Does |
| --- | --- | --- |
| `catalog/` | `sync_pokemon_gallery.py` | resumable TCGdex English Pokémon catalog + images → `datasets/sources/catalog/pokemon-en-tcgdex` |
| | `fill_pokemon_gallery_gaps.py` | exact-match pokemontcg.io images for TCGdex gaps |
| | `sync_onepiece_gallery.py`, `sync_onepiece_ja_gallery.py` | official Bandai English/Japanese card lists |
| | `acquire_card_backs.py` | pinned card-back references → `datasets/curated/recognition/backs` |
| `video/` | `mine_video_frames.py` | sharp, diverse frames from a local video excerpt |
| | `collect_video_frames.py` | validates mined-frame manifests into localization inputs |
| | `extract_detected_cards.py` | perspective crops of each detected card (corner detector) |
| `curation/` | `propose_card_labels.py` | OCR/catalog identity proposals (Tesseract) |
| | `propose_crop_review_batch.py`, `verify_card_geometry.py` | local-feature reference proposals and contact sheets for visual review |
| | `build_reference_manifest.py` | one hashed reference manifest from the catalog sources |
| | `curate_datasets.py` | copies reviewed references/queries into `datasets/curated` |
| `detection/` | `generate_detection_scenes.py` | synthetic scenes with exact corners (renderer v4 by default) |
| | `real_corner_labels.py` | real-frame corner proposals, review sheets and labels |
| | `build_corner_dataset.py` | YOLO-pose dataset from synthetic scenes + reviewed real corners |
| | `evaluate_corners_real.py`, `compare_detectors_real.py`, `review_real_detections.py` | detector evaluation on real frames |
| | `make_scene_replay.py` | synthetic camera-mock video for the web e2e tests |
| `release/` | `export_browser_models.py` | ONNX export + parity fixtures → `models/browser-export/` |
| | `build_browser_bundle.py` | content-versioned bundle → `web/public/bundle` |
| | `evaluate_card_backs.py` | card-back rows against real fronts and held-out backs |
| | `update_prices.py` | USD/EUR price snapshot → `datasets/prices/prices.json` |
| | `build_ref_thumbnails.py` | deploy thumbnails → `datasets/reference-thumbnails` |
| `maintenance/` | `relayout_2026_10.py` | one-off move of versioned data out of `runs/` (also for the GPU PC mirror) |

How the stages fit together: [datasets/README.md](../datasets/README.md),
[docs/VIDEO_MINING.md](../docs/VIDEO_MINING.md) and [docs/GPU_TRAINING.md](../docs/GPU_TRAINING.md).
