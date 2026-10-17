# YOLO OBB baseline

`train_obb.py` runs Ultralytics YOLOv8 OBB training with the dataset YAML and OBB labels emitted by the synthetic scene generator. The OBB labels contain a class id and four normalized point pairs. The separate scene manifest retains the original quadrilateral coordinates for geometry evaluation.

From the repository root, generate scenes, prepare canonical reference crops for the retrieval gallery, then train the detector:

```bash
.env/bin/python scripts/generate_synthetic_cards.py --output runs/card_detector/synthetic --scenes 200 --seed 42 --width 832 --height 624 --max-cards 14 --video-seconds 16
.env/bin/python scripts/prepare_card_gallery.py --cards-dir datasets/card_recognizer/cards --output runs/card_recognizer/gallery_prepared
.env/bin/python training/card_detector/train_obb.py --data runs/card_detector/synthetic/dataset.yaml --from-scratch --epochs 20 --imgsz 320 --batch 8 --device cpu --name synthetic-bootstrap
```

For a CPU smoke run, use a small epoch count and batch:

```bash
.env/bin/python training/card_detector/train_obb.py --data runs/card_detector/synthetic/dataset.yaml --from-scratch --epochs 1 --imgsz 320 --batch 2 --workers 0 --device cpu --name smoke
```

The default model, `yolov8n-obb.pt`, uses pretrained weights and may download them on first use. `--from-scratch` selects the matching `yolov8n-obb.yaml` architecture and does not fetch pretrained weights. Training output uses Ultralytics' `exist_ok=False`; choose a new `--name` for each run. Configuration and Matplotlib state are stored under `runs/card_detector/` in this repository.

The tested environment is Python 3.12 with the versions in `requirements.txt`. New environments can install those dependencies; this checkout already has them in `.env`. Scene/gallery output directories must be empty or new. Use a new output path and training name when repeating these commands.

Synthetic scenes bootstrap training and supply exact geometry without manual labels. The generator varies card count from zero through `--max-cards`, including nine-card pages, duplicates, binder/desk/plain backgrounds, rotation, perspective, illumination, glare, blur, noise and JPEG effects. Its split groups both source identity and exact source-image hashes; visually similar but byte-different references can remain across splits. Canonical query crops use renderer corners; they are not predicted-detector crops. Replay inputs are clean images with separate per-frame annotations; motion is simulated translation followed by a hold, with cuts between scenes.

A smoke run establishes that the pipeline runs. Synthetic validation metrics measure the generated domain; independent real captures are needed to claim phone-camera accuracy. Automated ingestion/teacher proposals can help build that evaluation without requiring the user to label a dataset before implementation starts. Keep proposals distinct from verified labels.
