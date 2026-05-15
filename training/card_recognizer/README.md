# Card Recognizer Training Pipeline

This folder contains the script-based One Piece exact-card recognizer pipeline.

The active approach is classifier-first:

1. keep the complete known card corpus under `datasets/card_recognizer/cards/`,
2. retrain the full classifier whenever a new card set is added,
3. evaluate the new experiment against a manually curated folder of phone images,
4. compare the new experiment against the previous baseline,
5. export one CoreML classifier package and one labels file for the iOS app.

No embedding database or similarity-search deployment path is used.

## 1. Configure The Dataset

There is no automatic holdout generation. Training always uses every scraped image under `datasets/card_recognizer/cards/`. Validation uses the single manual phone-image folder configured in YAML:

```yaml
dataset:
  cards_dir: datasets/card_recognizer/cards
  manual_eval_dir: datasets/card_recognizer/manual_eval
  eval_required: true
```

Put real phone images into folders named by card id or label:

```text
datasets/card_recognizer/manual_eval/
  OP01-001/
    IMG_1001.jpg
    IMG_1002.jpg
  OP03-056/
    IMG_2001.jpg
```

The folder name is mapped back to the scraped card labels in `datasets/card_recognizer/cards/`. You do not need examples for every card, but the evaluation score only reflects the manual examples present in this folder.

Multiple training images per class are supported in either of these formats:

```text
datasets/card_recognizer/cards/OP01/OP01-001.jpg
datasets/card_recognizer/cards/OP01/OP01-001__phone-front.jpg
datasets/card_recognizer/cards/OP01/OP01-001__phone-angle.jpg
```

or:

```text
datasets/card_recognizer/cards/OP01/OP01-001/
  official.jpg
  phone-front.jpg
  phone-angle.jpg
```

Use `__` for sample suffixes. Single-underscore IDs like `OP01-001_p1` remain separate classes, which is important for parallel art cards.

## 2. Train

```bash
python training/card_recognizer/train_lightning.py \
  --config training/card_recognizer/configs/mobilenetv3_classifier.yaml
```

By default the experiment ID is the current Unix timestamp in seconds. You can make it explicit:

```bash
python training/card_recognizer/train_lightning.py \
  --config training/card_recognizer/configs/mobilenetv3_classifier.yaml \
  --experiment-id 1778760000
```

Every training artifact for that run is local to one folder:

```text
runs/card_recognizer/experiments/<experiment_id>/
  artifacts/
  checkpoints/
  dataset/
    names.txt
    manifest.csv
    manifest.json
  logs/
  reports/
  experiment.json
  resolved_config.yaml
```

Training and evaluation scripts print stage logs and progress bars for long loops, so a quiet terminal usually means the current subprocess is waiting on framework-level work such as CoreML conversion or model initialization.

The old trainer froze the MobileNet backbone before fine tuning it. That behavior is available in Lightning through the YAML config:

```yaml
model:
  freeze_backbone_epochs: 30
```

Keep the default `0` for the current single-stage setup, where the backbone trains from the start with the lower `optimizer.backbone_lr_multiplier` learning rate.

## 3. Evaluate Classifier

```bash
python training/card_recognizer/evaluate_checkpoint.py \
  --experiment-id <experiment_id> \
  --dataset val
```

This reads `resolved_config.yaml`, uses the experiment's best checkpoint by default, and writes to `reports/eval/` inside the same experiment folder.

The report includes top-1/top-5 accuracy, per-set accuracy, NLL, Brier score, ECE, and a confusion matrix CSV.

## 4. Compare Experiments

After adding a new set and retraining, compare the candidate experiment against the previous baseline:

```bash
python training/card_recognizer/regression_report.py \
  --old-experiment-id <baseline_experiment_id> \
  --new-experiment-id <candidate_experiment_id> \
  --dataset val
```

The regression report is written into the candidate experiment's `reports/` folder.

## 5. Latency Benchmark

```bash
python training/card_recognizer/benchmark_inference.py \
  --experiment-id <experiment_id> \
  --coreml-model tcg-scanner-app/tcg-scanner-app/Models/card_recognizer.mlpackage
```

The report is written to `reports/latency_report.json`. The Python CoreML benchmark is host-side only. Physical iPhone timing still needs the app/device debug metrics.

## 6. CoreML Export

The classifier export accepts a Lightning checkpoint and performs a PyTorch smoke test plus a best-effort CoreML prediction smoke test.

```bash
python scripts/export_recognizer_coreml.py \
  --experiment-id <experiment_id>
```

By default this writes the export under `app_export/` inside the experiment folder. To install directly into the app bundle and stamp the app manifest with the recognizer experiment ID:

```bash
python scripts/export_recognizer_coreml.py \
  --experiment-id <experiment_id> \
  --install-to-app
```
