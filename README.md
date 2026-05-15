# TCG Scanner

## Setup

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
