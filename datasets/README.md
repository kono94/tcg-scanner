# Datasets

Everything here is versioned with DVC (`*.dvc` pointers in Git, contents in the DVC
remote on the LAN GPU PC). `dvc pull <pointer>` restores a folder; after changing one,
run `dvc add <folder>`, `dvc push` and commit the pointer.

```text
datasets/
  curated/                 accepted dataset the trainers and the bundle read
  staging/                 reviewed candidates that were not accepted
  sources/                 acquired raw material, kept unmodified
    catalog/               official/reference card images, one folder per provider sync
    videos/                public video excerpts, mined frames and derived crops
    phone/                 captures and decisions uploaded from the dev scanner
  review/                  AI review records that produced curated/
  detection/               corner-detector labels and fixtures
  prices/prices.json       USD/EUR price snapshot copied into the bundle
  reference-thumbnails/    small reference JPEGs served by the deployed scanner
```

Data flows one way: `sources/` → proposals and review (`review/`) → `curated/` (and
`staging/` for the rejects) → training, export and the browser bundle. Nothing in
`curated/` is labeled by hand; see [AGENTS.md](../AGENTS.md) for the labeling rules.

## curated/ and staging/

Written only by `scripts/curation/curate_datasets.py` from a reference manifest and a
root-reviewed query manifest. `curated/recognition/` holds the gallery
(`gallery/{game}/{language}/{card_id}/reference-*.{ext}`, 24,197 references), the
source-isolated real query crops (`train/`, `val/`, `test/`) and the card-back references
(`backs/`). Every image has a JSON sidecar with its source record and hashes. Layout and
manifest schemas: [docs/DATASET_LAYOUT.md](../docs/DATASET_LAYOUT.md).

## sources/catalog/

| Folder | Provider | Written by |
| --- | --- | --- |
| `pokemon-en-tcgdex/` | TCGdex English catalog and images | `scripts/catalog/sync_pokemon_gallery.py` |
| `pokemon-en-pokemontcgio/` | pokemontcg.io images for TCGdex gaps, exact matches only | `scripts/catalog/fill_pokemon_gallery_gaps.py` |
| `pokemon-en-targeted/` | a few individually resolved English references | review campaign |
| `pokemon-ja-official/` | official Japanese card-search scans with TCGdex records | review campaign |
| `onepiece-en-bandai/` | official English card list | `scripts/catalog/sync_onepiece_gallery.py` |
| `onepiece-ja-bandai/` | official Japanese card list | `scripts/catalog/sync_onepiece_ja_gallery.py` |
| `onepiece-en-bandai-2024/` | the original 1,399-card scrape (2024, archived notebook); many carry SAMPLE | `archive/original-card-acquisition/` |

Each sync writes a manifest with source URLs, hashes and per-card status and resumes
from it. `scripts/curation/build_reference_manifest.py` collects these into one
reference manifest for curation.

## sources/videos/

One folder per acquisition pass (for example `binder-search-op-slow/`): the downloaded
excerpt under `raw/`, mined frames with a manifest, detector crops and review sheets.
How to add more footage: [docs/VIDEO_MINING.md](../docs/VIDEO_MINING.md). Reuse rights
of public videos are unknown; these are local inspection and training inputs.

## sources/phone/

`captures/<date>/` holds scanner captures with at least one detected card (JPEG plus
unverified detections) and `feedback/<date>/` holds per-card user corrections. They come from
the app's "Share scans to improve recognition" setting (on by default): directly from the
Vite dev server, or from the deployed scanner's upload service via
`deploy/pull-uploads.sh`. `intake/` holds inventories of a capture session. These are the next
independent real-camera evaluation inputs; they are not reviewed yet. After new uploads:
`dvc add datasets/sources/phone`, `dvc push`, commit the pointer.

## review/

Inventories, OCR/feature-match proposals, contact sheets and the merged review decisions
(`pass2/references_v5.json`, `pass2/sol_review_merged.json`) that `curated/` was built
from. History and counts: [docs/DATASET_CLEANUP.md](../docs/DATASET_CLEANUP.md).

## detection/

- `real-corners/`: AI-reviewed real-frame corner labels. `heldout_labels_v1.json` is the
  detector test set used by `scripts/detection/evaluate_corners_real.py`;
  `train_labels_v1.json` feeds `build_corner_dataset.py --real`. Frames come from
  `sources/videos/`.
- `real-detection-review/`: reviewed detector proposals on real frames (frame card counts
  for `compare_detectors_real.py`).
- `parity-scenes-v4/`: 48 synthetic validation scenes (renderer v4) used as detector
  parity fixtures by the export and bundle scripts.

Synthetic training scenes and YOLO corner datasets are regenerated on demand under
`runs/detection/` and are not versioned.
