# Historical material

Nothing here is part of the supported workflow. Archived code is kept for reference
and is not maintained: paths, imports and defaults describe the repository when it
was archived and may no longer resolve. Archived tests are not run by
`unittest discover -s tests`.

- `ROADMAP_2026-10-17.md`: previous roadmap, retained as a dated historical snapshot.
  Its experiment instructions and paths describe the repository before cleanup;
  use [current development priorities](../docs/DEVELOPMENT.md) instead.
- `original-card-acquisition/`: original card-fetcher notebook and saved provider
  HTML, moved without altering notebook contents. Current acquisition uses the
  sync/curation scripts under `scripts/`.
- `docs/`: dated work logs, one-off probes and research notes whose conclusions are
  folded into the current docs.

## Archived scripts (2026-10-19)

| Script | What it was for | Replaced by |
| --- | --- | --- |
| `scripts/generate_synthetic_cards.py`, `training/card_detector/synthetic_effects.py` | v1 synthetic scenes with OBB labels (One Piece only) | `scripts/generate_detection_scenes.py` + `scene_renderer.py`, then `build_corner_dataset.py` |
| `scripts/evaluate_synthetic_pipeline.py` | detector-to-retrieval check on v1 synthetic scenes | `scripts/evaluate_corners_real.py`, `scripts/compare_detectors_real.py` |
| `scripts/evaluate_detector_geometry.py` | OBB vs pose geometry on synthetic scenes (the comparison that selected pose) | `scripts/evaluate_corners_real.py` |
| `scripts/run_foundation_detector.py`, `scripts/crop_foundation_proposals.py` | GroundingDINO zero-shot localization probe | the trained corner detector |
| `scripts/benchmark_ocr_scale.py` | Tesseract whole-card OCR scale probe | (OCR remains optional; see `docs/OCR_IDENTITY.md`) |
| `scripts/fetch_pokemon_references.py` | bounded 32-card TCGdex pilot import | `scripts/sync_pokemon_gallery.py` |
| `scripts/import_real_card_images.py` | classifier-era photo import into `datasets/card_recognizer` | curated recognition manifests |
| `scripts/prepare_card_gallery.py` | canonical reference crops without embeddings | gallery embedding in `train_retrieval.py` / export |
| `scripts/inventory_recognition_crops.py`, `scripts/propose_reference_pairs.py`, `scripts/propose_onepiece_reference_pairs.py`, `scripts/propose_named_card_pairs.py` | one-off proposal tools of the 2026-10 crop-review campaign (hard-coded campaign inputs) | `scripts/propose_crop_review_batch.py` + root visual review |

The deployed detector is the YOLOv8-pose corner model; the browser OBB decoder and
OBB training/evaluation paths were removed (see Git history before 2026-10-19).

Legacy classifier/native-app sources and superseded model artifacts were removed
from the current checkout. Git retains source history; retired DVC artifacts are
not part of the supported current workflow.
