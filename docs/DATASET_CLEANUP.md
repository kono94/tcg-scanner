# Dataset cleanup snapshot — 2026-10-16

The accepted snapshot is materialized at `datasets/curated/`. Root SOL has now
visually reviewed **all 519 candidates left after the first pass**, compared
usable faces with actual catalog scans, and reviewed the final source-frame
rectifications. No candidate in this review pool remains unreviewed.

| Data | Count | Location |
| --- | ---: | --- |
| English One Piece references | 4,373 | `recognition/gallery/onepiece/en/` |
| Physical English Pokémon references | 19,665 | `recognition/gallery/pokemon/en/` |
| Japanese One Piece references | 154 | `recognition/gallery/onepiece/ja/` |
| Japanese Pokémon references | 5 | `recognition/gallery/pokemon/ja/` |
| Accepted real single-card queries | **298 / 158 catalog-art identities** | `recognition/{train,val,test}/` |
| Bad derived crops removed, both passes | **304** | `rejected_crops.json` |
| Identity reviewed; usable full-card geometry unavailable | 45 | `datasets/staging/recognition/` |
| Exact identity/printing unresolved after review | 69 | `datasets/staging/recognition/` |
| Unreviewed in this pool | **0** | `recognition/review_audit.json` |

The 716 saved proposals from the expanded video acquisition partition into
**298 accepted queries + 304 quality rejects + 45 geometry exclusions + 69
unresolved identities**. The second pass adds 262 queries and removes 143 bad
crops. Gallery references are separate from these counts. Original full video
frames and catalog assets remain intact. Accepted staging links are retired;
114 explicitly reviewed exclusions remain outside training.

## Verification and labels

Root compared actual source faces, reference artwork and visible text/codes,
then inspected all 265 final warp proposals. Three more faces failed because
fingers obscured the card. Machine feature scores and observed names were
proposals throughout, never automatic approval. Incorrect character names,
Japanese/English mismatches, delta-species names and wrong alternate artwork
were corrected or excluded. Large reference comparisons also revoked tentative
Luffy matches that initially looked plausible in a thumbnail.

Japanese OP01 scans and five matching Japanese Pokémon fronts were acquired
with catalog provenance. Wailord uses its matching SV9-103 alternate art;
Professor's Research S8b-266 depicts Professor Juniper. No generated SAMPLE
restoration was enrolled. The gallery now contains **24,197 references**;
additional acquisition outputs remain staging until inventoried and admitted.

The 69 unresolved views include DON/resource cards, generic energies, shared-art
reprints without readable distinguishing details, and artwork not confidently
matched to the acquired catalog. An uncertain printing is not forced into an
exact catalog suffix. The 45 geometry exclusions retain the reviewed identity
and reason but do not enter encoder training as raw multi-card/partial crops.

Every accepted query records source-frame and original-crop paths/hashes,
source group, reference path/hash, explicit game/language/catalog identity,
root review provenance and rectification geometry. New crops use original
source pixels, the shared 224:312 full-card aspect, source-edge-limited output
sizes and a maximum 1,024-pixel edge. Missing edges are not synthesized.
Reference-supported quadrilaterals are annotation proposals, not independent
detector corner ground truth. Moderate real sleeve glare, blur and color cast
remain in usable crops and are recorded as capture conditions.

These are **AI-reviewed catalog-art labels**, not human-certified annotations
or a guarantee of every physical edition/finish. Printing and finish stay
unknown. No recognition accuracy or new model training is claimed here.

## Splits and repeated views

| Source group | Split | Queries |
| --- | --- | ---: |
| `squaaaaw_collection_tour` | train | 103 |
| `JINstudio_Japanese_collection_reveal` | train | 47 |
| `fG9cnN5AH6Q` | train | 35 |
| `ho7IFpPXI-A` | train | 3 |
| `evans-collection-community-binder-submissions-2026` | val | 19 |
| `pokemon-gen5-binder-pokeduell` | val | 5 |
| `4QxpRG4E6d0` | test | 82 |
| `marinefordtcg_op12_two_boxes` | test | 4 |

Totals: **188 train / 24 validation / 86 test**. The queries contain 192 English
One Piece, 47 Japanese One Piece, 54 English Pokémon and five Japanese Pokémon
views. They represent 158 catalog-art identities, not 298 distinct physical
cards. Repeated views from a video remain in one split; no video crosses splits.
The held-out sets are small diagnostics. This split alone is not an
unseen-identity experiment: enrollment references remain available, and an
encoder training configuration must separately hold out test identities if it
claims new-card generalization.

The gallery's exact-byte cross-identity alias remains One Piece `OP05-076_p2`
versus `OP05-076_p4`. Namespaces are preserved in
`recognition/exact_reference_aliases.json`; treat identical visible designs as
a visual family when constructing negatives.

## Detector data

The snapshot no longer contains detector data. Its 200 v1 synthetic OBB scenes
(One Piece only, superseded by the renderer-v2/v3 scene sets) were removed on
2026-10-19. The deployed corner detector trains on `generate_detection_scenes.py`
scenes plus AI-reviewed real corners via `build_corner_dataset.py` (see
`training/card_detector/README.md`).

## Checks and reproduction

Curation decodes and verifies SHA-256 for every enrolled reference and accepted
query/source/reference link, enforces source-group isolation and rejects
cross-split duplicate query bytes or evaluation/gallery byte overlap. Rejected
crop paths, hashes and preserved source frames are checked before deletion;
accepted outputs exist first. Images are hardlinked and immutable: transforms
must write new files.

The Python suite passes **100 tests**, including cached-feature source-coordinate
mapping, safe malformed-descriptor rejection and preview-versus-strict geometry
policy. The detector YAML and annotation paths are checked with the installed
Ultralytics loader. These checks do not measure camera recognition accuracy.

Current local source manifests:

- `runs/dataset_cleanup/pass2/references_v5.json`
- `runs/dataset_cleanup/pass2/sol_review_merged.json` (all accepted queries)
- `runs/dataset_cleanup/pass2/sol_review_complete.json` (all 519 decisions)
- `datasets/curated/recognition/review_audit.json`
- `datasets/staging/recognition/manifest.json` (reviewed exclusions only)

Build a new empty snapshot from the approved inputs:

```bash
.env/bin/python scripts/curate_datasets.py \
  --input-root . \
  --references-manifest runs/dataset_cleanup/pass2/references_v5.json \
  --review-manifest runs/dataset_cleanup/pass2/sol_review_merged.json \
  --output datasets/curated-next --copy-mode hardlink
```

Historical proposal manifests intentionally retain paths to removed rejected
files for audit. Consult the rejection audit and current staging manifest for
file availability. Next recognition work is pretrained-backbone retrieval fine
tuning with identity/visual-family-aware splits and negatives; frozen-DINO
comparisons remain stopped.
