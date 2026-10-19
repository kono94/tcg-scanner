# Extending the real-image data from videos

Public binder tours, collection showcases, pack openings and deck profiles supply real
sleeves, glare, hands, binder pockets and backgrounds that the synthetic renderer cannot.
This page is the end-to-end procedure for adding such footage: find a source, download an
excerpt, mine frames, crop cards, propose identities and corners, review, and promote. It
needs no manual labeling by the user; every identity and corner label comes from automated
proposals plus AI visual review, and stays labeled as such.

Everything lands under `datasets/sources/videos/<pass>/` (DVC). Earlier passes there and
their search logs in [research/](research/) and `resources/benchmarks/` are worked examples.

## 1. Find and vet a source

- Search for top-down binder pages and close card presentations (titles like "binder tour",
  "collection", "master set"). Record candidates with URL, channel, game, language and
  why they fit in `resources/benchmarks/<search>.json`; write a short note in
  `docs/research/` for each search pass (see `BINDER_OP_SLOW.md`).
- **Resolution gate:** the actually available format must be at least **1440p, 2160p
  preferred**. Check the real format table, not the title:
  `.env/bin/python -m yt_dlp -F --no-playlist <url>`. Unknown resolution stays a candidate.
- **Detail gate:** a card needs readable name/number pixels, not just a 4K frame. Prefer
  clips where cards fill a large part of the frame. Burned-in card graphics, host insets
  and "SAMPLE" overlays are not physical cards.
- Reuse rights of public videos are unknown; record that, and keep the material for local
  training/evaluation only.
- Footage types differ in value: binder tours match the scanner's page layout but give
  few pixels per card; close presentations and deck profiles give readable fronts;
  pack openings give hands and motion; tournament tables (candidates in
  `resources/benchmarks/video_candidates.json`) give tiny, occluded cards and broadcast
  graphics. Quality passes of earlier sources: [One Piece](VIDEO_ACQUISITION_ONEPIECE.md),
  [Pokémon](VIDEO_ACQUISITION_POKEMON.md), [audit](VIDEO_DATA_AUDIT.md).

## 2. Download a short excerpt

Download only the useful section (seconds, not whole videos):

```bash
.env/bin/python -m yt_dlp -f '<format id from -F>' --download-sections '*300-312' \
  -o 'datasets/sources/videos/<pass>/raw/%(id)s-300-312.%(ext)s' '<url>'
```

Section cuts are not frame-accurate (keyframe preroll); record times as clip-relative.

## 3. Mine frames

```bash
.env/bin/python scripts/video/mine_video_frames.py \
  --video datasets/sources/videos/<pass>/raw/<id>-300-312.webm \
  --output datasets/sources/videos/<pass>/<id>-300-312 \
  --source-url '<url>' --source-group <channel-or-video-id> \
  --reuse-basis 'public video; reuse rights unknown' --contains-overlays unknown \
  --sample-seconds 0.5 --window-seconds 2
```

The miner refuses sources with a decoded short edge below 1440 px (do not upscale), keeps
the sharpest frame per time window, drops near-duplicates and writes a manifest with
source/frame hashes. Adjacent frames are repeated views of the same physical cards, not
new cards. The **source group** decides the split: all frames and crops of one video or
channel stay in one of train/val/test.

## 4. Crop every detected card

```bash
.env/bin/python scripts/video/extract_detected_cards.py \
  --frames-manifest datasets/sources/videos/<pass>/<id>-300-312/manifest.json \
  --detector models/corner-detector/corners-s640-v4-ft/weights/best.pt \
  --output datasets/sources/videos/<pass>/<id>-300-312/detected_cards
```

Crops are perspective-warped from source resolution with the deployed corner detector and
carry their source quad. They are unlabeled candidates.

## 5. Propose identities, then review

- `scripts/curation/propose_card_labels.py` reads card codes with Tesseract OCR and
  resolves them against the catalog sidecars; unclear codes stay unresolved.
- `scripts/curation/propose_crop_review_batch.py` matches crops to reference images with
  local features and renders contact sheets (crop next to candidate references).
- **Root review:** the reviewing agent compares every crop with the actual reference
  artwork and records accept/reject with the exact catalog ID, or leaves it unresolved.
  OCR hits, feature-match inlier counts and matching character names are proposals only;
  alternate art and reprints look alike. Do not complete partial cards.

Accepted queries need a single complete card, an exact `{game, language, card_id}` and the
source group. Review records go under `datasets/review/`; the schema is in
[DATASET_LAYOUT.md](DATASET_LAYOUT.md) and the previous campaign in
[DATASET_CLEANUP.md](DATASET_CLEANUP.md).

## 6. Real corner labels for the detector

```bash
.env/bin/python scripts/detection/real_corner_labels.py propose \
  --teachers models/corner-detector/corners-s640-v4-ft/weights/best.pt@1280 \
  --splits train --output runs/detection/real-corners-<pass>
# AI-review the numbered review tiles, write review.json (accept / ignore / reject per id)
.env/bin/python scripts/detection/real_corner_labels.py apply \
  --proposals runs/detection/real-corners-<pass>/proposals.json \
  --review runs/detection/real-corners-<pass>/review.json \
  --output datasets/detection/real-corners/train_labels_v2.json
```

Proposals are snapped to straight card edges; only reviewed complete cards become labels,
and every card in a labeled frame must be accepted or marked ignore. New sources are
train-split unless added to `TEST_SOURCES`/`VAL_SOURCES` in
`scripts/detection/review_real_detections.py`; never move an existing source between splits.

## 7. Promote and retrain

1. `scripts/curation/build_reference_manifest.py` and `scripts/curation/curate_datasets.py`
   with the merged review manifest → new `datasets/curated` snapshot.
2. `dvc add datasets/sources/videos datasets/review datasets/curated datasets/detection/real-corners`,
   `dvc push`, commit the pointers.
3. Retrain on the GPU PC ([GPU_TRAINING.md](GPU_TRAINING.md)): the encoder with the new
   curated queries, the detector with `build_corner_dataset.py --real <labels>`.
4. A new model replaces the deployed one only if it wins the held-out checks
   (`scripts/detection/evaluate_corners_real.py`, `compare_detectors_real.py`, full-catalog
   retrieval on the unchanged val/test splits).

Report counts as source videos, retained frames, physical cards and accepted crops
separately; repeated views of one card are not independent examples.
