# Original One Piece reference corpus

`cards/<set>/<card-id>.jpg` plus a `.json` sidecar (id, name, original image URL) for
1,399 English One Piece cards, scraped by the archived
[card-fetcher notebook](../../archive/original-card-acquisition/). Stored in Git LFS.
Many images carry the provider's SAMPLE watermark; they are kept unmodified.

This is a provenance source, not a training layout. The scanner and trainers read
`datasets/curated/` (DVC), which already contains copies of these references under
`recognition/gallery/onepiece/en/`. The corpus is read only when the curated
gallery is rebuilt:

- `scripts/build_reference_manifest.py` lists these files as the `Bandai legacy corpus`
  provider, with hashes, before newer acquired references;
- `scripts/sync_onepiece_gallery.py --local-cards` marks cards already present here as
  `present_local` instead of downloading them again.

The classifier-era conventions (one class per folder, `__` extra samples,
`manual_eval/`, `real_raw/` imports) are retired. Do not add camera photos here:
real queries go through the video-mining and curation flow in
[DATASET_LAYOUT.md](../../docs/DATASET_LAYOUT.md).
