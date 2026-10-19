# Mining real card footage

Updated 2026-10-15. Tournament recordings, overhead deck profiles, pack openings and binder tours are useful acquisition targets. They supplement the synthetic renderer with real sleeves, reflection, hands, motion, compression and backgrounds. Their usefulness for exact recognition depends on actual pixels per card and identity evidence; a video title alone does not establish either.

## Located tournament recordings

- [TCG Day 1 — 2025 Pokémon World Championships, Play Pokémon](https://www.youtube.com/watch?v=Te-rE0Zifec): official TCG-specific coverage, published 2025-08-15. Prefer this to a mixed VGC/GO/TCG broadcast when searching for table footage.
- [TCG Day 1 — 2025 Pokémon World Championships, Pokémon](https://www.youtube.com/watch?v=NjNnJcK-JXg): another official upload. Treat possible duplicate broadcasts/reuploads as the same source group until checked; they are not automatically additional independent footage.
- [Championship 26–27 Finals, Utrecht Top Cut — Play Bandai Card Games](https://www.youtube.com/watch?v=qgwRa20C3lE): official One Piece tournament broadcast, published 2026-09-06.

These tournament URLs were found through public titles/descriptions and have not been downloaded. Separate collection/binder/pack sources below have now been inspected. Machine-readable tournament candidates remain in `resources/benchmarks/video_candidates.json`.

## Actual 4K acquisition and quality gate

Luna source manifests now include positively listed native 3840×2160 formats for four long-form One Piece sources and two Pokémon sources, plus two portrait One Piece shorts. See `onepiece_video_candidates.json` and `pokemon_video_candidates.json` under `resources/benchmarks/` for actual metadata evidence; unknown/missing format listings are not a 4K claim.

Short excerpts from One Piece collection video `igoDuKsMfJo`, Pokémon binder video `ZQ11RKIA7BE` and Pokémon pack video `BF_gXDPnhCg` have been downloaded/inspected and mined into **eight 3840×2160 frames per source**. Local artifact roots are `runs/video_mining/luna-onepiece/` and `runs/video_mining/luna-pokemon/`, with acquisition notes, source hashes, clip-relative timing caveats and candidate manifests. Reuse status remains unresolved; these are inspection candidates, not automatically training-promoted media.

The miner enforces a decoded source **short edge of at least 1440**, with 2160 preferred; this handles portrait footage as well. Lower the gate only for explicitly marked local test fixtures. Do not upscale 1080p into eligibility. The inspected Bob's Japan 1080p excerpt is retained as an acquisition-failure example. Also check actual card-pixel detail: the inspected 4K material still contains tiny/blurred physical cards, digital card overlays, packaging and captions.

Codec keyframes are compression reference frames, not a sharpness guarantee. Select sharp, low-motion and diverse moments across shots, then assess card crops; optionally use decoder keyframes for efficient seeking. Keep frame/event independence and avoid counting thousands of adjacent frames as thousands of new cards.

## Prioritize different footage for different tasks

| Footage | Expected value to test | Main limitation |
| --- | --- | --- |
| Binder tours / collection page flips | Closest to the scanner's page layout, pockets, sleeves, variable counts, page transitions and glare | Camera movement, low detail on whole pages, uncertain identities |
| Overhead deck profiles / close-up card reviews | Readable individual fronts, multiple views, broader clean-reference candidates | Not representative of full-page acquisition; printed number can still be obscured |
| Pack openings / sorting sessions | Card motion, changing angles, fingers, real backgrounds and clear-view anchors | Many occluded views and duplicate common cards |
| Tournament table footage | Generic detection, overlap, rotation, hands and hard recognition queries | Narrow competitive-card coverage; cards can be tiny; backs, counters, broadcast graphics and digital inserts |

These are acquisition hypotheses, not inspected quality ratings for the linked VODs. Keep physical table cards separate from broadcast card-preview graphics. A deck list constrains possible identities but does not identify every visible printing or give physical corners.

## Implemented frame miner

`scripts/mine_video_frames.py` reads a local constant-frame-rate video and:

1. Samples frames sequentially at a configurable interval.
2. Measures thumbnail sharpness and whole-frame change, retaining one best-scoring candidate per time window.
3. Suppresses visually near-identical candidates against retained thumbnails.
4. Saves full-resolution JPEG frames and a manifest with source/frame hashes, timestamps, source group, scores and explicit unlabeled identity/geometry status.

It is bounded by sample/frame limits, preserves the input and refuses nonempty output directories. It neither downloads videos nor detects/labels cards. `scripts/extract_detected_cards.py` adds source-resolution OBB crop extraction; `scripts/propose_card_labels.py` handles OCR-first catalog proposals with optional visual fallback. The first real crop batches were quarantined because the synthetic-trained detector selected subregions/graphics rather than reliable full-card boundaries. Sharpness/motion are whole-frame heuristics, not proof of card legibility. Variable-frame-rate sources need presentation-timestamp handling or a documented conversion first; this version uses frame index / reported FPS.

```bash
.env/bin/python scripts/mine_video_frames.py \
  --video /path/to/source.mp4 \
  --output runs/video_mining/match-001 \
  --source-url 'https://www.youtube.com/watch?v=Te-rE0Zifec' \
  --source-group pokemon-worlds-2025-broadcast-a \
  --reuse-basis 'record-the-actual-source-permission-or-license' \
  --contains-overlays yes \
  --start 600 --end 1200 --sample-seconds 0.5 --window-seconds 2
```

Split long recordings into distributed match/time chunks, with quotas for different scenes. The frame cap stops extraction early; one capped run over an eight-hour VOD would not provide balanced coverage. Keep all chunks and crops from the same source group together in dataset splitting.

Smoke verification on the (since replaced) v1 synthetic replay video sampled 32 frames and retained four diverse candidates, rejecting four duplicate window candidates. The output is `runs/card_recognizer/mined-synthetic-replay/manifest.json`; this is a tool check, not newly acquired real data. Two dedicated tests cover source/provenance retention, temporal selection/deduplication and overwrite/range safety.

## Automatic labeling and useful experiments next

- Filter candidate frames into physical-card views, broadcast graphics and irrelevant scenes. Run a stronger generic teacher and the new OBB baseline; preserve disagreement examples rather than retaining only easy teacher successes. Proposed boxes/corners remain pseudo-labels.
- Extract crops from source resolution and record available card pixels, blur, glare and occlusion. Do not use super-resolution/inpainting as evidence that a tiny card's exact variant was recovered.
- Find high-quality anchor views using retrieval plus visible card-number/text/variant evidence. Deck lists and titles are weak constraints; one source alone must not silently assert an exact alternate-art/printing label. Return unknown/ambiguous where detail is insufficient.
- Associate an anchor with nearby views of the same physical card, using short-range geometry/motion consistency and rejecting identity propagation across cuts, hands, swaps and ambiguous overlaps. This is offline data generation; it does not impose tracking on the browser's default snapshot scanner.
- Use same-card clear/glare/tilted pairs as real positive views. Preserve separate physical copies and partial/unknown cards. Do not declare unrelated nearby frames independent positives just because a teacher repeats the same label.
- Mine hard negatives from sleeves, card backs, deck boxes, playmat artwork, counters and empty pockets. Real backgrounds can inform procedural scene styles; known-corner composites still require controlled geometry or independently trustworthy masks.
- Compare synthetic-only versus synthetic plus video-derived training on independent held-out captures. Split by original match/session/creator and resolve reuploads/near duplicates before extracting crops. Keep teacher confidence and provenance; self-consistent pseudo-labels are not verified test truth.

The immediate implementation after frame selection is actual card filtering, OBB crops and retrieval proposals. No tournament-derived identity labels or model improvements are claimed yet.
