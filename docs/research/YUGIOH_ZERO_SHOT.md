# Yu-Gi-Oh! zero-shot retrieval (2026-10-21)

Can the deployed encoder (`dinov2-arcface-gpu-20261017`, trained on One Piece and Pokémon only)
recognize Yu-Gi-Oh! cards without retraining? This is the transfer evaluation named in AGENTS.md.

## Data

- Gallery: the English YGOPRODeck catalog (`scripts/catalog/sync_yugioh_gallery.py` →
  `datasets/sources/catalog/yugioh-en-ygoprodeck`): 14,617 cards, 14,781 artworks (one row per
  artwork), 44,667 TCG set-code × rarity printings. Embedded with the fp32 ONNX encoder through the
  known-full-card-bounds adapter (`scripts/curation/zero_shot_retrieval.py embed`).
- Queries: 30-second excerpts of five public videos with real ≥1440p formats
  (`datasets/sources/videos/yugioh-zero-shot`), frames mined with `mine_video_frames.py`, cards
  cropped by the deployed corner detector `corners-s640-v4-ft`. Only 2 of 15 first binder-tour
  candidates had a real ≥1440p format; a wider search found the rest.
- Labels: AI visual review of every crop against the reference artworks
  (`datasets/review/yugioh-zero-shot/review.json`), at card-name level; alternate artworks and
  printings/rarities are not distinguished.

## Results

| Clip | Content | Card crops | Top-1 | Top-5 | Excluded |
| --- | --- | --- | --- | --- | --- |
| `H-W_jzyPffY` | Goat-format deck profile (old cards) | 30 | 28 | 29 | 0 |
| `_cLjLR9aM-c` | Red-Eyes deck profile | 8 | 8 | 8 | 0 |
| `_zZ-qa_Q4ok` | Quarter Century Bonanza binder | 41 | 38 | 39 | 0 |
| `io3dtxiQw64` | 2016 Legendary Collection binder | 9 | 9 | 9 | 11 booster packs |
| `zaejAoUS85A` | modern Quarter Century Secret Rare binder | 91 | 80 | 87 | 5 unreadable |
| **Total** | | **179** | **163** | **172** | 16 |

The 179 crops are repeated views of 35 distinct cards from five videos, not independent samples.

- Misses concentrate on heavy-foil Quarter Century Secret Rares: Green Gadget fails on all four
  crops (correct card ranked 23–85), Black Luster Soldier – Envoy of the Beginning ranks second
  behind Kelbek the Ancient Vanguard on four crops. Two blurred, finger-occluded Sangan views fail.
- Scores: correct top-1 median 0.68 (min 0.38); wrong top-1 at most 0.56; booster packs at most
  0.38. With the app's uncalibrated defaults (score ≥ 0.45, margin ≥ 0.05) about 132–144 of the
  163 correct top-1 hits are accepted and none of the 16 wrong ones.
- The corner detector finds Yu-Gi-Oh! cards without retraining but also boxes booster packs
  (11 of 20 detections in the pack-heavy clip) and cuts some crops short.

These are retrieval diagnostics on few videos with AI-reviewed labels, not scanner accuracy.
Heavy foil is the main gap; foil Yu-Gi-Oh! training views or foil augmentation are the first
candidates if accuracy on Secret/Quarter Century Rares matters.
