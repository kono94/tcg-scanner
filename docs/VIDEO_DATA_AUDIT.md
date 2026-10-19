# Actual video acquisition and card-data yield

Updated 2026-10-16 after the expanded acquisition and exhaustive crop review. Count clips, selected frames, frame-local detector proposals, distinct card views and exact labeled examples separately. A detector box is not a new card identity, and adjacent views of one held card do not create independent training/test examples.

## Acquired baseline and first quality pass

| Source | Native source | Retained frames | Observed usefulness / limitation |
| --- | --- | --- | --- |
| One Piece, EJ Pokeventures `igoDuKsMfJo` | 3840×2160 | 8 | At least two binder spreads; repeated cards, sleeve glare, large digital card overlay obscures physical cards |
| One Piece, Geek Provisions `ridHY4Kt8CE` | 3840×2160 | 6 | Binder view plus detailed held card; host inset, fingers, foil glare and occasional SAMPLE graphic |
| Pokémon, PokeUnlocked `ZQ11RKIA7BE` | 3840×2160 | 8 | Two English closeup views and three page views; three other selected frames are graphics/outro |
| Pokémon, Midnight Rare ASMR `BF_gXDPnhCg` | 3840×2160 | 8 | One Japanese card view repeated eight times; seven near-duplicates and a language mismatch with the English gallery |
| Pokémon, No Frills ASMR `QIY0LGl_JAA` | 3840×2160 | 6 | One English Tyranitar V view, five near-duplicates; large face, glove/foil occlusion |

Total: **five actual 4K excerpts / 36 retained frames**. The Pokémon visual audit identifies four distinct closeup views across its three sources, not 22 distinct cards. The One Piece spread/card count has not been reliably deduplicated. See `VIDEO_ACQUISITION_ONEPIECE.md` and `VIDEO_ACQUISITION_POKEMON.md` for source URLs, retained manifests and timestamp limitations. The separate 1080p inspection clip is ineligible under the current resolution requirement.

## What reached training or independent testing

- The first synthetic-trained OBB batches produced **116 crop proposals**, quarantined after inspection found artwork/text fragments and graphics instead of reliable physical boundaries.
- A shared frozen GroundingDINO audit of the original **24 frames** produced **262 frame-local proposals**: 90 in One Piece, 116 in the Pokémon binder clip and 56 in the pack clip. Counts include repeated views, graphics and non-card regions. They are not 262 cards. Audit manifest: `datasets/sources/videos/acquisition_audit/foundation_proposals/manifest.json`.
- The earlier two-frame detector experiment yielded 23 source-resolution crops; OCR/catalog lookup resolved zero of 16 prioritized proposals. Frozen DINOv2 shortlists exist for those 16, with identities still unverified. OCR is now an optional clear-crop fast pass; similarity is the normal recognition path.
- Feature matching supported geometry proposals for four physical-card locations in one One Piece frame, with competing weaker matches on two. Those are six candidate warps, not six established exact identities or independent test examples.
- These baseline proposals were not promoted to exact-ID data. The later expanded acquisition supplies the 298 reviewed queries described below; no model has been trained/evaluated on them yet. No independent real-card corner benchmark has been established. Executed detector training still uses 200 synthetic scenes / 1,321 rendered instances.

## Expanded six-worker acquisition

The completed campaign acquired **12 excerpts from 11 videos and 69 retained native-4K frames**. Six Pokémon host/interstitial frames were excluded from localization, leaving 63 card-bearing frames. The English One Piece, modern/vintage Pokémon, slower One Piece, Japanese One Piece, and Pokémon master-set lanes ran in two waves of three Luna workers with bounded media budgets and a shared storage reserve.

| Lane | Acquired excerpts | Retained frames |
| --- | ---: | ---: |
| English One Piece | 2 | 16 |
| Modern Pokémon | 2 | 3 |
| Vintage Pokémon | 2 | 12 |
| Slower One Piece | 3 from 2 videos | 17 |
| Japanese One Piece | 1 | 9 |
| Pokémon master sets | 2 | 12 (6 excluded host frames) |

GroundingDINO localization on the 63 selected frames generated 802 raw proposals, of which 716 crop files were saved and 648 passed a provisional geometry screen. Visual review still found whole pages, background/PiP cards, non-cards, partial faces and severe blur/glare among those crops. Geometry eligibility is not a single-card quality label.

The [completed cleanup](DATASET_CLEANUP.md) admits **298 SOL-reviewed reference-matched queries across 158 catalog-art identities**, split **188 train / 24 val / 86 test** across eight source groups. Both passes remove **304 bad derived crops**. All **519 candidates remaining after the first pass** have been visually reviewed: 262 newly admitted, 143 rejected, 45 reviewed identities excluded for unusable geometry, and 69 unresolved exact identities/printings. The remaining 114 staged exclusions are outside training; zero in this pool remain unreviewed. Counts exhaust the 716 saved proposals. Repeated views are not additional physical cards. Video isolation is enforced; these AI-reviewed labels are not measured recognition accuracy, perfect printing/finish identification or independent corner ground truth.

Search lanes cover English One Piece binder tours, modern English Pokémon, vintage English Pokémon, slower One Piece binder presentations, international One Piece collections with explicit language, and Pokémon master-set/page tours. Verify actual available 2160p formats, then inspect the downloaded section. Avoid presenter intros, large card graphics, fast page flips, tiny pocket faces and repeated views masquerading as different cards. Retain sharp full faces with enough source pixels, and record glare/hand/edge occlusion explicitly.

The machine-readable candidate files are separate per worker to avoid concurrent edits. Acquired clips/frames and source hashes live under ignored `datasets/sources/videos/binder-search-*/` directories. Keep all frames from one creator/video/session together when splitting; do not use neighboring video frames as independent validation. Additional frozen similarity/local-feature evidence can propose catalog labels later; unknowns stay unknown rather than being guessed.
