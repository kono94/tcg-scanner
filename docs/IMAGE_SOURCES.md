# Card image sources and SAMPLE watermark experiment

Checked: 2026-10-15. Source discovery and one AI-restored example are complete; no marketplace photo corpus has been acquired and no recognition improvement has been measured.

A later [official-source clean-reference probe](ONEPIECE_CLEAN_REFERENCES.md) found one watermark-free Japanese-language `OP03-094` asset; two inspected OP01 variants still contained SAMPLE. This is a separate language-specific digital-reference candidate, not a clean English replacement or independent camera image. The probe does not establish broad clean-source coverage.

## Watermark findings and handling

The local references come from Bandai image URLs recorded in their JSON sidecars. For example, `datasets/card_recognizer/cards/OP03/OP03-094.jpg` has a large white SAMPLE word and drop shadow covering artwork. Removing the overlay can reduce the mismatch with physical cards, but the underlying pixels are missing. Inpainting produces an estimate of them, not an authentic scan.

The built-in image generation tool produced a complete-card cleanup example at `runs/card_recognizer/watermark_experiment/OP03-094__ai-restored.png`. The original corpus, prepared gallery and bundled models are unchanged. The adjacent manifest records source/output hashes, prompt and synthetic status. Visual inspection confirms the word was removed and the readable name/card number remain present; exact authenticity of reconstructed art is unverified. This example is not promoted into the gallery or an independent evaluation split.

Preferred reference order:

1. A genuine unwatermarked scan/photo with confirmed exact printing, language and artwork variant.
2. The original digital reference with watermark metadata, plus a validated descriptor policy that tolerates the missing region.
3. AI-restored variants as explicitly synthetic training/ablation candidates. They may change subtle art/text and should not silently become canonical truth.

Compare original-only, genuine-clean and AI-restored references on the same independent unwatermarked queries. Measure retrieval Recall@K, wrong variants, accepted precision/coverage and unknown false accepts. Synthetic scenes made from the same restored images are not independent evidence that restoration improves camera recognition. Test both original/restored training views rather than assuming a bulk replacement helps. Descriptor region weighting or localized watermark augmentation is an optional comparison; applying a large central mask everywhere would also discard useful artwork.

## Sources worth pursuing

Tournament broadcasts, binder tours, overhead deck profiles and pack openings are additional video sources. See [video mining](VIDEO_MINING.md) for located official VODs, the implemented frame-selection command and the proposed labeling pipeline.

| Source | Verified access/content | Role and next check |
| --- | --- | --- |
| [Roboflow OnePiece Card Game](https://universe.roboflow.com/one-piece-card-game/onepiece-card-game-legoh) | Public project lists CC BY 4.0, 649 source images, 125 classes, and a version with 1,599 images. Classes include exact `OP01` IDs and `_p1` variants. | Concrete detection/recognition data candidate. Download/export access, actual photo-versus-scan composition, source provenance, geometry and duplicate/augmentation history still need auditing. Do not treat the larger version count as independent captures. |
| [arjunkai/optcg-api](https://github.com/arjunkai/optcg-api) | Documents a public image proxy with curated scans for DON/JP-exclusive variants and upstream fallbacks. Data endpoints require an API key; the public proxy is documented separately. | Candidate reference enrichment, not verified clean coverage for the entire corpus. Regular-card fallback can return the same watermarked Bandai image. MIT applies to the Worker code; upstream image/data rights are separate. |
| [OPTCG API](https://www.optcgapi.com/about/updates/2) | Documents image URLs and official image IDs, preserving mappings such as `_p1`. Its image collection was downloaded from the official English card site. | Useful for catalog/variant coverage. It is not automatically a source of clean artwork or independent camera examples. |
| [eBay Browse API](https://edp.ebay.com/api-docs/buy/browse/resources/item/methods/getItem) | Documents primary and additional listing image URLs. | Technically useful real-photo discovery, but the current [API agreement, section 9.10](https://developer.ebay.com/join/api-license-agreement) restricts ML training and synthetic-dataset uses without prior written consent. Generic API access does not supply that permission. Its [user agreement](https://www.ebay.com/help/default/default/user-agreement?id=4259) also restricts unauthorized scraping. Do not build the default training source around it. |
| [Cardmarket](https://help.cardmarket.com/en/cardmarket-api) | Official help says new API applications are not being accepted. Product/listing pages are available, but image collection/ML access has not been established. | Low-priority acquisition dependency. Do not assume a catalog thumbnail is a unique seller photo, or that a current public training API exists. The terms page returned 403 during this check; no broader permission claim is made. |
| Shop/collector/grading image feeds with an explicit reuse arrangement | No feed acquired in this work. | A useful route to real fronts, sleeves, glare and multiple views with source-level identity information. Obtain imagery from the contributing source and retain its reuse terms rather than assuming marketplace availability grants ML rights. |

## Card-back references (2026-10-17)

`scripts/acquire_card_backs.py` pins each source by URL and SHA-256, keeps the
original bytes under `datasets/curated/recognition/backs/<game>/originals/` and
writes `datasets/curated/recognition/backs/manifest.json` (provenance, license
note, geometry, hashes). Nothing in it is synthetic.

| Identity | Source | Role |
| --- | --- | --- |
| `back:pokemon` | English Wikipedia `File:Pokemon_Trading_Card_Game_cardback.png` (non-free/fair use; scan of a physical card) | gallery reference |
| `back:pokemon` | Internet Archive `150000cardlot_batch14`, scan set 1 back side: 16 physical vintage-era backs on a flatbed scan (no license stated) | 2 crops as gallery references, 12 held out for evaluation; 2 cards were not extracted (contour too weak or touching) |
| `back:onepiece` | English Wikipedia `File:One_Piece_Card_Game_back.webp` (non-free/fair use, 200x280) | gallery reference |
| `back:onepiece` | Two tilted back illustrations from the official English site footer (`en.onepiece-cardgame.com`, transparent PNG/WebP), rectified from their alpha outline | gallery references |

No back was found in the local video-mining frames, so there are no real camera
views of backs yet; the other Internet Archive scan sets returned HTTP 500 at
acquisition time. The images are copyrighted card designs used locally as
recognition references only.

## Automatic ingestion policy

Use source adapters to emit a manifest, then validate and stage imports. `scripts/import_real_card_images.py` already accepts CSV/JSONL containing card ID, image URL/local path and provenance; its training promotion is explicit. It currently targets existing One Piece identities, not a universal cross-game identity schema.

For each candidate, retain source URL/item/image ID, source identity and confidence, language/variant, image hashes, acquisition/reuse terms and actual role. Distinguish clean scan, real capture, watermarked reference and generated restoration. Resolve ambiguous listing titles with visible card-number/variant evidence; lots, backs, slabs, reprints and reused stock images must not become automatic exact-ID labels.

Deduplicate exact and near-identical images across providers. Keep one listing/session and all its derived crops/augmentations in one split. Localize physical cards, rectify from source resolution, then use the shared canonicalizer. Real photographs can train recognition and detection; source AABBs do not become trustworthy OBB/corner labels merely by expanding them. Teacher geometry remains a proposal with confidence/provenance.

Keep originals and staged candidates outside the active gallery until their identity and preprocessing contracts pass. Rebuild compatible embeddings after changing reference pixels. The immediate acquisition priority is auditing the named Roboflow dataset and resolving clean-reference coverage, alongside the OBB-to-cosine baseline.
