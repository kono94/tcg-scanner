# Slow One Piece Binder Search

Date: 2026-10-15. This pass looked for top-down One Piece binder views and quiet collection presentations with public, unauthenticated video access. Format availability was checked against yt-dlp's actual format table; titles or a `[4K]` tag alone were not treated as resolution evidence. Three new 2160p leads were verified. Two yielded useful excerpts; one sampled intro had no cards. Candidate images are unlabeled visual inputs for localization and similarity inspection. None are training labels, independent validation, or real-camera benchmark evidence. Reuse rights are unknown.

## Best binder source: Bros Open Packs

[How is our One Piece binder doing?](https://www.youtube.com/watch?v=4QxpRG4E6d0) offers the closest fit: a top-down binder, no host inset or digital card graphics, and a full two-page spread with about 15–20 visible pocket cards. The two adjacent requested 12-second sections are in `runs/video_mining/binder-search-op-slow/raw/`. The verified downloadable stream was format `315`, 3840×2160 at 60 fps VP9. Both excerpts were decoded and confirmed at that resolution. Source-time alignment is approximate because section cutting can include keyframe/container preroll.

The 12 retained JPEGs are in the two excerpt directories, with per-frame SHA-256 and decoded dimensions in each `manifest.json`. Most are repeated views of the same spread, not 12 new pages. The card faces appear about 450–550 source pixels across their short edge. Some frames show a hand flipping pages or obscuring pockets; mild sleeve reflections and a monitor/keyboard along the upper edge remain. The last section contains the clearest full spread. This is useful multi-instance detector and similarity-input material, subject to later crop review.

## Supplemental close presentation: Stoneyyoptcg

[My $3000+ One Piece TCG Collection Binder!! Graded Cards & SPs](https://www.youtube.com/watch?v=ho7IFpPXI-A) exposes actual format `313`, 3840×2160 at 30 fps VP9. Five selected frames from the requested 90–102 second excerpt show large physical card faces on a tabletop. a pale green character card and a blue-purple foil card recur, with red/yellow card faces and other partial faces; this is a small set of repeated instances, not a broad binder page. Card text and art are clear enough to inspect, but the light is dark and foil reflections are visible. A sixth sampled frame had severe vertical color corruption and was discarded. The local excerpt and cleaned frame manifest preserve that note.

## Additional qualified lead with a rejected intro probe

[One Piece Card Game - Collection Tour!!!!!!](https://www.youtube.com/watch?v=_caIcnf-CrU) by squaaaaw has actual format `313`, 3840×2160 at 30 fps VP9 and a 21:59 duration. A brief requested 80–92 second probe showed presenter/room and storage boxes without cards, so that probe was removed and no source frames were retained. Only its metadata and the probe decision remain in the JSON; the rest of the tour has not been inspected.

## Acquisition accounting and limits

The acquisition used 3 short sections totaling the byte count in `resources/benchmarks/binder_search_op_slow.json`, below the 300 MB raw-media limit. Seventeen 3840×2160 candidate JPEGs remain, below the 24-frame cap. At the start of the pass, free space was 28 GiB, leaving the requested 3 GiB reserve. The full source videos were not downloaded. The mining script uses a whole-frame sharpness/diversity heuristic; it does not establish card boxes, exact identities, printing labels, or distinct physical instance counts. No visible card identity has been transcribed or guessed.

The acquisition manifest is `resources/benchmarks/binder_search_op_slow.json`. Frame-level provenance, approximate excerpt-relative timestamps, SHA-256 hashes, and the unlabeled status are in each folder's `manifest.json` beneath `runs/video_mining/binder-search-op-slow/`.
