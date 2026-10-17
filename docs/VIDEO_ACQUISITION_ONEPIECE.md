# One Piece 4K footage quality pass

Checked 2026-10-15. This pass audits a small public-video excerpt for acquisition suitability. Public access does not establish permission to reuse the footage; all local samples are inspection-only and remain unpromoted.

## Existing EJ Pokeventures excerpt

The inspected source is [“My COMPLETE One Piece Card Collection – OP01 to OP11”](https://www.youtube.com/watch?v=igoDuKsMfJo), excerpted at requested source interval 150–162 seconds. Its local frame manifest is [`manifest.json`](../runs/video_mining/luna-onepiece/igoDuKsMfJo-clip-local/manifest.json), with eight selected 3840×2160 frames at requested-offset times from 1 to 15 seconds. The eight frames are temporal samples, not eight unique scenes or a card count. Adjacent samples repeat binder pages and the same physical cards. The visible material includes at least two binder spreads; many pocket cards show readable artwork and names, while printed set codes are too small to treat as identity evidence. A rough full-spread count is around a dozen to the high teens of pockets, but partial edges and overlays prevent a reliable exact count.

These are physical cards in sleeves and binder pockets. Sleeve reflections are visible, and the first inspected instant has strong hand occlusion. Several frames also contain a large burned-in “SAMPLE” card graphic covering part of the binder. That graphic is an on-screen overlay, not a physical card. Existing detector/teacher boxes are proposals and are not counted as verified annotations or card identities.

## Supplemental Geek Provisions excerpt

The additional source is [“Every Luffy Fan Needs These Cards! | One Piece Card Game Binder Tour”](https://www.youtube.com/watch?v=ridHY4Kt8CE), channel Geek Provisions. Public yt-dlp metadata listed format 313 at 3840×2160, 30 fps (VP9). A 12-second section was requested at 120–132 seconds; the section download is 18,622,393 bytes. Its manifest is [`manifest.json`](../runs/video_mining/onepiece-quality-pass/ridHY4Kt8CE-120-132/manifest.json) and records six selected 4K frames. The requested section interval is provenance, not frame-accurate source-time verification: yt-dlp section cuts and keyframe preroll can affect exact frame alignment.

Visual inspection shows a real binder spread with about twelve clearly visible pocket cards in one sampled view, plus partial pockets, and a separate physical card held close to the camera. At that close distance the card’s artwork, printed name and set-number area are readable in the footage; this visual observation does not constitute an identity label. The source has an always-present host picture-in-picture graphic in the lower-right area, and one sampled spread also has a large burned-in “SAMPLE” card graphic. Hand occlusion and reflective foil glare affect the held-card views. The larger physical card is much more detailed than the existing EJ full-page view, while the inset and occasional sample overlay make the new source less clean for full-page detection. It is a useful supplemental source, not a replacement for the clearer unobstructed binder views.

A shared, cached Grounding DINO tiny proposal-only pass processed all six supplemental frames at 800px input short edge. It emitted 18 boxes across the frames: 9, 2, 1, 2, 2, and 2. These are raw per-frame detections, not 18 distinct cards; they include repeated views of the same held card and can include presenter/overlay regions. No card identities or verified card boxes were produced. The six detector visualizations and manifest are in [`detector_audit`](../runs/video_mining/onepiece-quality-pass/detector_audit/manifest.json); boxes are unverified proposals, with no teacher annotations supplied for this source.

Two earlier 12-second probe intervals from this same video (20–32 and 60–72 seconds) showed only the presenter and closed binder. Their 12 saved frames were removed; six frames from the card-bearing 120–132 second interval remain. Total retained frame count for this pass is six, below the requested maximum of twelve. New raw excerpt media totals 18.6 MB, below the 150 MB limit.

## Labels and limitations

This pass makes no card identity, corner, or physical-print ground-truth claims. Frame candidates record pixel hashes, dimensions, source group, and selection metrics. “Physical card” is a visual quality observation, not an annotation export. All footage reuse rights remain unknown; no creator was contacted. Model boxes on these frames remain unverified proposals unless separately checked against the pixels.
