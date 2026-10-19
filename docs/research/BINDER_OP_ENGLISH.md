# English One Piece Binder Video Search

Checked 2026-10-15. This bounded search looked for new English One Piece TCG binder or collection-tour videos suitable for automatically mining individual physical-card crops. Existing candidates `igoDuKsMfJo` and `_caIcnf-CrU` were excluded from the new-source list.

Two distinct sources were format-verified and sampled. The initial three binder search results did not qualify: Simplycozytcg and JOHKU DMD were limited to 1080p in the probe, and the AlbinoManatee Short was 720×1280 with Arabic language metadata. Additional search found new 4K candidates. yt-dlp 2026.08.19 warned that no JavaScript runtime was available, so positive format listings establish an available encoded format while missing higher formats remain inconclusive.

| Initially screened candidate | Highest listed video format | Subject evidence | Decision |
|---|---:|---|---|
| [Simplycozytcg collection and binder organization](https://www.youtube.com/watch?v=hbCEpdhthhs) | 1920×1080, 30 fps, AV1 | Description says the creator is organizing an entire One Piece collection into binders. | Strongest subject match, but below the 1440-pixel minimum; framing, card size, and overlays uninspected. |
| [JOHKU DMD Premium Bandai binder opening](https://www.youtube.com/watch?v=ogldPHdj0b0) | 1920×1080, 30 fps, H.264 | English metadata identifies a binder opening. | Product opening may show few populated pages; below the minimum and visual quality uninspected. |
| [AlbinoManatee binder collection Short](https://www.youtube.com/watch?v=9UZtKQ7XWVM) | 720×1280 portrait, 30 fps, AV1 | Description says Romance Dawn and Paramount War master set. | Below the minimum; YouTube language metadata is Arabic, so English-language content is not established. |

The initial candidates were retained as negative search results. Further probes found: [One Piece Card Game Binders!](https://www.youtube.com/watch?v=9dzC1JnnOWQ), 8bitkevin, 3840×2160/30 VP9, and [STUNNING CARDS! Opening 2 Boxes of OP12!](https://www.youtube.com/watch?v=rvFC1HtK2BY), MarinefordTCG, 3840×2160/60 AV1. The 8bitkevin video was excluded after two inspected sections showed binder covers/opening and a persistent presenter inset, without usable card faces.

Two short segments from distinct videos were acquired and visually inspected:

| Source segment | Extracted frames | Visual result and limits |
|---|---:|---|
| [squaaaaw collection tour](https://www.youtube.com/watch?v=_caIcnf-CrU), requested 248–264s; [manifest](../../datasets/sources/videos/binder-search-op-english/_caIcnf-CrU-248-264-frames/manifest.json) | 8 | Shows two different populated 3×3 spreads, about 18 physical cards each. Wide tilted view, small card faces, hands, moderate reflections, room clutter and visible monitors. |
| [MarinefordTCG OP12 opening](https://www.youtube.com/watch?v=rvFC1HtK2BY), requested 353–369s; [manifest](../../datasets/sources/videos/binder-search-op-english/rvFC1HtK2BY-353-369-frames/manifest.json) | 8 | Sequential close-held physical cards; card faces and printed text are sharp enough for crop proposals. Persistent upper-right presenter inset, finger occlusion, and blurry background cards; supplementary single-card footage only. |

The section ranges are requested yt-dlp bounds, not guaranteed frame-accurate source times. Manifest timestamps are relative to the extracted excerpt and use decoded frame index/reported fps. Original source hashes, selected-format details, and quality notes are recorded in the JSON manifest. Identity and geometry labels remain unresolved, reuse rights are unknown, and neither segment is promoted to training or independent query truth.

Machine-readable metadata and per-candidate caveats are in [`resources/benchmarks/binder_search_op_english.json`](../../resources/benchmarks/binder_search_op_english.json). No video media or preview bytes were saved; the assigned ignored artifact directory remains empty.
