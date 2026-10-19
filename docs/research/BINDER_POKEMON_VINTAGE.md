# Vintage English Pokémon binder video search

Updated 2026-10-15. This lane searched for vintage and older English Pokémon binder footage distinct from the already mined modern binder and Japanese pack-opening sources `ZQ11RKIA7BE`, `BF_gXDPnhCg`, and `QIY0LGl_JAA`.

Three distinct public YouTube sources positively listed a 3840×2160 video format in yt-dlp metadata. The metadata command was:

```sh
PYTHONPATH=datasets/sources/videos/tools .env/bin/python -m yt_dlp --skip-download --dump-single-json --no-playlist --socket-timeout 15 --retries 0 --extractor-retries 0 URL
```

Probe JSON summaries are in `datasets/sources/videos/binder-search-pokemon-vintage/`. yt-dlp warned that available formats can be incomplete without a JavaScript runtime. Positive 2160p format entries are verified listings; absence of a 2160p entry in other candidates is inconclusive. The short clips below were separately confirmed by ffprobe as 3840×2160 AV1 at the listed frame rates. “4K” here means the platform offered 3840×2160; the source camera/original master resolution is unknown.

## Best inspected windows

1. [Kreesky — “I BOUGHT A VINTAGE BINDER OF POKEMON CARDS - AND FOUND GOLD STAR CARDS!!!”](https://www.youtube.com/watch?v=fG9cnN5AH6Q), uploaded 2022-11-25. A 12-second source window at **600–612 s** was acquired as 3840×2160 AV1, 60 fps. It shows an English vintage binder open on a table, with page turns and several different cards visible across two facing pages. At least six distinct card faces appear over the sampled frames; these are visual-view estimates, not verified identity labels. Sleeves produce glare, the hand/forearm crosses parts of the pages, and the ring binder divides the view. The clip has an overhead view with substantial, stable card faces and is the strongest page-scene candidate in this lane.

2. [Deep Pocket Monster — “Complete Set in 1-Hour or Lose Them All (RISKY Pokémon Card Challenge)”](https://www.youtube.com/watch?v=W43vmn9zkRE), uploaded 2022-11-27. A 12-second source window at **300–312 s** was acquired as 3840×2160 AV1, 29.97 fps. It contains several older English card views in a store/set challenge, with a mix of a small binder and individually held cards. The sampled sequence includes roughly four distinct useful card views; some are repeated or blurred during handling. Fingers, sleeve reflections, retail background, and intermittent close framing limit page-level utility. Its cards differ from the Kreesky pages, so it adds visual variety.

The clips are in `datasets/sources/videos/binder-search-pokemon-vintage/`; 12 extracted 3840×2160 JPEGs (six per clip, about two seconds apart) are in its `frames/` subdirectory. Clip-relative frame times are approximate because section cuts can align to nearby keyframes. Raw acquisition sections total about 39 MB; selected JPEGs total about 11 MB. Both are retained for the parent run’s detector-proposal/crop inspection. They have not been labeled, annotated, promoted to a dataset, or treated as independent evaluation. Exact card/printing IDs remain unknown. Similarity retrieval remains the normal identity path; OCR is only a possible fast pass on exceptionally clear crops.

Exact useful source windows are known only for the two inspected clips above. The first attempted Kreesky window at 120–132 s showed outdoor package pickup/unboxing context rather than cards and was discarded. No broader source-time regions are claimed.

## Additional verified 4K lead

[HLD Cards & Collectibles — “Pokemon Binders but they get increasingly more ridiculous”](https://www.youtube.com/watch?v=xFEyeGtsIS0) positively listed 3840×2160 at 30 fps (VP9 and AV1 formats) on 2026-10-15. Its thumbnail depicts a binder page but includes a prominent title overlay and repeated custom-looking card artwork. No clip was acquired and no content-quality pass was made, so it ranks below the two inspected sources.

Reuse rights are unknown for all sources. Public watch access does not grant reuse permission. No cookies, accounts, proxies, or raw full-video downloads were used.
