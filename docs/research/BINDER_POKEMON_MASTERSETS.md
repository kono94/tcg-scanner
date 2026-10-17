# Pokémon master-set binder source search

Updated 2026-10-15. This lane searched for English Pokémon master-set or complete-set binder tours that could add clean page imagery to the current native-4K candidate pool. Existing source IDs `ZQ11RKIA7BE`, `BF_gXDPnhCg`, `QIY0LGl_JAA`, `fG9cnN5AH6Q`, `W43vmn9zkRE`, `Wv8WJmU3xQc` and `xFEyeGtsIS0` were avoided.

Four fresh sources positively listed at least one 3840×2160 stream in yt-dlp metadata: PokeUnlocked's [What 1 Year of Pokémon Master Sets Looks Like](https://www.youtube.com/watch?v=lZOg6TUQI9s), Poké Duell's [This Gen 5 Pokémon Binder Took Over My Life](https://www.youtube.com/watch?v=zJOawf6wEgc), Deep Pocket Monster's [Collect All 1,000+ Pokémon in ONE Binder](https://www.youtube.com/watch?v=6uXaVLcBo5I), and its [Complete Set in 151 Min. challenge](https://www.youtube.com/watch?v=riEB2yKmlyg). The exact format metadata, probe paths and acquisition status are in `resources/benchmarks/binder_search_pokemon_mastersets.json`. Listed 2160p is platform format evidence; it does not prove the creator recorded natively at 4K. Reuse permission is unknown.

Two bounded excerpts were actually acquired, totaling about 56 MB. The Poké Duell excerpt (`300–312 s` requested) yields six mined 3840×2160 JPEGs. Visual inspection shows one physical English-card closeup repeated across the samples, with a host inset, fingers, sleeve reflection and busy playmat. This is a candidate for an individual crop experiment, not binder-page acquisition. The Deep Pocket Monster excerpt (`300–312 s` requested) yields six 4K JPEGs, but these show the host/interstitial graphics and no card faces. It is discarded for recognition use. There are zero useful full binder-page views and approximately one unique physical card view among the acquired excerpts.

The PokeUnlocked tour appears strongest from titles and chapter structure, with 14 master-set sections, but both attempted section downloads returned HTTP 403 to ffmpeg. No frames from it were retained. The 151 challenge remains a metadata-only lead. These results do not meet the requested quality target for a binder benchmark; they document a clean failure instead of treating 4K metadata as useful imagery.

Retained frame manifests:

- `runs/video_mining/binder-search-pokemon-mastersets/zjoawf6wEgc-frames/manifest.json`
- `runs/video_mining/binder-search-pokemon-mastersets/6uXaVLcBo5I-frames/manifest.json`

Clip-local frame times use reported FPS and keyframe-aligned extraction; requested source windows are approximate, not exact presentation-time annotations. All frames remain unlabeled, and repeated frames from a single source are not independent identity examples. No frames are promoted to training or evaluation. The source group must remain intact if later proposals are evaluated. Similarity is the normal recognition path; OCR is an optional bounded fast pass only for exceptionally clear card crops.
