# One Piece clean-reference probe

Checked 2026-10-15. This is a three-image probe of official card-list assets, not a bulk acquisition or a real-camera benchmark. The unchanged samples and SHA-256 manifest are in [`runs/card_recognizer/clean_reference_probe/`](../runs/card_recognizer/clean_reference_probe/manifest.json).

## Finding

One official, apparently watermark-free stock image was found: Japanese-language `OP03-094` (Air Door). It is a clean-reference candidate for a Japanese-language catalog entry, naturally namespaced as `onepiece:ja:OP03-094`; it should not replace the English `onepiece:en:OP03-094` reference. The image visibly carries Japanese card text and the printed `OP03-094` code. The official English card data identifies `OP03-094` as Air Door, and the Japanese/Asia official card-list host serves the same numeric card code without a visible SAMPLE overlay in this asset.

This is an official card-list image, not an independently captured photograph or scan. It can potentially supply a clean digital reference for its language-specific identity or visual-family comparison. It does not establish how the card looks through a camera, and it does not justify replacing or rewriting the existing English reference.

The other two inspected assets, `OP01-001` and `OP01-001_p1`, both show the SAMPLE overlay. The official card list distinguishes Roronoa Zoro from Roronoa Zoro (Parallel) while displaying the same printed code `OP01-001`; the parallel image URL uses `_p1`. This matches the repository's base/variant naming convention and shows why code-only OCR cannot separate those two arts. Both inspected variants remain watermarked.

The official English `OP03-094` image URL in the existing sidecar is on `en.onepiece-cardgame.com`; inspection of the local corpus image shows the SAMPLE overlay. The corresponding English-site image also contains SAMPLE. The unwatermarked result is the Japanese-language image delivered from `asia-en.onepiece-cardgame.com`, not an English-language clean replacement.

## Source and identity evidence

- Bandai's [official Asia-English card list](https://asia-en.onepiece-cardgame.com/cardlist/?search=true&series=556101) lists `OP01-001 | L | LEADER` as Roronoa Zoro and then Roronoa Zoro (Parallel) under the same printed code. The linked image files resolve to [`OP01-001.png`](https://asia-en.onepiece-cardgame.com/images/cardlist/card/OP01-001.png?260929=) and [`OP01-001_p1.png`](https://asia-en.onepiece-cardgame.com/images/cardlist/card/OP01-001_p1.png?260929=).
- Bandai's [official OP03 card Q&A](https://asia-en.onepiece-cardgame.com/pdf/qa_op03.pdf?20230526=) identifies `OP03-094` as Air Door. The downloaded [`OP03-094.png`](https://asia-en.onepiece-cardgame.com/images/cardlist/card/OP03-094.png?260929=) is 600×838, visibly Japanese-language, and showed no SAMPLE overlay on inspection. The local English sidecar keeps the same code/name mapping and points to the English-hosted image.
- A direct official English-hosted comparison, `https://en.onepiece-cardgame.com/images/cardlist/card/OP03-094.png?260929=`, displays English card text and SAMPLE. It agrees with the existing watermarked English corpus image.

The probe preserves three images byte-for-byte, all below 225 KB. Its manifest records exact URLs, dimensions, hashes, observed language and visual watermark status. No rights or reuse permission is inferred from public availability.

## Practical recommendation

Do not replace the watermark-restoration hypothesis with the assumption that Bandai's public assets are generally clean: two of three inspected assets were still watermarked. However, add a source check for the Japanese/Asia card-list image family before any restoration work. When a raw official asset is visibly clean and its in-image card code and official catalog mapping agree, it is a better authentic digital reference than an AI-inpainted image for that language-specific identity. Keep language and art variant explicit, preserve original references, and do not mix this stock asset into independent photo evaluation.

This probe has only one clean candidate and does not measure its corpus coverage. The next automated check should query a bounded list of exact IDs from the existing catalog against official card-list asset names, download only low-resolution card-view assets to a staging area, verify image hashes/language/watermark status, and keep a candidate unresolved when the image's language or `_pN` variant cannot be matched. Do not promote assets in bulk based only on URL shape.
