#!/usr/bin/env python3
"""Download the official game logos shown in the app's game picker.

Every logo is pinned by URL and SHA-256 and stored unchanged under
`datasets/sources/catalog/game-logos/` (DVC, not git: the repository is public and most of these
are non-free images). `build_browser_bundle.py` copies them into the bundle (`logos/`).
Logos are trademarks of their publishers, used only to label which game a toggle selects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = 'tcg-scanner-research/0.1'
SOURCES = [
    {
        'game': 'pokemon', 'file': 'pokemon.svg',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/1/1a/Pok%C3%A9mon_Trading_Card_Game_logo.svg',
        'page': 'https://commons.wikimedia.org/wiki/File:Pok%C3%A9mon_Trading_Card_Game_logo.svg',
        'sha256': '41f1ee7831d77f38a45d8da7266128ce6cfda7310999842652c33e9edf7068be',
        'license': 'Wikimedia Commons: public domain (text logo); Pokémon is a trademark of Nintendo/Creatures/GAME FREAK.',
    },
    {
        'game': 'onepiece', 'file': 'onepiece.webp',
        'url': 'https://upload.wikimedia.org/wikipedia/en/c/c2/One_Piece_Card_Game_logo.webp',
        'page': 'https://en.wikipedia.org/wiki/File:One_Piece_Card_Game_logo.webp',
        'sha256': '4088f78de2ed04a37b79b286f5f2d4c98dedf000f3b57e8a23f522a67083f165',
        'license': 'English Wikipedia: non-free logo (fair use); © Eiichiro Oda/Shueisha, Toei Animation; Bandai.',
        # Black artwork on transparency: the app inverts it on dark backgrounds.
        'monochrome_dark': True,
    },
    {
        'game': 'yugioh', 'file': 'yugioh.png',
        'url': 'https://upload.wikimedia.org/wikipedia/en/0/0a/Yu-Gi-Oh%21_Trading_Card_Game_logo.png',
        'page': 'https://en.wikipedia.org/wiki/File:Yu-Gi-Oh!_Trading_Card_Game_logo.png',
        'sha256': '497ae19eaf7a422b5132f937e9c2da4564e9d7e9037fa64a319c9801a0b33f4c',
        'license': 'English Wikipedia: non-free logo (fair use); Yu-Gi-Oh! is a trademark of Konami / Kazuki Takahashi.',
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'datasets/sources/catalog/game-logos')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logos = []
    for src in SOURCES:
        path = args.output / src['file']
        data = path.read_bytes() if path.exists() else urllib.request.urlopen(
            urllib.request.Request(src['url'], headers={'User-Agent': USER_AGENT}), timeout=60).read()
        if hashlib.sha256(data).hexdigest() != src['sha256']:
            raise ValueError(f"{src['url']} changed: expected {src['sha256']}")
        path.write_bytes(data)
        logos.append({**src, 'bytes': len(data)})
    (args.output / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'artifact_type': 'game_logos', 'logos': logos},
                                                          indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'logos': [l['file'] for l in logos], 'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
