"""Materialize DVC-tracked directories from a DVC store without DVC (stdlib only).

The store (filled by deploy/push-artifacts.sh) uses DVC's cache layout: files/md5/<2>/<30>, with
one `.dir` JSON listing per tracked directory. Each file is hard-linked into place, so a checkout
costs no space and takes seconds even for the 24k catalog thumbnails.

    python checkout-artifacts.py <store> web/public/bundle.dvc datasets/reference-thumbnails.dvc
"""
import json
import os
import shutil
import sys
from pathlib import Path


def obj(store: Path, md5: str) -> Path:
    return store / 'files' / 'md5' / md5[:2] / md5[2:]


def checkout(store: Path, pointer: Path) -> None:
    # .dvc pointers are tiny YAML; read the two keys needed instead of requiring PyYAML.
    fields = dict(line.strip().lstrip('- ').split(': ', 1) for line in pointer.read_text().splitlines()
                  if ': ' in line)
    md5, target = fields['md5'], pointer.parent / fields['path']
    if not md5.endswith('.dir'):
        sys.exit(f'{pointer}: only directory outputs are supported')
    listing = obj(store, md5)
    if not listing.exists():
        sys.exit(f'{pointer}: {md5} is not in {store} (run deploy/push-artifacts.sh on the Mac)')
    entries = json.loads(listing.read_text())
    missing = [e['relpath'] for e in entries if not obj(store, e['md5']).exists()]
    if missing:
        sys.exit(f'{pointer}: {len(missing)} objects missing in {store}, e.g. {missing[0]}')
    tmp = target.with_name(target.name + '.checkout')
    shutil.rmtree(tmp, ignore_errors=True)
    for e in entries:
        dest = tmp / e['relpath']
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(obj(store, e['md5']), dest)
        except OSError:  # other filesystem, or protected_hardlinks without group write
            shutil.copyfile(obj(store, e['md5']), dest)
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    print(f'{pointer}: {len(entries)} files -> {target}')


if __name__ == '__main__':
    store = Path(sys.argv[1])
    for p in sys.argv[2:]:
        checkout(store, Path(p))
