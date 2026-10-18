#!/usr/bin/env bash
# Copy the DVC objects of the deployed artifacts (model bundle, catalog thumbnails) into the VPS DVC
# store, where a tag deploy hard-links them into place (deploy/checkout-artifacts.py). Same result as
# `dvc push -r vps <pointers>`, but one rsync instead of 24k SFTP uploads (minutes instead of hours).
#
#   deploy/push-artifacts.sh                  # before pushing a tag that changed a .dvc pointer
#   DEPLOY_HOST=user@host deploy/push-artifacts.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${DEPLOY_HOST:-jan@lwenstrom.net}"
STORE="${DVC_STORE:-/opt/stacks/scanner/dvc-store}"
POINTERS=(web/public/bundle.dvc runs/ref_thumbs.dvc)
PY="$REPO/.env/bin/python"
[[ -x "$PY" ]] || PY=python3

cd "$REPO"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
# The remote uses the cache layout: files/md5/<2>/<30>, plus one .dir object per tracked directory.
"$PY" - "${POINTERS[@]}" > "$LIST" <<'EOF'
import json, pathlib, sys, yaml
cache = pathlib.Path('.dvc/cache/files/md5')
def rel(md5): return f'{md5[:2]}/{md5[2:]}'
out = set()
for ptr in sys.argv[1:]:
    md5 = yaml.safe_load(open(ptr))['outs'][0]['md5']
    out.add(rel(md5))
    if md5.endswith('.dir'):
        out.update(rel(e['md5']) for e in json.load(open(cache / rel(md5))))
missing = [r for r in out if not (cache / r).exists()]
if missing:
    sys.exit(f'{len(missing)} objects are not in the local cache: run `dvc pull` (or `dvc add`) first')
print('\n'.join(sorted(out)))
EOF
echo "== $(wc -l < "$LIST" | tr -d ' ') objects -> $HOST:$STORE"
# Group-writable so the deploy user may hard-link the objects (fs.protected_hardlinks).
rsync -a --chmod=Dg+rwxs,Fg+rw --stats --files-from="$LIST" .dvc/cache/files/md5/ "$HOST:$STORE/files/md5/"
