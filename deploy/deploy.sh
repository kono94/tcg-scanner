#!/usr/bin/env bash
# Build the web app and upload it to the VPS as a new release, then switch `current` to it.
#
#   DEPLOY_HOST=user@vps.example deploy/deploy.sh          # build + upload + switch
#   DEPLOY_HOST=... deploy/deploy.sh --dry-run             # build, show what would be uploaded
#
# Needs: the model bundle in web/public/bundle (`dvc pull web/public/bundle`), the curated
# references for thumbnails (`dvc pull datasets/curated`), node/npm, rsync, ssh access.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${DEPLOY_HOST:?set DEPLOY_HOST, e.g. jan@scanner.lwenstrom.net}"
DIR="${DEPLOY_DIR:-/opt/stacks/scanner/www}"
KEEP="${DEPLOY_KEEP:-3}"
DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"
PY="$REPO/.env/bin/python"
[[ -x "$PY" ]] || PY=python3

cd "$REPO"
[[ -f web/public/bundle/manifest.json ]] || { echo "No bundle: run 'dvc pull web/public/bundle'" >&2; exit 1; }
BUNDLE=$("$PY" -c "import json; print(json.load(open('web/public/bundle/manifest.json'))['version'])")
RELEASE="$(date -u +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- web || echo -dirty)-${BUNDLE:0:8}"

echo "== build $RELEASE"
(cd web && npm ci --no-audit --no-fund && npm run build)
"$PY" scripts/build_ref_thumbnails.py

echo "== upload"
ssh "$HOST" "mkdir -p '$DIR/releases' '$DIR/refs'"
# Unchanged files (model bundle, runtime) are hard-linked from the live release: no re-upload.
# -c compares content: Vite gives every copied public/ file a new timestamp on each build.
LINK=""
ssh "$HOST" "test -d '$DIR/current'" && LINK="--link-dest=$DIR/current/"
# Test videos for the camera mock stay local.
rsync -a -c --stats $DRY $LINK --exclude 'media/' web/dist/ "$HOST:$DIR/releases/$RELEASE/"
rsync -a --stats $DRY runs/ref_thumbs/ "$HOST:$DIR/refs/"
[[ -n "$DRY" ]] && { echo "dry run: nothing switched"; exit 0; }

echo "== switch"
# Relative link so it resolves inside the container too; mv -T replaces it atomically.
ssh "$HOST" "cd '$DIR' && ln -sfn 'releases/$RELEASE' current.new && mv -Tf current.new current \
  && ls -1dt releases/* | tail -n +$((KEEP + 1)) | xargs -r rm -rf"
echo "live: $RELEASE (bundle $BUNDLE)"
