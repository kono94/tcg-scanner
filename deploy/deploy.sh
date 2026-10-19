#!/usr/bin/env bash
# Build the web app and upload it to the VPS as a new release, then switch `current` to it.
# Manual path; tag pushes deploy through deploy/server-deploy.sh on the VPS instead.
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
"$PY" scripts/release/build_ref_thumbnails.py

echo "== upload"
at() { ssh "$HOST" "$1"; }
DEST="$HOST:"
source "$REPO/deploy/release.sh"
publish_release web/dist datasets/reference-thumbnails "$RELEASE"
[[ -n "$DRY" ]] && exit 0
echo "live: $RELEASE (bundle $BUNDLE)"
