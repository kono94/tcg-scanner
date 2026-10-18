#!/usr/bin/env bash
# Build and publish the checked-out tag on the VPS. Run by deploy-trigger after a tag push:
#
#   /usr/local/bin/deploy-trigger scanner   (checks out the tag in /opt/scanner, then runs this)
#
# The model bundle and thumbnails come from the VPS DVC store (/opt/dvc-store), filled from the
# Mac with `dvc push -r vps`. Node and DVC run in containers; the host only needs Docker and rsync.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${DEPLOY_DIR:-/opt/stacks/scanner/www}"
KEEP="${DEPLOY_KEEP:-3}"
DRY=""
NODE_IMAGE="node:24-alpine"
DVC_VERSION="3.67.1"
DVC_IMAGE="scanner-dvc:$DVC_VERSION"
USER_ARGS=(-u "$(id -u):$(id -g)" -e HOME=/tmp)

cd "$REPO"
TAG="$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)"

echo "== artifacts"
docker image inspect "$DVC_IMAGE" >/dev/null 2>&1 || docker build -q -t "$DVC_IMAGE" - <<EOF
FROM python:3.12-slim
RUN pip install --no-cache-dir "dvc==$DVC_VERSION"
EOF
# config.local (server only) points the `vps` remote at the local store and hard-links checkouts.
docker run --rm "${USER_ARGS[@]}" -v "$REPO:$REPO" -v /opt/dvc-store:/opt/dvc-store -w "$REPO" "$DVC_IMAGE" \
  dvc pull -r vps web/public/bundle.dvc runs/ref_thumbs.dvc \
  || { echo "Artifacts for $TAG are missing on the VPS: run deploy/push-artifacts.sh on the Mac." >&2; exit 1; }
BUNDLE=$(docker run --rm -v "$REPO/web/public/bundle:/b:ro" "$DVC_IMAGE" \
  python -c "import json; print(json.load(open('/b/manifest.json'))['version'])")
RELEASE="$(date -u +%Y%m%d-%H%M%S)-$TAG-${BUNDLE:0:8}"

echo "== build $RELEASE"
docker run --rm "${USER_ARGS[@]}" -e npm_config_cache=/tmp/.npm -v "$REPO:$REPO" -w "$REPO/web" "$NODE_IMAGE" \
  sh -c "npm ci --no-audit --no-fund && npm run build"

echo "== publish"
at() { sh -c "$1"; }
DEST=""
source "$REPO/deploy/release.sh"
publish_release web/dist runs/ref_thumbs "$RELEASE"
echo "live: $RELEASE (bundle $BUNDLE)"
