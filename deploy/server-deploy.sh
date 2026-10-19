#!/usr/bin/env bash
# Build and publish the checked-out tag on the VPS. Run by deploy-trigger after a tag push:
#
#   /usr/local/bin/deploy-trigger scanner   (checks out the tag in /opt/stacks/scanner/src, runs this)
#
# The model bundle and thumbnails come from the VPS DVC store (/opt/stacks/scanner/dvc-store), filled
# from the Mac with deploy/push-artifacts.sh. Node and Python run in containers; the host only needs
# Docker and rsync. Checkout, store and www share one parent so hard links work inside one mount.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${DEPLOY_DIR:-/opt/stacks/scanner/www}"
KEEP="${DEPLOY_KEEP:-3}"
DRY=""
NODE_IMAGE="node:24-alpine"
PY_IMAGE="python:3.12-alpine"
STORE="${DVC_STORE:-/opt/stacks/scanner/dvc-store}"
USER_ARGS=(-u "$(id -u):$(id -g)" -e HOME=/tmp)

cd "$REPO"
TAG="$(git describe --tags --exact-match 2>/dev/null || git rev-parse --short HEAD)"

echo "== artifacts"
# Hard-link the bundle and thumbnails named by the .dvc pointers out of the store (no DVC needed).
ROOT="$(dirname "$REPO")"   # one mount holding both the checkout and the store
docker run --rm "${USER_ARGS[@]}" -v "$ROOT:$ROOT" -w "$REPO" "$PY_IMAGE" \
  python deploy/checkout-artifacts.py "$STORE" web/public/bundle.dvc datasets/reference-thumbnails.dvc
BUNDLE=$(sed -n 's/^ *"version": *"\([^"]*\)".*/\1/p' web/public/bundle/manifest.json | head -n 1)
RELEASE="$(date -u +%Y%m%d-%H%M%S)-$TAG-${BUNDLE:0:8}"

echo "== build $RELEASE"
docker run --rm "${USER_ARGS[@]}" -e npm_config_cache=/tmp/.npm -v "$REPO:$REPO" -w "$REPO/web" "$NODE_IMAGE" \
  sh -c "npm ci --no-audit --no-fund && npm run build"

echo "== publish"
at() { sh -c "$1"; }
DEST=""
source "$REPO/deploy/release.sh"
publish_release web/dist datasets/reference-thumbnails "$RELEASE"
echo "live: $RELEASE (bundle $BUNDLE)"
