#!/usr/bin/env bash
# Copy training uploads from the VPS into datasets/sources/phone/ (captures/ and feedback/, by UTC date).
# Test-run uploads (*-tests) stay on the server. Files are never deleted on either side.
#   DEPLOY_HOST=user@VPS deploy/pull-uploads.sh
#   then: dvc add datasets/sources/phone && dvc push && git commit datasets/sources/phone.dvc
set -euo pipefail
: "${DEPLOY_HOST:?set DEPLOY_HOST=user@host}"
REMOTE_DIR=${REMOTE_DIR:-/opt/stacks/scanner/uploads}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for kind in captures feedback; do
  mkdir -p "$ROOT/datasets/sources/phone/$kind"
  rsync -a --ignore-existing --exclude '*.part' "$DEPLOY_HOST:$REMOTE_DIR/$kind/" "$ROOT/datasets/sources/phone/$kind/"
done
find "$ROOT/datasets/sources/phone" -name '*.json' -newer "$ROOT/datasets/sources/phone.dvc" | wc -l | xargs echo "new records since the last dvc add:"
