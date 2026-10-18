# Shared release steps, sourced by deploy.sh (Mac -> VPS) and server-deploy.sh (on the VPS).
#
# Callers set: DIR (www root on the VPS), KEEP, DRY ("" or --dry-run), and
#   at   <shell command>  runs a command where DIR lives (ssh to the VPS, or a local shell)
#   DEST                  rsync destination prefix ("user@host:" or "" for local)

# publish_release <dist dir> <thumbnail dir> <release name>
publish_release() {
  local dist="$1" thumbs="$2" release="$3" link=""
  at "mkdir -p '$DIR/releases' '$DIR/refs'"
  # Unchanged files (model bundle, runtime) are hard-linked from the live release: no re-upload.
  # -c compares content: Vite gives every copied public/ file a new timestamp on each build.
  at "test -d '$DIR/current'" && link="--link-dest=$DIR/current/"
  # Content only (-rlt, no owner/group/perms/dir times): www/ is shared by the deploy user (tag
  # deploys) and jan (manual deploys) through the setgid `deployers` group.
  local copy=(-rlt --omit-dir-times --stats)
  [[ -z "$DRY" ]] || copy+=("$DRY")
  # Test videos for the camera mock stay local.
  rsync "${copy[@]}" -c $link --exclude 'media/' "$dist/" "$DEST$DIR/releases/$release/"
  rsync "${copy[@]}" "$thumbs/" "$DEST$DIR/refs/"
  [[ -n "$DRY" ]] && { echo "dry run: nothing switched"; return 0; }

  echo "== switch"
  # Relative link so it resolves inside the container too; mv -T replaces it atomically.
  at "cd '$DIR' && ln -sfn 'releases/$release' current.new && mv -Tf current.new current \
    && ls -1dt releases/* | tail -n +$((KEEP + 1)) | xargs -r rm -rf"
}
