# Hosting on the VPS

The scanner is a static site. Recognition runs in the visitor's browser; the server
delivers the app, ONNX runtime, versioned model bundle and catalog thumbnails (loaded
as needed), and a small upload service stores training uploads (captures and corrections)
when visitors leave "Share scans to improve recognition" on. Bundle `27047332` contains a 46.0 MB raw / 38.6 MB gzip detector; this is
only the detector, and encoders, gallery, catalog and runtime add download size. Actual
transfer size depends on compression, execution provider and browser cache.

One shared Caddy proxy serves every subdomain on the VPS with automatic HTTPS
(Let's Encrypt). Each app runs in its own Docker Compose stack, joins the `web` network
and publishes no ports itself.

```text
internet ──443──> proxy (Caddy, certificates)
                    ├── scanner.lwenstrom.net ──> scanner:80  (Caddy, static files)
                    │                               └── /api/* ──> uploads:8080 (training uploads)
                    └── tft.lwenstrom.net     ──> tft:<port>  (the TFT compose app)
```

| Path on the VPS | From this repo | Purpose |
| --- | --- | --- |
| `/opt/stacks/proxy/` | `deploy/proxy/` | shared HTTPS proxy, one Caddyfile entry per domain |
| `/opt/stacks/scanner/` | `deploy/scanner/` | static file server + upload service for the scanner |
| `/opt/stacks/scanner/uploads/` | upload service | training uploads: `captures/`, `feedback/` by UTC date |
| `/opt/stacks/scanner/www/releases/<release>/` | `deploy/deploy.sh` | built app + bundle, one folder per deploy |
| `/opt/stacks/scanner/www/current` | `deploy/deploy.sh` | symlink to the live release |
| `/opt/stacks/scanner/www/refs/` | `deploy/deploy.sh` | catalog thumbnails, shared by releases |
| `/opt/stacks/scanner/src/` | git clone (owner `github-deployer`) | tag checkout that `server-deploy.sh` builds |
| `/opt/stacks/scanner/dvc-store/` | `deploy/push-artifacts.sh` | DVC remote holding bundle + thumbnails for tag deploys |
| `/usr/local/bin/deploy-trigger` | `deploy/deploy-trigger` | GitOps entry point, one project per deploy key |

## One-time setup

1. **DNS:** `A` (and `AAAA` if the VPS has IPv6) records for `scanner.lwenstrom.net` and
   `tft.lwenstrom.net` pointing to the VPS. Certificates are only issued once these resolve.
2. **Firewall:** allow TCP 22, 80, 443 and UDP 443 (HTTP/3).
3. **Docker** with the compose plugin ([docs.docker.com/engine/install](https://docs.docker.com/engine/install/)).
4. **Shared network and proxy:**

   ```bash
   docker network create web
   sudo mkdir -p /opt/stacks && sudo chown "$USER" /opt/stacks
   # from the Mac:
   rsync -a deploy/proxy deploy/scanner user@VPS:/opt/stacks/
   # on the VPS:
   cd /opt/stacks/proxy && cp .env.example .env && nano .env   # ACME_EMAIL=...
   docker compose up -d
   cd /opt/stacks/scanner && mkdir -p www && docker compose up -d
   ```

5. **First deploy** (from the Mac, see below). Until then the scanner answers 404.

## Deploying the scanner (GitOps: push a tag)

A pushed tag on this repo deploys **only the scanner**:

1. `.github/workflows/deploy.yml` runs `npm ci`, `npm run typecheck` and `npm test` (bundle parity
   tests skip without the DVC bundle).
2. It SSHes to `github-deployer@VPS`. That user's key is pinned in `authorized_keys` to
   `command="/usr/local/bin/deploy-trigger scanner",restrict`, so it can only choose the tag.
3. `deploy-trigger` fetches, checks out the tag in `/opt/stacks/scanner/src` and runs `deploy/server-deploy.sh`.
   That script hard-links the bundle and thumbnails named by the `.dvc` pointers out of the store
   (`deploy/checkout-artifacts.py`, no DVC on the server), runs `npm ci && npm run build`
   in `node:24-alpine`, and the same release/switch steps as `deploy.sh` (`deploy/release.sh`).

The model bundle and thumbnails live in DVC, not git, and the LAN DVC store (`gpu-pc`) can't be
reached from GitHub or the VPS. **Before tagging a commit that changes `web/public/bundle.dvc` or
`datasets/reference-thumbnails.dvc`, push the artifacts to the VPS store from the Mac:**

```bash
dvc push web/public/bundle.dvc datasets/reference-thumbnails.dvc   # LAN store, as before
DEPLOY_HOST=user@VPS deploy/push-artifacts.sh       # VPS store for tag deploys (one rsync)
git tag v1.2.3 && git push origin v1.2.3
```

`dvc push -r vps ...` gives the same result, but takes hours for 24k thumbnails over SFTP.

If they're missing, the deploy stops before building and the live release stays unchanged. Thumbnails
are rebuilt with `scripts/release/build_ref_thumbnails.py` and tracked with `dvc add datasets/reference-thumbnails`. The store
uses DVC's cache layout, so it can also serve as a DVC remote, configured per machine
(`dvc remote add --local vps ssh://user@VPS/opt/stacks/scanner/dvc-store`). Checkout, store and `www/` share one parent
so the hard links work inside one container mount.
GitHub secrets: `SERVER_IP` (the VPS address) and `SSH_PRIVATE_KEY` (`~/.ssh/id_scanner_deploy` on the Mac).
Manual trigger: `ssh -i ~/.ssh/id_scanner_deploy github-deployer@VPS trigger-deploy <tag>`.

## Deploying the scanner manually

From the repository root on the Mac (bypasses git tags, uploads straight from the working tree):

```bash
dvc pull web/public/bundle datasets/curated     # bundle + reference images (once per change)
DEPLOY_HOST=user@VPS deploy/deploy.sh           # build, upload, switch
DEPLOY_HOST=user@VPS deploy/deploy.sh --dry-run # build and show what would be uploaded
```

`deploy.sh` runs `npm ci && npm run build`, builds missing thumbnails
(`scripts/release/build_ref_thumbnails.py`), uploads a new release next to the live one, then
switches the `current` symlink in one step. Files that did not change (usually the whole
model bundle) are hard-linked from the previous release instead of uploaded again. The
last three releases are kept (`DEPLOY_KEEP`). Test videos (`web/public/media`) are not
deployed.

**Rollback:** on the VPS, `cd /opt/stacks/scanner/www && ls releases` and
`ln -sfn releases/<older> current.new && mv -Tf current.new current`.

**New prices or models:** rebuild the bundle (`scripts/release/update_prices.py`,
`scripts/release/build_browser_bundle.py`), check it locally, `dvc add web/public/bundle && dvc push`,
commit the `.dvc` file, deploy. Visitors get the new bundle on their next visit; the app
checks `bundle/manifest.json` on every start.

## Training uploads

"Share scans to improve recognition" is on by default in the app and can be turned off in
Settings. With it on, each capture with at least one detected card (`/api/capture`) and each
correction (`/api/feedback`) is POSTed to the scanner origin. The scanner Caddy forwards
`/api/*` to the `uploads` service (`scanner/upload-service/server.py`, stock
`python:3.12-alpine`, no build), which validates JPEG/JSON, limits sizes (25 MB per request),
allows 300 uploads per visitor per hour (`PER_CLIENT_PER_HOUR`) and stops accepting at 20 GB
(`QUOTA_GB`, answers 507). It writes `uploads/{captures,feedback}/<UTC date>/<name>.jpg + .json`;
automated test runs go to `*-tests/`. Visitor IPs are only used for the rate limit and are
not stored. Caddy cannot write request bodies itself, hence the service.

One-time setup (and after changing `deploy/scanner/`), on the VPS:

```bash
# from the Mac:
rsync -a deploy/scanner/ user@VPS:/opt/stacks/scanner/ --exclude www --exclude uploads
# on the VPS:
cd /opt/stacks/scanner && mkdir -p uploads && sudo chown 65534:65534 uploads
docker compose up -d                                   # starts `uploads`, reloads the scanner Caddy
curl -sI https://scanner.lwenstrom.net/api/health      # 204
```

Bring uploads into the dataset (from the Mac; never deletes anything):

```bash
DEPLOY_HOST=user@VPS deploy/pull-uploads.sh
dvc add datasets/sources/phone && dvc push && git commit -m "Pull scanner uploads" datasets/sources/phone.dvc
```

Uploaded detections are unverified proposals and user corrections are user decisions; both go
through the normal review before they become training labels ([VIDEO_MINING.md](../docs/VIDEO_MINING.md),
step 5).

## Adding the TFT app (or any other subdomain)

1. In the TFT `compose.yaml`, attach the web-facing service to the shared network and drop
   its public port mapping (the proxy reaches it by service name):

   ```yaml
   services:
     tft:                 # the service that serves HTTP
       # ports: ["3000:3000"]   <- remove
       networks: [default, web]
   networks:
     web:
       external: true
   ```

2. `/opt/stacks/proxy/Caddyfile` already has the `tft.lwenstrom.net` block
   (`reverse_proxy tft-nginx:80`). Until TFT runs there, that host answers 502; its certificate is issued.
3. Reload without downtime: `docker compose -f /opt/stacks/proxy/compose.yaml exec caddy caddy reload --config /etc/caddy/Caddyfile`.

## What the scanner needs from the server

- **HTTPS**: browsers only allow camera access on secure origins.
- **Cross-origin isolation** headers (`COOP`/`COEP`, set in `scanner/Caddyfile`): enable
  multi-threaded WASM. Every file is same-origin, so nothing else needs CORS.
- **Caching**: `index.html`, `bundle/` and `ort/` revalidate on every start (cheap `304`s);
  Vite's hashed `assets/` are immutable; thumbnails are cached for a week.

## Limits of a deployment (vs. the dev server)

- No `/__log`: phone errors are not logged on the server (dev-server feature). Captures and
  corrections do reach the server through the upload service (below).
- Saved collections live in each visitor's browser (localStorage + IndexedDB): no accounts,
  no sync between devices, and clearing site data deletes them.
- Thresholds are uncalibrated; the detector combines synthetic pretraining with
  AI-reviewed real-corner fine tuning. Physical-phone performance remains unverified
  (see [browser evidence](../docs/BROWSER_BUNDLE.md)); show these limits on the site.

## Tested

2026-10-19, on the production host: Caddy got Let's Encrypt certificates for both domains. Tag
`v1.2.3` deployed through GitHub Actions → `deploy-trigger scanner` → `server-deploy.sh`. Checked over
HTTPS: COOP/COEP, `no-cache` HTML, bundle `270473320b53d5b4`, `/api/health` 204, HTTP→HTTPS redirect.
Not yet checked there: the page in a browser (cross-origin isolation, model load) and any physical phone.

Earlier:

2026-10-17: the full flow (`deploy.sh` → release folder → `current` switch → `scanner`
container behind the `web` network) was run against Docker on the LAN GPU PC: headers,
caching, thumbnails, WASM MIME type, a second deploy with zero re-upload, and the production
build booting in Chrome with cross-origin isolation and loading the models. The proxy
Caddyfile passes `caddy validate`. Certificates and the real VPS are not tested yet.
