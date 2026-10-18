# Hosting on the VPS

The scanner is a static site. Recognition runs in the visitor's browser; the server only
delivers the app, ONNX runtime, versioned model bundle and catalog thumbnails (loaded
as needed). Bundle `27047332` contains a 46.0 MB raw / 38.6 MB gzip detector; this is
only the detector, and encoders, gallery, catalog and runtime add download size. Actual
transfer size depends on compression, execution provider and browser cache.

One shared Caddy proxy serves every subdomain on the VPS with automatic HTTPS
(Let's Encrypt). Each app runs in its own Docker Compose stack, joins the `web` network
and publishes no ports itself.

```text
internet ──443──> proxy (Caddy, certificates)
                    ├── scanner.lwenstrom.net ──> scanner:80  (Caddy, static files)
                    └── tft.lwenstrom.net     ──> tft:<port>  (the TFT compose app)
```

| Path on the VPS | From this repo | Purpose |
| --- | --- | --- |
| `/opt/stacks/proxy/` | `deploy/proxy/` | shared HTTPS proxy, one Caddyfile entry per domain |
| `/opt/stacks/scanner/` | `deploy/scanner/` | static file server for the scanner |
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
2. It SSHes to `github-deployer@VPS -p 2222`. That user's key is pinned in `authorized_keys` to
   `command="/usr/local/bin/deploy-trigger scanner",restrict`, so it can only choose the tag.
3. `deploy-trigger` fetches, checks out the tag in `/opt/stacks/scanner/src` and runs `deploy/server-deploy.sh`.
   That script hard-links the bundle and thumbnails named by the `.dvc` pointers out of the store
   (`deploy/checkout-artifacts.py`, no DVC on the server), runs `npm ci && npm run build`
   in `node:24-alpine`, and the same release/switch steps as `deploy.sh` (`deploy/release.sh`).

The model bundle and thumbnails live in DVC, not git, and the LAN DVC store (`gpu-pc`) can't be
reached from GitHub or the VPS. **Before tagging a commit that changes `web/public/bundle.dvc` or
`runs/ref_thumbs.dvc`, push the artifacts to the VPS store from the Mac:**

```bash
dvc push web/public/bundle.dvc runs/ref_thumbs.dvc   # LAN store, as before
deploy/push-artifacts.sh                             # VPS store for tag deploys (one rsync)
git tag v1.2.3 && git push origin v1.2.3
```

`dvc push -r vps ...` gives the same result, but takes hours for 24k thumbnails over SFTP.

If they're missing, the deploy stops before building and the live release stays unchanged. Thumbnails
are rebuilt with `scripts/build_ref_thumbnails.py` and tracked with `dvc add runs/ref_thumbs`. The store
uses DVC's cache layout, so it is also the `vps` DVC remote. Checkout, store and `www/` share one parent
so the hard links work inside one container mount.
GitHub secrets: `SERVER_IP` (173.249.10.6) and `SSH_PRIVATE_KEY` (`~/.ssh/id_scanner_deploy` on the Mac).
Manual trigger: `ssh -i ~/.ssh/id_scanner_deploy -p 2222 github-deployer@lwenstrom.net trigger-deploy <tag>`.

## Deploying the scanner manually

From the repository root on the Mac (bypasses git tags, uploads straight from the working tree):

```bash
dvc pull web/public/bundle datasets/curated     # bundle + reference images (once per change)
DEPLOY_HOST=user@VPS deploy/deploy.sh           # build, upload, switch
DEPLOY_HOST=user@VPS deploy/deploy.sh --dry-run # build and show what would be uploaded
```

`deploy.sh` runs `npm ci && npm run build`, builds missing thumbnails
(`scripts/build_ref_thumbnails.py`), uploads a new release next to the live one, then
switches the `current` symlink in one step. Files that did not change (usually the whole
model bundle) are hard-linked from the previous release instead of uploaded again. The
last three releases are kept (`DEPLOY_KEEP`). Test videos (`web/public/media`) are not
deployed.

**Rollback:** on the VPS, `cd /opt/stacks/scanner/www && ls releases` and
`ln -sfn releases/<older> current.new && mv -Tf current.new current`.

**New prices or models:** rebuild the bundle (`scripts/update_prices.py`,
`scripts/build_browser_bundle.py`), check it locally, `dvc add web/public/bundle && dvc push`,
commit the `.dvc` file, deploy. Visitors get the new bundle on their next visit; the app
checks `bundle/manifest.json` on every start.

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

- No `/__feedback` or `/__log`: corrections made in the app are not collected as training
  examples, and phone errors are not logged on the server. Both are dev-server features.
- Saved collections live in each visitor's browser (localStorage + IndexedDB): no accounts,
  no sync between devices, and clearing site data deletes them.
- Thresholds are uncalibrated; the detector combines synthetic pretraining with
  AI-reviewed real-corner fine tuning. Physical-phone performance remains unverified
  (see [browser evidence](../docs/BROWSER_BUNDLE_2026-10-17.md)); label the site as a pilot.

## This VPS (CentOS 7)

The host is CentOS 7 (EOL) with SELinux enforcing, set up 2026-10-18: docker-ce 26.1.4 and the
compose plugin 2.27.1 from the docker-ce `centos/7` repo. Docker's SELinux integration stays on, so:
`/opt/stacks` carries `container_file_t` (`semanage fcontext`) and the
compose bind mounts use `:z`. `jan` is SELinux `staff_u`: admin commands need
`sudo -r sysadm_r -t sysadm_t`. firewalld opens 80/tcp, 443/tcp+udp and 2222/tcp; Docker-published
ports bypass firewalld. sshd listens on 22 and 2222; `github-deployer` is in `AllowUsers` with a
`Match` block that disables forwarding and TTYs. Git on the host is 1.8, so `deploy-trigger` avoids newer flags.

## Tested

2026-10-18, on the real VPS: Caddy got Let's Encrypt certificates for `scanner.lwenstrom.net` and
`tft.lwenstrom.net`. Tag `v1.2.2` deployed through GitHub Actions → `deploy-trigger scanner` →
`server-deploy.sh`: artifact checkout, container build and release switch together took under a
minute; 24,040 thumbnails were copied on the first run. Checked over HTTPS: COOP/COEP, `no-cache` HTML,
immutable `/assets/`, `application/wasm` for `/ort/`, bundle `270473320b53d5b4`, a `/refs/` WebP with
week-long caching, HTTP→HTTPS redirect. Not yet checked there: the page in a browser (cross-origin
isolation, model load) and any physical phone.

Earlier:

2026-10-17: the full flow (`deploy.sh` → release folder → `current` switch → `scanner`
container behind the `web` network) was run against Docker on the LAN GPU PC: headers,
caching, thumbnails, WASM MIME type, a second deploy with zero re-upload, and the production
build booting in Chrome with cross-origin isolation and loading the models. The proxy
Caddyfile passes `caddy validate`. Certificates and the real VPS are not tested yet.
