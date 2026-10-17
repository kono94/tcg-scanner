# Hosting on the VPS

The scanner is a static site. Recognition runs in the visitor's browser; the server only
delivers files: the app, the ONNX runtime and model bundle (~220 MB on disk; a phone
downloads roughly 80 MB of it once, a desktop about 150 MB) and catalog thumbnails (~530 MB, loaded as needed).

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

## Deploying the scanner

From the repository root on the Mac:

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

2. In `/opt/stacks/proxy/Caddyfile`, uncomment the `tft.lwenstrom.net` block and set the
   service name and internal port (`reverse_proxy tft:3000`).
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
- Thresholds are uncalibrated and the detector is trained on synthetic scenes (see
  `web/README.md`); label the site as a pilot.

## Tested

2026-10-17: the full flow (`deploy.sh` → release folder → `current` switch → `scanner`
container behind the `web` network) was run against Docker on the LAN GPU PC: headers,
caching, thumbnails, WASM MIME type, a second deploy with zero re-upload, and the production
build booting in Chrome with cross-origin isolation and loading the models. The proxy
Caddyfile passes `caddy validate`. Certificates and the real VPS are not tested yet.
