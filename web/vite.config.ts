import { appendFileSync, createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Connect, Plugin } from 'vite';
import { defineConfig } from 'vitest/config';

// Cross-origin isolation enables optional WASM threads; everything is same-origin.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

// Local HTTPS for phones on the LAN (camera access needs a secure context): self-signed cert in .cert/.
const certDir = resolve(dirname(fileURLToPath(import.meta.url)), '.cert');
const https = existsSync(resolve(certDir, 'cert.pem'))
  ? { cert: readFileSync(resolve(certDir, 'cert.pem')), key: readFileSync(resolve(certDir, 'key.pem')) }
  : undefined;

const curated = resolve(dirname(fileURLToPath(import.meta.url)), '../datasets/curated');
const TYPES: Record<string, string> = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' };

/** Local-only: serve catalog reference images from datasets/curated at /refs/ (not part of a deployed build). */
function referenceImages(): Plugin {
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    if (!req.url?.startsWith('/refs/')) return next();
    const rel = decodeURIComponent(req.url.slice('/refs/'.length).split('?')[0]);
    const file = resolve(curated, rel);
    const type = TYPES[extname(file).toLowerCase()];
    if (!file.startsWith(curated + sep) || !type || !existsSync(file) || !statSync(file).isFile()) {
      res.statusCode = 404;
      return res.end();
    }
    res.setHeader('Content-Type', type);
    res.setHeader('Cache-Control', 'public, max-age=86400');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    createReadStream(file).pipe(res);
  };
  return {
    name: 'reference-images',
    configureServer: (server) => { server.middlewares.use(handler); },
    configurePreviewServer: (server) => { server.middlewares.use(handler); },
  };
}

/** Dev only: phones POST breadcrumbs/errors to /__log; appended to web/.logs/client.log. */
function clientLog(): Plugin {
  const dir = resolve(dirname(fileURLToPath(import.meta.url)), '.logs');
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    if (req.url !== '/__log' || req.method !== 'POST') return next();
    let body = '';
    req.on('data', (c) => { body += c; if (body.length > 65536) req.destroy(); });
    req.on('end', () => {
      mkdirSync(dir, { recursive: true });
      appendFileSync(resolve(dir, 'client.log'), `${new Date().toISOString()} ${req.socket.remoteAddress} ${body.replace(/\n/g, ' ')}\n`);
      res.statusCode = 204;
      res.end();
    });
  };
  return {
    name: 'client-log',
    configureServer: (server) => { server.middlewares.use(handler); },
    // A changed source file makes connected phones reload (HMR full reload): log it next to the
    // client breadcrumbs so such reloads are not mistaken for crashes.
    hotUpdate({ file, type }) {
      if (this.environment.name !== 'client' || file.includes(`${sep}.logs${sep}`)) return;
      mkdirSync(dir, { recursive: true });
      appendFileSync(resolve(dir, 'client.log'), `${new Date().toISOString()} server {"event":"server-file-${type}","file":${JSON.stringify(file)}}\n`);
    },
  };
}

/** Dev only: user decisions (crop + labels) for retraining, stored under runs/feedback/<date>/. */
function feedback(): Plugin {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '../runs/feedback');
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    if (req.url !== '/__feedback' || req.method !== 'POST') return next();
    let body = '';
    req.on('data', (c) => { body += c; if (body.length > 8_000_000) req.destroy(); });
    req.on('end', () => {
      try {
        const data = JSON.parse(body) as Record<string, unknown> & { crop_jpeg_base64: string; page: number; card_number: number; test_run?: boolean };
        const day = new Date().toISOString().slice(0, 10);
        // Automated test runs pick arbitrarily: keep them out of the training folder.
        const dir = resolve(data.test_run ? `${root}-tests` : root, day);
        mkdirSync(dir, { recursive: true });
        const name = `${Date.now()}_p${Number(data.page) || 0}_c${Number(data.card_number) || 0}`;
        writeFileSync(resolve(dir, `${name}.jpg`), Buffer.from(data.crop_jpeg_base64, 'base64'));
        const { crop_jpeg_base64: _omit, ...meta } = data;
        void _omit;
        writeFileSync(resolve(dir, `${name}.json`), JSON.stringify({ ...meta, image: `${name}.jpg`, client: req.socket.remoteAddress }, null, 1));
        res.statusCode = 204;
      } catch {
        res.statusCode = 400;
      }
      res.end();
    });
  };
  return { name: 'feedback', configureServer: (server) => { server.middlewares.use(handler); } };
}

/**
 * Dev only: full captures (JPEG) with their detections, for detector/recognizer training, when the
 * "Save captures for training" setting is on. Stored under runs/captures/<date>/ (test runs under
 * runs/captures-tests/). Body: the JPEG; metadata JSON in the `meta` query parameter.
 */
function captures(): Plugin {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '../runs/captures');
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    if (!req.url?.startsWith('/__capture') || req.method !== 'POST') return next();
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (c: Buffer) => { size += c.length; if (size > 20_000_000) req.destroy(); else chunks.push(c); });
    req.on('end', () => {
      try {
        const meta = JSON.parse(new URL(req.url!, 'http://x').searchParams.get('meta') ?? '') as Record<string, unknown> & { test_run?: boolean; page?: number };
        const jpeg = Buffer.concat(chunks);
        if (jpeg.length < 4 || jpeg[0] !== 0xff || jpeg[1] !== 0xd8) throw new Error('not a JPEG');
        const day = new Date().toISOString().slice(0, 10);
        const dir = resolve(meta.test_run ? `${root}-tests` : root, day);
        mkdirSync(dir, { recursive: true });
        const name = `${Date.now()}_p${Number(meta.page) || 0}`;
        writeFileSync(resolve(dir, `${name}.jpg`), jpeg);
        writeFileSync(resolve(dir, `${name}.json`), JSON.stringify({ ...meta, image: `${name}.jpg`, client: req.socket.remoteAddress }, null, 1));
        res.statusCode = 204;
      } catch {
        res.statusCode = 400;
      }
      res.end();
    });
  };
  return { name: 'captures', configureServer: (server) => { server.middlewares.use(handler); } };
}

export default defineConfig({
  plugins: [referenceImages(), clientLog(), feedback(), captures()],
  server: { headers: isolation, host: '0.0.0.0', https },
  preview: { headers: isolation, host: '0.0.0.0', https },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022' },
  test: { environment: 'node', include: ['tests/**/*.test.ts'], testTimeout: 60000 },
});

