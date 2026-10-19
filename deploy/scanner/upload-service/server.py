#!/usr/bin/env python3
"""Training uploads from the deployed scanner ("Share scans to improve recognition").

POST /api/capture?meta=<json>   body: the capture JPEG (at least one detected card)
POST /api/feedback              body: JSON with a base64 crop JPEG and the user's decision
GET  /api/health

Writes the same layout as the dev server (web/vite.config.ts), under DATA_DIR:
  captures/<UTC date>/<name>.jpg + .json      feedback/<UTC date>/<name>.jpg + .json
Automated test runs (test_run: true) go to captures-tests/ and feedback-tests/.
deploy/pull-uploads.sh copies them into datasets/sources/phone/ for training.

Public endpoint, so it is bounded: JPEG/JSON validation, size limits, a per-client hourly
rate limit and a total disk quota. Client IPs are used for rate limiting only and are never
stored. Python standard library only.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

DATA_DIR = Path(os.environ.get('DATA_DIR', '/data'))
PORT = int(os.environ.get('PORT', '8080'))
QUOTA_BYTES = int(float(os.environ.get('QUOTA_GB', '20')) * 1024 ** 3)
PER_CLIENT_PER_HOUR = int(os.environ.get('PER_CLIENT_PER_HOUR', '300'))
MAX_CAPTURE_BYTES = 20_000_000
MAX_FEEDBACK_BYTES = 8_000_000
MAX_META_BYTES = 32_000
CAPTURE_KINDS = {'scanner_capture'}
FEEDBACK_KINDS = {'user_selected_identity', 'user_rejected_all_candidates'}

_lock = threading.Lock()
_used = 0
_requests: dict[str, list[float]] = {}


def disk_usage(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


def admit(client: str, size: int) -> int | None:
    """Reserve quota and a rate-limit slot; returns an HTTP error status or None."""
    global _used
    now = time.monotonic()
    with _lock:
        recent = [t for t in _requests.get(client, []) if now - t < 3600]
        if len(recent) >= PER_CLIENT_PER_HOUR:
            _requests[client] = recent
            return 429
        if _used + size > QUOTA_BYTES:
            return 507
        recent.append(now)
        _requests[client] = recent
        _used += size
        if len(_requests) > 10_000:  # forget idle clients
            for key in [k for k, v in _requests.items() if not v or now - v[-1] > 3600]:
                del _requests[key]
    return None


def is_jpeg(data: bytes) -> bool:
    return len(data) > 4 and data[:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9'


def store(kind: str, test_run: bool, suffix: str, jpeg: bytes, meta: dict) -> None:
    now = datetime.now(timezone.utc)
    folder = DATA_DIR / (f'{kind}-tests' if test_run else kind) / now.strftime('%Y-%m-%d')
    folder.mkdir(parents=True, exist_ok=True)
    name = f'{int(now.timestamp() * 1000)}_{secrets.token_hex(4)}{suffix}'
    record = {**meta, 'image': f'{name}.jpg', 'received_at': now.isoformat(timespec='seconds'), 'origin': 'upload-service'}
    for path, payload in ((folder / f'{name}.jpg', jpeg), (folder / f'{name}.json', (json.dumps(record, indent=1) + '\n').encode())):
        tmp = path.with_suffix(path.suffix + '.part')
        tmp.write_bytes(payload)
        tmp.replace(path)


def small_int(value) -> int:
    return value if isinstance(value, int) and 0 <= value < 100_000 else 0


class Handler(BaseHTTPRequestHandler):
    server_version = 'scanner-uploads'
    sys_version = ''

    def log_message(self, fmt, *args):  # no client addresses in the logs
        print(f'{self.log_date_time_string()} {fmt % args}', flush=True)

    def address_string(self):
        return '-'

    def log_request(self, code='-', size='-'):  # path only: the capture metadata travels in the query
        self.log_message('%s %s %s', self.command, urlsplit(self.path).path, code)

    def client(self) -> str:
        # The scanner Caddy trusts the shared proxy, so the first X-Forwarded-For entry is the visitor.
        forwarded = self.headers.get('X-Forwarded-For', '')
        return forwarded.split(',')[0].strip() or self.client_address[0]

    def reply(self, status: int) -> None:
        self.send_response(status)
        self.send_header('Content-Length', '0')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def body(self, limit: int) -> bytes | None:
        try:
            length = int(self.headers.get('Content-Length', ''))
        except ValueError:
            return None
        if length <= 0 or length > limit:
            return None
        return self.rfile.read(length)

    def do_GET(self):
        self.reply(204 if urlsplit(self.path).path == '/api/health' else 404)

    def do_POST(self):
        path = urlsplit(self.path)
        try:
            if path.path == '/api/capture':
                self.capture(path.query)
            elif path.path == '/api/feedback':
                self.feedback()
            else:
                self.reply(404)
        except ValueError:  # malformed JSON or base64
            self.reply(400)
        except Exception as exc:  # never leak details to the client
            print(f'error {type(exc).__name__}: {exc}', flush=True)
            self.reply(500)

    def capture(self, query: str):
        raw = parse_qs(query).get('meta', [''])[0]
        if self.headers.get('Content-Type') != 'image/jpeg' or not raw or len(raw) > MAX_META_BYTES:
            return self.reply(400)
        meta = json.loads(raw)
        if not isinstance(meta, dict) or meta.get('kind') not in CAPTURE_KINDS:
            return self.reply(400)
        detections = meta.get('detections')
        if not isinstance(detections, list) or not detections:
            return self.reply(400)
        jpeg = self.body(MAX_CAPTURE_BYTES)
        if jpeg is None or not is_jpeg(jpeg):
            return self.reply(400)
        if (status := admit(self.client(), len(jpeg) + len(raw))) is not None:
            return self.reply(status)
        store('captures', meta.get('test_run') is True, f'_p{small_int(meta.get("page"))}', jpeg, meta)
        self.reply(204)

    def feedback(self):
        if self.headers.get('Content-Type') != 'application/json':
            return self.reply(400)
        raw = self.body(MAX_FEEDBACK_BYTES)
        if raw is None:
            return self.reply(400)
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get('kind') not in FEEDBACK_KINDS:
            return self.reply(400)
        jpeg = base64.b64decode(data.pop('crop_jpeg_base64', ''), validate=True)
        if not is_jpeg(jpeg):
            return self.reply(400)
        if (status := admit(self.client(), len(raw))) is not None:
            return self.reply(status)
        suffix = f'_p{small_int(data.get("page"))}_c{small_int(data.get("card_number"))}'
        store('feedback', data.get('test_run') is True, suffix, jpeg, data)
        self.reply(204)


def main():
    global _used
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _used = disk_usage(DATA_DIR)
    print(f'upload service on :{PORT}, {DATA_DIR} holds {_used / 1e6:.1f} MB of {QUOTA_BYTES / 1e9:.1f} GB', flush=True)
    ThreadingHTTPServer(('', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
