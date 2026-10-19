#!/usr/bin/env python3
"""Sync the full English TCGdex Pokémon catalog and a bounded image gallery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from PIL import Image, UnidentifiedImageError


API_BASE = "https://api.tcgdex.net/v2"
ASSET_HOST = "assets.tcgdex.net"
ASSETS_DOC = "https://tcgdex.dev/assets"
USER_AGENT = "tcg-scanner-pokemon-gallery-sync/1.0"
TIMEOUT = (8, 25)
ATTEMPTS = 2
CHUNK_SIZE = 64 * 1024
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")
ALLOWED_ROOT_NAMES = {"catalog.json", "catalog_source.json", "pocket_series_source.json", "manifest.json", "sync_state.json", "progress.jsonl", "cards"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_segment(value: object, label: str) -> str:
    text = str(value) if value is not None else ""
    if not text or not SAFE_SEGMENT.fullmatch(text) or text in {".", ".."}:
        raise ValueError(f"Unsafe {label}: {text!r}")
    return text


def safe_catalog_id(value: object) -> str:
    """Validate an opaque provider ID; punctuation is percent-encoded for paths."""
    text = str(value) if value is not None else ""
    if not text or text in {".", ".."} or any(ch in text for ch in ("/", "\\", "?", "#")):
        raise ValueError(f"Unsafe catalog id: {text!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"Unsafe catalog id: {text!r}")
    return text


def filename_component(value: str) -> str:
    return quote(value, safe="-_")


def parse_set_id(card_id: str, local_id: object) -> str | None:
    """Return a set prefix only when id ends in the exact supplied localId."""
    if not isinstance(local_id, (str, int)) or isinstance(local_id, bool):
        return None
    local_text = str(local_id)
    if not local_text:
        return None
    match = re.fullmatch(rf"(.+)-{re.escape(local_text)}", card_id)
    if not match:
        return None
    prefix = match.group(1)
    if not SAFE_SEGMENT.fullmatch(prefix):
        return None
    return prefix


def normalize_catalog(payload: Any, language: str = "en") -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("TCGdex cards endpoint did not return a JSON array")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("TCGdex catalog contains a non-object entry")
        card_id = safe_catalog_id(item.get("id"))
        if card_id in seen:
            raise ValueError(f"Duplicate TCGdex catalog id: {card_id}")
        seen.add(card_id)
        local_id = item.get("localId")
        set_id = parse_set_id(card_id, local_id)
        if set_id is not None:
            safe_segment(set_id, "set id")
        raw_image = item.get("image")
        image_base = None
        if isinstance(raw_image, str) and raw_image:
            parsed = urlparse(raw_image)
            if parsed.scheme == "https" and parsed.netloc == ASSET_HOST:
                image_base = raw_image.rstrip("/")
        records.append({
            "provider": "tcgdex",
            "game": "pokemon",
            "language": language,
            "provider_id": card_id,
            "catalog_id": card_id,
            "catalog_identity": f"tcgdex:{language}:{card_id}",
            "set_id": set_id,
            "local_id": str(local_id) if local_id is not None else None,
            "local_number": str(local_id) if local_id is not None else None,
            "official_total": None,
            "name": item.get("name"),
            "category": item.get("category"),
            "image_base_url": image_base,
            "image_url": f"{image_base}/high.webp" if image_base else None,
            "image_status": "pending" if image_base else "unavailable",
            "physical_finish": "unresolved",
            "physical_edition": "unresolved",
            "catalog_record": item,
            "source_catalog_url": f"{API_BASE}/{language}/cards",
            "image_sha256": None,
            "image_width": None,
            "image_height": None,
            "image_bytes": None,
            "image_path": None,
        })
    return records


def fair_round_robin(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for record in records:
        if record["image_status"] == "pending":
            groups[record["set_id"] or "unknown-set"].append(record)
    queues = deque(groups[key] for key in sorted(groups))
    result: list[dict[str, Any]] = []
    while queues:
        queue = queues.popleft()
        result.append(queue.popleft())
        if queue:
            queues.append(queue)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_webp(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            if image.format != "WEBP":
                raise ValueError(f"Expected WEBP bytes; received {image.format or 'unknown'}")
            size = image.size
            image.verify()
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("Image has invalid dimensions")
        return size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("TCGdex response is not a valid WEBP image") from exc


def output_layout(output: Path) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    unexpected = [p.name for p in output.iterdir() if p.name not in ALLOWED_ROOT_NAMES]
    if unexpected:
        raise ValueError(f"Output contains unexpected files; refusing to overwrite: {unexpected}")
    cards = output / "cards"
    if cards.is_symlink():
        raise ValueError("Output cards directory must not be a symlink")
    cards.mkdir(exist_ok=True)


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def request_json(url: str, session: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < ATTEMPTS:
                time.sleep(0.25)
    raise RuntimeError(f"Catalog request failed: {url}: {last_error}") from last_error


def request_json_source(url: str, session: Any) -> tuple[Any, bytes]:
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            response.raise_for_status()
            body = response.content
            return json.loads(body), body
        except Exception as exc:
            last_error = exc
            if attempt + 1 < ATTEMPTS:
                time.sleep(0.25)
    raise RuntimeError(f"Catalog request failed: {url}: {last_error}") from last_error


def classify_domains(records: list[dict[str, Any]], pocket_sets: set[str]) -> None:
    for record in records:
        set_id = record.get("set_id")
        image_path = urlparse(record.get("image_base_url") or "").path.split("/")
        if set_id and set_id in pocket_sets:
            record["domain"] = "pokemon_tcg_pocket_digital"
            record["domain_evidence"] = "set id is listed in the TCGdex tcgp series endpoint"
        elif set_id and "tcgp" not in image_path:
            record["domain"] = "pokemon_physical_tcg"
            record["domain_evidence"] = "catalog set id is parseable and absent from the TCGdex tcgp series set list"
        else:
            record["domain"] = "unknown"
            record["domain_evidence"] = "insufficient provider series/set evidence"


class BudgetExceeded(RuntimeError):
    pass


class DiskFloorExceeded(RuntimeError):
    pass


class DownloadBudget:
    """Concurrent reservations keep downloaded bytes and disk use below hard limits."""
    def __init__(self, byte_limit: int, output: Path, floor_bytes: int, used: int):
        self.byte_limit = byte_limit
        self.output = output
        self.floor_bytes = floor_bytes
        self.used = used
        self.reserved_bytes = 0
        self.lock = threading.Lock()

    def reserve(self, amount: int) -> None:
        with self.lock:
            if amount <= 0 or self.used + self.reserved_bytes + amount > self.byte_limit:
                raise BudgetExceeded("image Content-Length exceeds remaining cumulative byte budget")
            free = shutil.disk_usage(self.output).free
            if free - self.reserved_bytes - amount < self.floor_bytes:
                raise DiskFloorExceeded("image reservation would cross the configured free-space floor")
            self.reserved_bytes += amount

    def release(self, amount: int, *, used: int = 0) -> None:
        with self.lock:
            self.reserved_bytes -= amount
            self.used += used


def download_webp(url: str, destination: Path, session: Any, budget: DownloadBudget) -> tuple[int, str, int, int]:
    """Stream a validated WEBP after reserving its declared bytes and disk space."""
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        temp_path: Path | None = None
        reserved = 0
        try:
            response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, stream=True)
            response.raise_for_status()
            content_type = str(response.headers.get("content-type", "")).lower()
            if content_type and "image/" not in content_type and "octet-stream" not in content_type:
                raise ValueError(f"Image response has unexpected content type: {content_type}")
            length = response.headers.get("content-length")
            if length is None:
                raise ValueError("Image response lacks Content-Length; safe byte and disk reservation is unavailable")
            reserved = int(length)
            budget.reserve(reserved)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".download", dir=destination.parent)
            temp_path = Path(name)
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(fd, "wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > reserved:
                        raise ValueError("Streamed response exceeded declared Content-Length")
                    handle.write(chunk)
                    digest.update(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            width, height = validate_webp(temp_path)
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing image: {destination}")
            os.replace(temp_path, destination)
            temp_path = None
            budget.release(reserved, used=size)
            reserved = 0
            return size, digest.hexdigest(), width, height
        except (BudgetExceeded, DiskFloorExceeded):
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < ATTEMPTS:
                time.sleep(0.25)
        finally:
            if reserved:
                budget.release(reserved)
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            close = getattr(locals().get("response"), "close", None)
            if callable(close):
                close()
    raise RuntimeError(f"Image request failed: {url}: {last_error}") from last_error


def card_destination(output: Path, record: dict[str, Any]) -> Path:
    provider_id = safe_catalog_id(record["provider_id"])
    set_id = safe_segment(record["set_id"] or "unknown-set", "set id")
    return output / "cards" / set_id / f"{filename_component(provider_id)}.webp"


def build_manifest(records: list[dict[str, Any]], state: dict[str, Any], *, complete: bool,
                   stopped_reason: str | None, args: argparse.Namespace, catalog_url: str) -> dict[str, Any]:
    downloaded = sum(r["image_status"] == "downloaded" and r.get("domain") == "pokemon_physical_tcg" for r in records)
    unavailable = sum(r["image_status"] == "unavailable" and r.get("domain") == "pokemon_physical_tcg" for r in records)
    return {
        "updated_at_utc": now_utc(),
        "provider": "TCGdex",
        "game": "pokemon",
        "language": args.language,
        "catalog_url": catalog_url,
        "catalog_source_sha256": state.get("catalog_source_sha256"),
        "pocket_series_source_sha256": state.get("pocket_series_source_sha256"),
        "assets_documentation": ASSETS_DOC,
        "catalog_complete": True,
        "catalog_count": len(records),
        "physical_catalog_count": sum(r.get("domain") == "pokemon_physical_tcg" for r in records),
        "pocket_digital_catalog_count": sum(r.get("domain") == "pokemon_tcg_pocket_digital" for r in records),
        "domain_unknown_count": sum(r.get("domain") == "unknown" for r in records),
        "gallery_images_complete": complete,
        "downloaded_count": downloaded,
        "downloaded_physical_count": downloaded,
        "excluded_digital_download_count": sum(r.get("image_status") == "excluded_digital" for r in records),
        "image_unavailable_count": unavailable,
        "image_pending_count": sum(r["image_status"] == "pending" and r.get("domain") == "pokemon_physical_tcg" for r in records),
        "excluded_unknown_domain_count": sum(r.get("image_status") == "excluded_unknown_domain" for r in records),
        "downloaded_bytes_total": state.get("downloaded_bytes_total", 0),
        "gallery_bytes_on_disk": sum(r.get("image_bytes", 0) or 0 for r in records if r["image_status"] == "downloaded" and r.get("domain") == "pokemon_physical_tcg"),
        "excluded_digital_bytes_removed": state.get("excluded_digital_bytes_removed", 0),
        "byte_budget_mib": args.max_download_mib,
        "image_limit": args.max_images,
        "workers": args.workers,
        "minimum_free_gib": args.min_free_gib,
        "stopped_reason": stopped_reason,
        "physical_finish_and_edition": "unresolved; catalog images are not claims about a specific physical finish or edition",
        "records_file": "catalog.json",
        "progress_file": "sync_state.json",
    }


def sync(language: str = "en", max_images: int = 256, max_download_mib: int = 256,
         min_free_gib: float = 3.0, output: Path = Path("datasets/sources/catalog/pokemon-en-tcgdex"),
         session: Any | None = None, workers: int = 3) -> dict[str, Any]:
    if language != "en":
        # This first gallery sync intentionally scopes full-catalog identity to English.
        raise ValueError("This full-catalog sync currently supports only --language en")
    if max_images < 0 or max_download_mib < 0 or min_free_gib < 0 or not 1 <= workers <= 4:
        raise ValueError("image and disk budgets must be non-negative")
    output = Path(output)
    output_layout(output)
    if session is None:
        try:
            import requests
        except ModuleNotFoundError as exc:
            raise RuntimeError("This command requires the project's requests dependency") from exc
        session = requests
    catalog_path = output / "catalog.json"
    catalog_source_path = output / "catalog_source.json"
    pocket_source_path = output / "pocket_series_source.json"
    state_path = output / "sync_state.json"
    progress_path = output / "progress.jsonl"
    manifest_path = output / "manifest.json"
    catalog_url = f"{API_BASE}/{language}/cards"
    state = read_json(state_path, {"language": language, "downloaded_bytes_total": 0, "cards": {}})
    if state.get("language") != language or not isinstance(state.get("cards"), dict):
        raise ValueError("Existing sync_state.json has an incompatible schema/language")
    cards_state: dict[str, Any] = state["cards"]
    last_journal_total = 0
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Ignore only a torn final journal record.
                card_id = item.get("provider_id")
                if card_id:
                    cards_state[card_id] = item
                last_journal_total = max(last_journal_total, int(item.get("cumulative_bytes_total", 0)))

    old_catalog = read_json(catalog_path, None)
    if catalog_source_path.exists():
        source_bytes = catalog_source_path.read_bytes()
        source_payload = json.loads(source_bytes)
    else:
        source_payload, source_bytes = request_json_source(catalog_url, session)
        atomic_bytes(catalog_source_path, source_bytes)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if pocket_source_path.exists():
        pocket_bytes = pocket_source_path.read_bytes()
        pocket_payload = json.loads(pocket_bytes)
    else:
        pocket_url = f"{API_BASE}/{language}/series/tcgp"
        pocket_payload, pocket_bytes = request_json_source(pocket_url, session)
        atomic_bytes(pocket_source_path, pocket_bytes)
    pocket_hash = hashlib.sha256(pocket_bytes).hexdigest()
    sets = pocket_payload.get("sets") if isinstance(pocket_payload, dict) else None
    if not isinstance(sets, list):
        raise ValueError("TCGdex tcgp series response is missing its sets list")
    pocket_sets = {safe_segment(item.get("id"), "Pocket set id") for item in sets if isinstance(item, dict) and item.get("id")}
    records = normalize_catalog(source_payload, language)
    old_by_id = {record.get("provider_id"): record for record in old_catalog or [] if isinstance(record, dict)}
    carry_fields = ("image_status", "image_sha256", "image_width", "image_height", "image_bytes", "image_path", "last_error")
    for record in records:
        previous = old_by_id.get(record["provider_id"], {})
        for field in carry_fields:
            if field in previous:
                record[field] = previous[field]
    classify_domains(records, pocket_sets)
    state["catalog_source_sha256"] = source_hash
    state["pocket_series_source_sha256"] = pocket_hash
    state["catalog_count"] = len(records)

    # Reconcile the authoritative image files and saved hashes before resuming.
    for record in records:
        card_id = record["provider_id"]
        saved = cards_state.get(card_id, {})
        destination = card_destination(output, record)
        if record["domain"] == "pokemon_tcg_pocket_digital":
            if destination.exists():
                record["image_status"] = "excluded_digital"
                state["excluded_digital_bytes_removed"] = int(state.get("excluded_digital_bytes_removed", 0)) + destination.stat().st_size
                destination.unlink()
                destination.with_suffix(".json").unlink(missing_ok=True)
            else:
                record["image_status"] = "excluded_digital"
            cards_state[card_id] = {"status": "excluded_digital"}
            continue
        if record["domain"] == "unknown":
            record["image_status"] = "excluded_unknown_domain"
            cards_state[card_id] = {"status": "excluded_unknown_domain"}
            continue
        if record.get("image_status") not in {"pending", "downloaded"}:
            record["image_status"] = "pending" if record.get("image_url") else "unavailable"
        if destination.exists():
            actual_hash = sha256_file(destination)
            sidecar_path = destination.with_suffix(".json")
            expected = saved.get("image_sha256") or record.get("image_sha256")
            if not sidecar_path.is_file():
                # A process may stop after atomically installing an image but before its
                # next catalog checkpoint. These are our own output paths and the
                # catalog's official asset URL supplies the source provenance.
                if expected and expected != actual_hash:
                    raise ValueError(f"Existing image hash mismatch for {card_id}; refusing resume")
                width, height = validate_webp(destination)
                sidecar = {"game": "pokemon", "catalog_id": record["catalog_id"],
                           "provider_id": card_id, "language": language,
                           "set_id": record.get("set_id"), "local_id": record.get("local_id"),
                           "domain": record.get("domain"), "source_url": record.get("image_url"),
                           "image_sha256": actual_hash, "image_bytes": destination.stat().st_size,
                           "image_width": width, "image_height": height,
                           "physical_finish": "unresolved", "physical_edition": "unresolved",
                           "recovered_after_checkpoint": True}
                atomic_json(sidecar_path, sidecar)
            else:
                sidecar = read_json(sidecar_path, {})
            if not expected:
                expected = sidecar.get("image_sha256") if sidecar.get("recovered_after_checkpoint") else None
            if not expected:
                raise ValueError(f"Existing image has no trusted resume hash for {card_id}; refusing to adopt")
            if actual_hash != expected or sidecar.get("image_sha256") != actual_hash or sidecar.get("provider_id") != card_id:
                raise ValueError(f"Existing image sidecar mismatch for {card_id}; refusing resume")
            width, height = validate_webp(destination)
            record.update({"image_status": "downloaded", "image_sha256": actual_hash,
                           "image_bytes": destination.stat().st_size, "image_width": width,
                           "image_height": height, "image_path": destination.relative_to(output).as_posix()})
            cards_state[card_id] = {"status": "downloaded", "image_sha256": actual_hash,
                                    "image_bytes": destination.stat().st_size,
                                    "image_path": record["image_path"]}
        elif saved.get("status") == "downloaded":
            # A missing file invalidates saved completion; allow it to be retried.
            cards_state[card_id] = {"status": "pending"}
            record["image_status"] = "pending"

    bytes_total = max(int(state.get("downloaded_bytes_total", 0)), last_journal_total)
    byte_cap = max_download_mib * 1024 * 1024
    stopped_reason: str | None = None
    downloaded_this_run = 0
    candidates = fair_round_robin([r for r in records if r.get("domain") == "pokemon_physical_tcg"])
    state.update({"language": language, "downloaded_bytes_total": bytes_total,
                  "catalog_source_sha256": source_hash, "pocket_series_source_sha256": pocket_hash,
                  "catalog_count": len(records), "cards": {}})
    def checkpoint(stopped: str | None = None) -> None:
        state.update({"language": language, "downloaded_bytes_total": bytes_total,
                      "catalog_source_sha256": source_hash, "pocket_series_source_sha256": pocket_hash,
                      "catalog_count": len(records), "cards": {},
                      "excluded_digital_bytes_removed": int(state.get("excluded_digital_bytes_removed", 0))})
        atomic_json(catalog_path, records)
        atomic_json(state_path, state)
        physical_records = [r for r in records if r.get("domain") == "pokemon_physical_tcg"]
        gallery_complete = bool(physical_records) and all(r["image_status"] == "downloaded" for r in physical_records)
        manifest_value = build_manifest(records, state, complete=gallery_complete, stopped_reason=stopped,
                                        args=argparse.Namespace(language=language, max_download_mib=max_download_mib,
                                                                max_images=max_images, min_free_gib=min_free_gib,
                                                                workers=workers),
                                        catalog_url=catalog_url)
        manifest_value["downloaded_this_run"] = downloaded_this_run
        atomic_json(manifest_path, manifest_value)

    checkpoint("resumable sync in progress")
    floor_bytes = int(min_free_gib * 1024**3)
    budget = DownloadBudget(byte_cap, output, floor_bytes, bytes_total)
    candidate_iter = iter(candidates)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        stop_scheduling = False
        while not stop_scheduling:
            batch: dict[Any, dict[str, Any]] = {}
            downloaded_count = sum(r["image_status"] == "downloaded" and r.get("domain") == "pokemon_physical_tcg" for r in records)
            while len(batch) < workers:
                if downloaded_count + len(batch) >= max_images:
                    stopped_reason = f"image limit reached ({max_images})"
                    stop_scheduling = True
                    break
                try:
                    record = next(candidate_iter)
                except StopIteration:
                    stop_scheduling = True
                    break
                if record["image_status"] != "pending":
                    continue
                if budget.used + budget.reserved_bytes >= byte_cap:
                    stopped_reason = f"cumulative byte budget reached ({max_download_mib} MiB)"
                    stop_scheduling = True
                    break
                destination = card_destination(output, record)
                if destination.exists():
                    raise ValueError(f"Unexpected existing image without a valid resume record: {destination}")
                record["image_status"] = "downloading"
                future = executor.submit(download_webp, record["image_url"], destination, session, budget)
                batch[future] = record
            if not batch:
                break
            for future in as_completed(batch):
                record = batch[future]
                destination = card_destination(output, record)
                try:
                    size, digest, width, height = future.result()
                except (BudgetExceeded, DiskFloorExceeded) as exc:
                    record["image_status"] = "pending"
                    stopped_reason = str(exc)
                    stop_scheduling = True
                    continue
                except Exception as exc:
                    record["image_status"] = "pending"
                    record["last_error"] = str(exc)
                    cards_state[record["provider_id"]] = {"status": "pending", "last_error": str(exc)}
                    append_jsonl(progress_path, {"provider_id": record["provider_id"], "status": "pending",
                                                 "last_error": str(exc), "cumulative_bytes_total": budget.used})
                    continue
                bytes_total = budget.used
                downloaded_this_run += 1
                record.update({"image_status": "downloaded", "image_sha256": digest, "image_bytes": size,
                               "image_width": width, "image_height": height,
                               "image_path": destination.relative_to(output).as_posix(), "last_error": None})
                atomic_json(destination.with_suffix(".json"), record)
                journal_record = {"provider_id": record["provider_id"], "status": "downloaded",
                                  "image_sha256": digest, "image_bytes": size,
                                  "image_path": record["image_path"], "cumulative_bytes_total": bytes_total}
                append_jsonl(progress_path, journal_record)
                cards_state[record["provider_id"]] = journal_record
                if downloaded_this_run % 100 == 0:
                    checkpoint("resumable sync in progress")
            bytes_total = budget.used
            if stopped_reason and stopped_reason.startswith(("image Content-Length", "image reservation")):
                stop_scheduling = True

    state.update({"language": language, "downloaded_bytes_total": bytes_total,
                  "catalog_source_sha256": source_hash, "pocket_series_source_sha256": pocket_hash,
                  "catalog_count": len(records), "cards": {}})
    physical = [r for r in records if r.get("domain") == "pokemon_physical_tcg"]
    complete = bool(physical) and all(r["image_status"] == "downloaded" for r in physical)
    if complete:
        stopped_reason = None
    elif stopped_reason is None:
        if any(r["image_status"] == "pending" for r in records):
            stopped_reason = f"some images failed after {ATTEMPTS} bounded attempts; rerun to resume"
        else:
            stopped_reason = "one or more catalog entries have no image asset; missing images are recorded as unavailable"
    checkpoint(stopped_reason)
    return read_json(manifest_path, {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-images", type=int, default=30000)
    parser.add_argument("--max-download-mib", type=int, default=1536)
    parser.add_argument("--min-free-gib", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--output", type=Path, default=Path("datasets/sources/catalog/pokemon-en-tcgdex"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = sync(args.language, args.max_images, args.max_download_mib, args.min_free_gib,
                        args.output, workers=args.workers)
    except Exception as exc:
        raise SystemExit(f"Pokémon catalog sync failed: {exc}") from exc
    print(f"Catalog: {manifest['catalog_count']} cards; downloaded images: {manifest['downloaded_count']}")
    print(f"Bytes: {manifest['downloaded_bytes_total']}; complete gallery: {manifest['gallery_images_complete']}")
    if manifest["stopped_reason"]:
        print(f"Stopped: {manifest['stopped_reason']}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
