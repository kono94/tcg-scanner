#!/usr/bin/env python3
"""Zero-shot retrieval check for a game the encoder was not trained on.

`embed` encodes a synced catalog's references with the deployed fp32 ONNX encoder through
the known-full-card-bounds adapter (no detector), as the bundle builder does for card backs.
`query` ranks detector crops against that gallery and writes top-k proposals per crop.
Top-k hits are proposals for visual review, not labels; cosine scores are not probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.recognition.backbone import sha256_file  # noqa: E402
from training.recognition.canonical import PREPROCESSING_VERSION, canonicalize_card, prepare_encoder_input  # noqa: E402

DEFAULT_ENCODER = Path("models/browser-export/gpu-20261017/encoder.onnx")


def session(encoder: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(encoder), providers=["CPUExecutionProvider"])


def embed_batch(sess: ort.InferenceSession, tensors: list[np.ndarray]) -> np.ndarray:
    name = sess.get_inputs()[0].name
    out = np.concatenate([sess.run(None, {name: t[None]})[0] for t in tensors]).astype(np.float64)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    if not np.isfinite(out).all() or (norms < 1e-6).any():
        raise ValueError("invalid embedding")
    return (out / norms).astype(np.float32)


def cmd_embed(args) -> int:
    manifest = json.loads(args.catalog.read_text())
    root = args.catalog.parent
    sess = session(args.encoder)
    refs = [r for r in manifest["cards"] if r.get("status") == "downloaded"]
    if args.limit:
        refs = refs[:args.limit]
    rows, index = [], []
    for i, r in enumerate(refs, 1):
        path = root / r["image_path"]
        if sha256_file(path) != r["image_sha256"]:
            raise ValueError(f"{path} differs from its manifest hash")
        rgb = np.asarray(Image.open(path).convert("RGB"))
        rows.append(embed_batch(sess, [prepare_encoder_input(canonicalize_card(rgb))])[0])
        index.append({k: r[k] for k in ("catalog_identity", "card_id", "artwork_id", "name", "image_path")})
        if i % 1000 == 0:
            print(f"{i}/{len(refs)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "gallery.npy", np.stack(rows))
    (args.output / "gallery_index.json").write_text(json.dumps({
        "encoder": str(args.encoder), "encoder_sha256": sha256_file(args.encoder),
        "preprocessing": PREPROCESSING_VERSION, "adapter": "known full-card bounds (no detector)",
        "catalog_manifest": str(args.catalog), "catalog_manifest_sha256": sha256_file(args.catalog),
        "dimension": int(rows[0].shape[0]), "normalization": "L2", "rows": index,
    }, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(rows)} rows -> {args.output}")
    return 0


def cmd_query(args) -> int:
    gallery = np.load(args.gallery / "gallery.npy")
    index = json.loads((args.gallery / "gallery_index.json").read_text())
    if index["encoder_sha256"] != sha256_file(args.encoder):
        raise ValueError("gallery was embedded with a different encoder")
    sess = session(args.encoder)
    crops = sorted(p for p in args.crops.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    results = []
    for path in crops:
        # Detector crops are already rectified to the canonical size; canonicalize again with full bounds.
        rgb = np.asarray(Image.open(path).convert("RGB"))
        q = embed_batch(sess, [prepare_encoder_input(canonicalize_card(rgb))])[0]
        scores = gallery @ q
        order = np.argsort(-scores)
        hits, cards = [], set()
        for j in order:
            row = index["rows"][j]
            if row["card_id"] in cards and len(hits) >= args.top_k:
                continue
            hits.append({**row, "score": round(float(scores[j]), 4)})
            cards.add(row["card_id"])
            if len(cards) >= args.top_k:
                break
        results.append({"crop": str(path), "hits": hits})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gallery": str(args.gallery), "encoder": str(args.encoder),
                                       "note": "proposals for visual review, not labels", "results": results},
                                      ensure_ascii=False, indent=1) + "\n")
    print(f"{len(results)} crops -> {args.output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--catalog", type=Path, required=True, help="catalog manifest.json from a sync script")
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    e.add_argument("--limit", type=int, default=0)
    q = sub.add_parser("query")
    q.add_argument("--gallery", type=Path, required=True)
    q.add_argument("--crops", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    q.add_argument("--top-k", type=int, default=5, help="distinct cards per crop")
    args = ap.parse_args()
    return cmd_embed(args) if args.cmd == "embed" else cmd_query(args)


if __name__ == "__main__":
    raise SystemExit(main())
