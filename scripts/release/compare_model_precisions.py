#!/usr/bin/env python3
"""fp16 copies of the browser encoder/detector and their agreement with fp32 on real data.

Writes `encoder.fp16.onnx`, `detector.fp16.onnx` and `precision_report.json` (with the fp32
source hashes the bundle builder checks) next to the fp32 export. Converts weights to float16 with float32 inputs/outputs (the worker's tensors stay
unchanged), then compares fp32, fp16 and the existing int8 encoder:

- encoder: every labeled real query crop (train/val/test, AI-reviewed labels) against the
  bundle gallery (fp16 rows embedded with the fp32 encoder): exact-ID top-1/top-5, top-1
  agreement with fp32, cosine to the fp32 embedding and the change in top-1 score/margin.
- detector: held-out real-corner frames, matched against reviewed corners, plus agreement
  between fp32 and fp16 detections on any extra frame folders given.

Results are diagnostics on repeated video views with pseudo-labels, not scanner accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release.export_browser_models import letterbox, ort_session  # noqa: E402
from training.recognition.backbone import sha256_file  # noqa: E402
from training.recognition.canonical import canonicalize_card, prepare_encoder_input  # noqa: E402



def to_fp16(src: Path, dst: Path) -> None:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    model = onnx.load(str(src))
    # The encoder's final L2 normalization (ReduceL2 -> Clip(eps) -> Div) stays float32: the
    # 1e-12 clip floor underflows in fp16 and the whole embedding turns NaN.
    tail = [n.name for n in model.graph.node if n.op_type in ("ReduceL2", "Clip", "Expand", "Div")
            and any(n2.op_type == "ReduceL2" for n2 in model.graph.node)]
    last_matmul = max(i for i, n in enumerate(model.graph.node) if n.op_type in ("MatMul", "Conv"))
    tail = [n.name for n in model.graph.node[last_matmul + 1:] if n.name in tail]
    model = convert_float_to_float16(model, keep_io_types=True, node_block_list=tail)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))


def embed(sess, rgb: np.ndarray) -> np.ndarray:
    v = sess.run(None, {sess.get_inputs()[0].name: prepare_encoder_input(canonicalize_card(rgb))[None]})[0][0].astype(np.float64)
    return v / np.linalg.norm(v)


def compare_encoders(variants: dict[str, Path], dataset: Path, export: Path) -> dict:
    meta = json.loads((export / "gallery.json").read_text())
    gallery = np.frombuffer((export / meta["vectors"]).read_bytes(), "<f2").astype(np.float32).reshape(-1, meta["dimension"])
    ids = meta["identities"]
    manifest = json.loads((dataset / "recognition" / "manifest.json").read_text())
    queries = [(split, q) for split, s in manifest["splits"].items() for q in s["queries"]]
    sessions = {k: ort_session(p) for k, p in variants.items()}
    rows = []
    for split, q in queries:
        truth = "{game}:{language}:{card_id}".format(**q["identity"])
        rgb = np.asarray(Image.open(dataset / q["image"]).convert("RGB"))
        row = {"split": split, "truth": truth}
        for name, sess in sessions.items():
            v = embed(sess, rgb)
            scores = gallery @ v.astype(np.float32)
            order = np.argsort(-scores)[:5]
            top = [ids[i] for i in order]
            row[name] = {"v": v, "top": top, "s1": float(scores[order[0]]), "margin": float(scores[order[0]] - scores[order[1]])}
        rows.append(row)
    out = {"queries": len(rows), "by_split": {s: sum(r["split"] == s for r in rows) for s in manifest["splits"]}}
    for name in variants:
        res = [r[name] for r in rows]
        out[name] = {
            "top1": sum(r["truth"] == x["top"][0] for r, x in zip(rows, res)),
            "top5": sum(r["truth"] in x["top"] for r, x in zip(rows, res)),
        }
        if name != "fp32":
            ref = [r["fp32"] for r in rows]
            cos = [float(a["v"] @ b["v"]) for a, b in zip(res, ref)]
            out[name].update({
                "top1_same_as_fp32": sum(a["top"][0] == b["top"][0] for a, b in zip(res, ref)),
                "top5_same_order_as_fp32": sum(a["top"] == b["top"] for a, b in zip(res, ref)),
                "cosine_to_fp32": {"min": round(min(cos), 5), "mean": round(float(np.mean(cos)), 5)},
                "top1_score_abs_change": {"max": round(max(abs(a["s1"] - b["s1"]) for a, b in zip(res, ref)), 5),
                                          "mean": round(float(np.mean([abs(a["s1"] - b["s1"]) for a, b in zip(res, ref)])), 5)},
                "margin_abs_change": {"max": round(max(abs(a["margin"] - b["margin"]) for a, b in zip(res, ref)), 5),
                                      "mean": round(float(np.mean([abs(a["margin"] - b["margin"]) for a, b in zip(res, ref)])), 5)},
            })
    return out


def detect(sess, rgb: np.ndarray, size: int = 640, conf: float = 0.6, iou: float = 0.7) -> list[np.ndarray]:
    """Corner quads in source pixels: confidence filter + greedy NMS on boxes (as the worker does)."""
    tensor, lb = letterbox(rgb, size)
    out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]
    keep = np.where(out[4] >= conf)[0]
    keep = keep[np.argsort(-out[4, keep])]
    chosen = []
    for i in keep:
        cx, cy, w, h = out[:4, i]
        box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        if all(box_iou(box, b) <= iou for b, _ in chosen):
            chosen.append((box, i))
    quads = []
    for _, i in chosen:
        k = out[5:, i].reshape(4, 3)[:, :2]
        quads.append((k - [lb["pad_x"], lb["pad_y"]]) / lb["scale"])
    return quads


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def quad_error(a: np.ndarray, b: np.ndarray) -> float:
    """Mean corner distance as a fraction of b's diagonal."""
    diag = np.linalg.norm(b[0] - b[2])
    return float(np.mean(np.linalg.norm(a - b, axis=1)) / diag)


def match(found: list[np.ndarray], wanted: list[np.ndarray], tol: float = 0.05) -> tuple[int, list[float]]:
    errs, used = [], set()
    for w in wanted:
        best = min(((quad_error(f, w), j) for j, f in enumerate(found) if j not in used), default=None)
        if best and best[0] <= tol:
            used.add(best[1])
            errs.append(best[0])
    return len(errs), errs


def compare_detectors(fp32: Path, fp16: Path, labels: Path, extra: list[Path]) -> dict:
    s32, s16 = ort_session(fp32), ort_session(fp16)
    frames = json.loads(labels.read_text())["frames"]
    stats = {k: {"cards": 0, "found": 0, "errors": []} for k in ("fp32", "fp16")}
    for f in frames:
        rgb = np.asarray(Image.open(f["frame"]).convert("RGB"))
        wanted = [np.array(c["corners"]) for c in f["cards"]]
        for name, sess in (("fp32", s32), ("fp16", s16)):
            n, errs = match(detect(sess, rgb), wanted)
            stats[name]["cards"] += len(wanted)
            stats[name]["found"] += n
            stats[name]["errors"] += errs
    labeled = {k: {"cards": v["cards"], "found_within_5pct": v["found"],
                   "mean_corner_error_pct_of_diagonal": round(100 * float(np.mean(v["errors"])), 3) if v["errors"] else None}
               for k, v in stats.items()}
    agree = {"frames": 0, "fp32_detections": 0, "fp16_detections": 0, "matched_within_1pct": 0, "max_matched_error_pct": 0.0}
    for folder in extra:
        for path in sorted(folder.glob("*.jpg")):
            rgb = np.asarray(Image.open(path).convert("RGB"))
            a, b = detect(s32, rgb), detect(s16, rgb)
            n, errs = match(b, a, tol=0.01)
            agree["frames"] += 1
            agree["fp32_detections"] += len(a)
            agree["fp16_detections"] += len(b)
            agree["matched_within_1pct"] += n
            if errs:
                agree["max_matched_error_pct"] = round(max(agree["max_matched_error_pct"], 100 * max(errs)), 4)
    return {"labeled_heldout": {"frames": len(frames), **labeled}, "fp16_vs_fp32_unlabeled": agree}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, default=Path("models/browser-export/gpu-20261017"))
    ap.add_argument("--dataset", type=Path, default=Path("datasets/curated"))
    ap.add_argument("--labels", type=Path, default=Path("datasets/detection/real-corners/heldout_labels_v1.json"))
    ap.add_argument("--extra-frames", type=Path, nargs="*", default=[], help="frame folders for fp32/fp16 detection agreement")
    args = ap.parse_args()
    ex = args.export
    enc32, det32, int8 = ex / "encoder.onnx", ex / "detector.onnx", ex / "encoder.int8.onnx"
    enc16, det16 = ex / "encoder.fp16.onnx", ex / "detector.fp16.onnx"
    to_fp16(enc32, enc16)
    to_fp16(det32, det16)
    encoders = {"fp32": enc32, "fp16": enc16, **({"int8": int8} if int8.exists() else {})}
    report = {
        "schema_version": 1,
        "conversion": "onnxruntime.transformers.float16, keep_io_types; encoder L2-normalization tail kept float32",
        "models": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in (enc32, enc16, det32, det16, *([int8] if int8.exists() else []))},
        "fp16_sources": {enc16.name: sha256_file(enc32), det16.name: sha256_file(det32)},
        "encoder": compare_encoders(encoders, args.dataset, ex),
        "detector": compare_detectors(det32, det16, args.labels, args.extra_frames),
        "note": "CPU onnxruntime in Python; in-browser WebGPU parity of the fp16 models is unverified.",
    }
    (ex / "precision_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
