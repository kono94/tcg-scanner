#!/usr/bin/env python3
"""Create a read-only quality and repeat-view inventory for detector crops.

The inventory is advisory. It does not assign card identities, edit manifests,
or alter/delete any source crop or source frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [
    "runs/video_mining/binder-campaign/batch01/crops/queries.json",
    "runs/video_mining/binder-campaign/batch02/crops/queries.json",
    "runs/video_mining/binder-campaign/batch03/crops/queries.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def crop_path(manifest_path: Path, image_ref: str) -> Path:
    # Current crop manifests store e.g. "crops/example.png" relative to the
    # proposal output directory, whose parent is the manifest directory.
    p = (manifest_path.parent / image_ref).resolve()
    if ROOT.resolve() not in p.parents:
        raise ValueError(f"Crop path escapes repository: {image_ref}")
    return p


def box_for(q: dict) -> list[float]:
    return q.get("bbox_aabb_source_pixels_clipped") or q.get("bbox_aabb_source_pixels") or []


def box_metrics(box: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return w, h, (x1 + x2) / 2, (y1 + y2) / 2


def sharpness(image_path: Path) -> float:
    """Approximate Laplacian variance after bounded grayscale downsampling."""
    with Image.open(image_path) as im:
        im = im.convert("L")
        im.thumbnail((384, 384), Image.Resampling.LANCZOS)
        a = np.asarray(im, dtype=np.float32)
    if min(a.shape) < 3:
        return 0.0
    lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1]
           + a[1:-1, :-2] + a[1:-1, 2:])
    return float(np.var(lap))


def signature(image_path: Path) -> tuple[int, np.ndarray, float]:
    """Return horizontal dHash, 64x64 luminance and crop aspect ratio."""
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        aspect = rgb.width / max(1, rgb.height)
        gray = rgb.convert("L")
        dh = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        bits = dh[:, 1:] > dh[:, :-1]
        h = 0
        for b in bits.flat:
            h = (h << 1) | int(b)
        thumb = np.asarray(gray.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32)
    return h, thumb, aspect


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def normalized_ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Global SSIM on identical-size thumbnails. This deliberately conservative
    # check prevents a near hash alone from joining similar card designs.
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var()), float(b.var())
    cov = float(((a - ma) * (b - mb)).mean())
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


def looks_like_parent(q: dict, others: list[dict]) -> list[str]:
    box = q.get("bbox", box_for(q))
    if len(box) != 4:
        return []
    pw, ph, *_ = box_metrics(box)
    pa = pw * ph
    nested = []
    for o in others:
        if o is q or o.get("source_frame_image") != q.get("source_frame_image"):
            continue
        b = o.get("bbox", box_for(o))
        if len(b) != 4:
            continue
        w, h, *_ = box_metrics(b)
        if w <= 0 or h <= 0 or w * h >= pa * 0.4:
            continue
        if b[0] >= box[0] and b[1] >= box[1] and b[2] <= box[2] and b[3] <= box[3]:
            nested.append(o.get("crop_id", o.get("image", "unknown")))
    return nested


def near_repeat(a: dict, b: dict) -> tuple[bool, dict]:
    if a["source_group"] != b["source_group"]:
        return False, {}
    if a["source_frame_image"] == b["source_frame_image"]:
        # Different detections in one scene may be different physical copies.
        return False, {}
    ta, tb = a.get("time_seconds"), b.get("time_seconds")
    if ta is not None and tb is not None and abs(float(ta) - float(tb)) > 4.0:
        return False, {}
    aw, ah, ax, ay = box_metrics(a["bbox"])
    bw, bh, bx, by = box_metrics(b["bbox"])
    sw, sh = max(a["source_width"], b["source_width"]), max(a["source_height"], b["source_height"])
    center_delta = math.hypot((ax - bx) / max(sw, 1), (ay - by) / max(sh, 1))
    size_ratio = min(aw * ah, bw * bh) / max(aw * ah, bw * bh, 1.0)
    if center_delta > 0.055 or size_ratio < 0.78:
        return False, {}
    if hamming(a["dhash"], b["dhash"]) > 1 or abs(a["aspect"] - b["aspect"]) > 0.025:
        return False, {}
    ssim = normalized_ssim(a["thumb"], b["thumb"])
    if ssim < 0.99:
        return False, {"ssim": round(ssim, 5)}
    return True, {"ssim": round(ssim, 5), "dhash_distance": hamming(a["dhash"], b["dhash"]),
                  "normalized_center_delta": round(center_delta, 5), "area_ratio": round(size_ratio, 5)}


def components(rows: list[dict]) -> list[list[dict]]:
    # Connected components over exceptionally strict pairwise near-repeat edges.
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            if near_repeat(a, rows[j])[0]:
                union(i, j)
    groups: dict[int, list[dict]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def contact_sheets(rows: list[dict], output: Path, per_page: int = 24) -> list[str]:
    """Create contact sheets of sharpest representatives, split by batch/source."""
    result = []
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["batch"], row["source_group"])].append(row)
    for (batch, source), bucket in sorted(buckets.items()):
        bucket.sort(key=lambda x: x["cluster_id"])
        for page_start in range(0, len(bucket), per_page):
            page = bucket[page_start:page_start + per_page]
            cols, cell_w, image_h, label_h = 4, 360, 280, 72
            rows_n = math.ceil(len(page) / cols)
            sheet = Image.new("RGB", (cols * cell_w, rows_n * (image_h + label_h)), "white")
            draw = ImageDraw.Draw(sheet)
            for k, row in enumerate(page):
                x, y = (k % cols) * cell_w, (k // cols) * (image_h + label_h)
                with Image.open(row["crop_path"]) as crop:
                    crop = crop.convert("RGB")
                    crop.thumbnail((cell_w - 12, image_h - 12), Image.Resampling.LANCZOS)
                    sheet.paste(crop, (x + (cell_w - crop.width) // 2, y + (image_h - crop.height) // 2))
                flags = ", ".join(row["review_flags"]) or "no size/page alert"
                lines = [f"{row['cluster_id']} · n={row['cluster_member_count']} · sharp#{row['sharpness_rank_global']}",
                         row["crop_id"][:48], f"frame {row['source_frame']} · {flags}"[:54]]
                draw.multiline_text((x + 6, y + image_h + 3), "\n".join(lines), fill="black", spacing=2)
            name = f"{batch}-{source}-p{page_start // per_page + 1:02d}.jpg".replace("/", "_")
            dest = output / "contact_sheets" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(dest, quality=86, optimize=True)
            result.append(str(dest.relative_to(ROOT)))
    return result


def run(manifests: list[Path], output: Path) -> dict:
    inventory: list[dict] = []
    input_summaries = []
    for manifest_path in manifests:
        if not manifest_path.exists():
            continue
        data = json.loads(manifest_path.read_text())
        batch = manifest_path.parents[1].name if manifest_path.parent.name == "crops" else manifest_path.parent.name
        queries = data.get("queries", [])
        input_summaries.append({"manifest": str(manifest_path.relative_to(ROOT)),
                                "batch": batch, "query_count": len(queries),
                                "all_detection_count": data.get("all_detection_count"),
                                "geometry_eligible_count": data.get("recognition_geometry_eligible_count")})
        for index, q in enumerate(queries):
            image = crop_path(manifest_path, q["image"])
            if not image.is_file():
                raise FileNotFoundError(f"Missing crop file: {image}")
            digest = sha256(image)
            if q.get("sha256") and digest != q["sha256"]:
                raise ValueError(f"Crop checksum mismatch: {image}")
            with Image.open(image) as im:
                width, height = im.size
            if q.get("crop_width") is not None and width != int(q["crop_width"]):
                raise ValueError(f"Crop width changed since manifest: {image}")
            if q.get("crop_height") is not None and height != int(q["crop_height"]):
                raise ValueError(f"Crop height changed since manifest: {image}")
            score = sharpness(image)
            dh, thumb, aspect = signature(image)
            bbox = box_for(q)
            short_edge = min(width, height)
            flags = []
            if short_edge < 300:
                flags.append("short_edge_under_300px_provisional")
            if short_edge < 180:
                flags.append("very_small_subregion_or_noncard_candidate")
            if not 0.42 <= aspect <= 0.95:
                flags.append("unusual_portrait_aspect_or_rotated_card_review")
            if q.get("aabb_clipped_to_source"):
                flags.append("crop_clipped_at_source_edge_possible_partial_card")
            area_fraction = q.get("source_area_fraction")
            if area_fraction is not None and float(area_fraction) >= 0.18:
                flags.append("large_source_region_review_possible_page_or_multiple_cards")
            inventory.append({
                "batch": batch, "manifest": str(manifest_path.relative_to(ROOT)),
                "query_index": index, "crop_id": Path(q["image"]).stem,
                "crop_path": str(image.relative_to(ROOT)), "crop_sha256": digest,
                "width": width, "height": height, "short_edge_pixels": short_edge,
                "sharpness_laplacian_variance": score, "sharpness_rank_global": None,
                "sharpness_rank_sourcegroup": None, "sharpness_rank_within_repeat_cluster": None,
                "source_group": q.get("source_group", "unknown"),
                "source_frame": q.get("source_frame"), "time_seconds": q.get("time_seconds"),
                "source_frame_image": q.get("source_frame_image"),
                "source_frame_sha256": q.get("source_frame_sha256"),
                "bbox": bbox, "detection_score": q.get("detection_score"),
                "source_area_fraction": q.get("source_area_fraction"),
                "aabb_clipped_to_source": bool(q.get("aabb_clipped_to_source", False)),
                "geometry_eligible": bool(q.get("recognition_geometry_eligible", False)),
                "geometry_screen_reasons": q.get("geometry_screen_reasons", []),
                "aspect_width_over_height": aspect,
                "aspect": aspect,
                "source_width": q.get("source_width", 0), "source_height": q.get("source_height", 0),
                "review_flags": flags,
                "complete_physical_card_verified": False,
                "identity_status": "unresolved",
                "dhash": dh, "thumb": thumb, "_query": q,
            })

    for rank, row in enumerate(sorted(inventory, key=lambda x: x["sharpness_laplacian_variance"], reverse=True), 1):
        row["sharpness_rank_global"] = rank
    low_sharpness_floor_rank = math.floor(len(inventory) * 0.90)
    for row in inventory:
        if row["sharpness_rank_global"] > low_sharpness_floor_rank:
            row["review_flags"].append("low_sharpness_bottom_decile_provisional")
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in inventory:
        by_group[row["source_group"]].append(row)
    for group in by_group.values():
        for rank, row in enumerate(sorted(group, key=lambda x: x["sharpness_laplacian_variance"], reverse=True), 1):
            row["sharpness_rank_sourcegroup"] = rank

    clusters = []
    rep_rows = []
    edge_rows = []
    for row in inventory:
        row["nested_smaller_crop_ids"] = []
    per_frame: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in inventory:
        per_frame[(row["batch"], row.get("source_frame_image") or "")].append(row)
    for frame_rows in per_frame.values():
        for row in frame_rows:
            nested = looks_like_parent(row, frame_rows)
            row["nested_smaller_crop_ids"] = nested
            if len(nested) >= 3:
                row["review_flags"].append("possible_parent_page_box_contains_3plus_detections")

    for batch in sorted({r["batch"] for r in inventory}):
        batch_rows = [r for r in inventory if r["batch"] == batch]
        for group in components(batch_rows):
            group.sort(key=lambda x: x["sharpness_laplacian_variance"], reverse=True)
            cluster_id = f"{batch}-c{len(clusters)+1:04d}"
            for local_rank, row in enumerate(group, 1):
                row["cluster_id"] = cluster_id
                row["sharpness_rank_within_repeat_cluster"] = local_rank
                row["cluster_member_count"] = len(group)
            representative = group[0]
            representative["is_contact_sheet_representative"] = True
            rep_rows.append(representative)
            clusters.append({"cluster_id": cluster_id, "batch": batch,
                             "source_group": representative["source_group"],
                             "candidate_repeat_view": len(group) > 1,
                             "cluster_rule": "same source_group; distinct frames; <=4s; normalized centers <=0.055; area ratio >=0.78; dHash distance <=1; aspect delta <=0.025; 64x64 luminance SSIM >=0.99",
                             "representative_crop_id": representative["crop_id"],
                             "representative_crop_path": representative["crop_path"],
                             "member_crop_ids": [r["crop_id"] for r in group],
                             "member_crop_paths": [r["crop_path"] for r in group],
                             "member_frames": [r["source_frame_image"] for r in group],
                             "member_count": len(group),
                             "near_repeat_evidence": [
                                 {"left_crop_id": group[i]["crop_id"], "right_crop_id": group[j]["crop_id"],
                                  **near_repeat(group[i], group[j])[1]}
                                 for i in range(len(group)) for j in range(i + 1, len(group))
                                 if near_repeat(group[i], group[j])[0]
                             ]})
            for row in group:
                row.pop("dhash", None); row.pop("thumb", None); row.pop("_query", None)

    # Report every crop pair that meets the strict repeat rule inside clusters;
    # all cluster members remain separate inventory rows for human review.
    sheet_paths = contact_sheets(rep_rows, output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "inventory.json").write_text(json.dumps({
        "schema_version": 1, "artifact_type": "read_only_recognition_crop_inventory",
        "generated_from": input_summaries,
        "identity_assignment": "none; all labels remain unresolved",
        "mutation_policy": "source crop and frame files were not modified or deleted",
        "thresholds_calibrated": False,
        "near_repeat_policy": "conservative candidate grouping only; all member crops retained and listed; matching crops may still be separate physical copies",
        "quality_flags_are_advisory": True,
        "excluded_roots": [
            {"path": "runs/card_recognizer/real_video_crops/", "reason": "historical known-bad 116 proposals; kept separate and not included in this campaign inventory"}
        ],
        "review_flag_rules": {
            "possible_parent_page_box_contains_3plus_detections": "same source frame: this box fully contains at least three smaller crop boxes, each under 40% of parent area",
            "short_edge_under_300px_provisional": "crop short edge below 300 source pixels; advisory, uncalibrated",
            "very_small_subregion_or_noncard_candidate": "crop short edge below 180 source pixels; advisory",
            "unusual_portrait_aspect_or_rotated_card_review": "crop width/height outside 0.42–0.95; may be landscape/rotated card or a partial/non-card region",
            "low_sharpness_bottom_decile_provisional": "among the lowest 10% of Laplacian-variance sharpness scores across inventoried crops; uncalibrated and size-sensitive",
            "crop_clipped_at_source_edge_possible_partial_card": "detector box clipping flag indicates it reached a source-frame boundary",
            "large_source_region_review_possible_page_or_multiple_cards": "detected crop covers at least 18% of source frame area; may be a binder page or scene region"
        },
        "counts": {"crop_rows": len(inventory), "repeat_view_clusters": len(clusters),
                   "candidate_repeat_clusters": sum(c["member_count"] > 1 for c in clusters),
                   "candidate_repeat_rows": sum(c["member_count"] for c in clusters if c["member_count"] > 1),
                   "possible_parent_page_boxes": sum("possible_parent_page_box_contains_3plus_detections" in r["review_flags"] for r in inventory),
                   "short_edge_under_300px": sum("short_edge_under_300px_provisional" in r["review_flags"] for r in inventory),
                   "very_small_subregion_or_noncard_candidates": sum("very_small_subregion_or_noncard_candidate" in r["review_flags"] for r in inventory),
                   "unusual_aspect_or_rotated_review": sum("unusual_portrait_aspect_or_rotated_card_review" in r["review_flags"] for r in inventory),
                   "low_sharpness_bottom_decile_provisional": sum("low_sharpness_bottom_decile_provisional" in r["review_flags"] for r in inventory),
                   "crop_clipped_at_source_edge": sum("crop_clipped_at_source_edge_possible_partial_card" in r["review_flags"] for r in inventory),
                   "large_source_region_review": sum("large_source_region_review_possible_page_or_multiple_cards" in r["review_flags"] for r in inventory),
                   "geometry_eligible": sum(r["geometry_eligible"] for r in inventory)},
        "contact_sheets": sheet_paths,
        "clusters": clusters,
        "crops": inventory,
    }, indent=2) + "\n")
    with (output / "inventory.csv").open("w") as f:
        import csv
        fields = ["batch", "crop_id", "crop_path", "crop_sha256", "width", "height", "short_edge_pixels",
                  "sharpness_laplacian_variance", "sharpness_rank_global", "sharpness_rank_sourcegroup",
                  "sharpness_rank_within_repeat_cluster", "cluster_id", "cluster_member_count",
                  "is_contact_sheet_representative", "source_group", "source_frame", "time_seconds",
                  "source_frame_image", "source_frame_sha256", "bbox", "detection_score", "geometry_eligible",
                  "geometry_screen_reasons", "nested_smaller_crop_ids", "source_area_fraction",
                  "aabb_clipped_to_source", "review_flags", "identity_status"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in inventory:
            writer.writerow({**row, "bbox": json.dumps(row["bbox"]),
                             "geometry_screen_reasons": json.dumps(row["geometry_screen_reasons"]),
                             "nested_smaller_crop_ids": json.dumps(row["nested_smaller_crop_ids"]),
                             "review_flags": json.dumps(row["review_flags"])})
    return {"inventory": str((output / "inventory.json").relative_to(ROOT)),
            "csv": str((output / "inventory.csv").relative_to(ROOT)),
            "crops": len(inventory), "clusters": len(clusters), "contact_sheets": len(sheet_paths),
            "inputs": input_summaries}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", action="append", type=Path,
                   help="Crop query manifest; repeat to inventory multiple batches")
    p.add_argument("--output", type=Path, default=ROOT / "runs/dataset_cleanup/inventory")
    args = p.parse_args()
    manifests = args.manifest or [ROOT / x for x in DEFAULT_MANIFESTS]
    manifests = [x if x.is_absolute() else ROOT / x for x in manifests]
    out = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(run(manifests, out), indent=2))


if __name__ == "__main__":
    main()
