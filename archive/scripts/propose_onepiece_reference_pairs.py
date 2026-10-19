#!/usr/bin/env python3
"""Propose English One Piece crop/reference pairs for human verification.

All matches remain unresolved proposals. This tool does not change or delete
crop/reference files and does not promote any result into training/evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from scripts.verify_card_geometry import verify_homography  # noqa: E402

TARGET_GROUPS = {
    "squaaaaw_collection_tour", "4QxpRG4E6d0", "ho7IFpPXI-A",
    "marinefordtcg_op12_two_boxes",
}
EXCLUDED_FLAGS = {
    "possible_parent_page_box_contains_3plus_detections",
    "short_edge_under_300px_provisional",
    "very_small_subregion_or_noncard_candidate",
    "low_sharpness_bottom_decile_provisional",
    "crop_clipped_at_source_edge_possible_partial_card",
    "large_source_region_review_possible_page_or_multiple_cards",
}
CARD_ID_RE = re.compile(r"\b(OP\s*\d{1,2}|ST\s*\d{1,2}|EB\s*\d{1,2}|P|PRB\s*\d{1,2})\s*[-_ ]\s*(\d{3})\b", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def normalize_card_id(text: str) -> list[str]:
    ids = []
    for prefix, number in CARD_ID_RE.findall(text.upper()):
        p = re.sub(r"\s+", "", prefix)
        ids.append(f"{p}-{number}")
    return list(dict.fromkeys(ids))


def ocr_text(image_path: Path) -> dict:
    """Read raw text with whole-crop and lower-title-band passes."""
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        # One Piece names are normally on the lower name bar. Retain the full
        # lower third and enlarge only for OCR; the original crop is untouched.
        band = rgb.crop((0, int(h * 0.62), w, h))
        band = band.resize((max(w, 900), max(int(h * .38), 360)), Image.Resampling.LANCZOS)
        full = rgb.copy()
    outputs = {}
    for tag, img, psm in [("full_psm6", full, "6"), ("lower_band_psm6", band, "6")]:
        import io
        buf = io.BytesIO(); img.save(buf, format="PNG")
        run = subprocess.run(["tesseract", "stdin", "stdout", "-l", "eng", "--psm", psm],
                             input=buf.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        outputs[tag] = run.stdout.decode("utf-8", "replace").strip()
    combined = "\n".join(x for x in outputs.values() if x)
    outputs["raw_text"] = combined
    outputs["observed_card_id_tokens"] = normalize_card_id(combined)
    return outputs


def load_references() -> list[dict]:
    refs: list[dict] = []
    cards_root = ROOT / "datasets/card_recognizer/cards"
    for path in sorted(cards_root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parts = path.relative_to(cards_root).parts
        if len(parts) < 2:
            continue
        card_id = path.stem
        base_id = re.sub(r"_p\d+$", "", card_id)
        meta_candidates = [path.with_suffix(".json"), path.parent / f"{card_id}.json", cards_root / parts[0] / f"{base_id}.json"]
        meta_path = next((m for m in meta_candidates if m.exists()), None)
        meta = json.loads(meta_path.read_text()) if meta_path else {}
        refs.append({"reference_path": str(path.relative_to(ROOT)), "card_id": card_id,
                     "catalog_card_id": meta.get("id", base_id), "name": meta.get("name", ""),
                     "set_code": parts[0], "reference_source": "existing_gallery",
                     "reference_sha256": sha256(path)})

    sync_root = ROOT / "runs/card_recognizer/onepiece_catalog_sync"
    manifest_path = sync_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for record in manifest.get("cards", []):
            rel = record.get("image_path")
            if not rel or record.get("status") not in {"downloaded", "present_local"}:
                continue
            path = sync_root / rel
            if not path.is_file():
                continue
            card_id = Path(rel).stem
            refs.append({"reference_path": str(path.relative_to(ROOT)),
                         "card_id": card_id, "catalog_card_id": record.get("card_id", card_id),
                         "catalog_identity": record.get("catalog_identity"),
                         "name": record.get("name", ""), "set_code": record.get("set_id", ""),
                         "reference_source": "current_english_catalog_sync",
                         "reference_sha256": sha256(path)})
    # Keep distinct art files even when they share a base catalog name.
    return refs


def name_candidates(text: str, refs: list[dict], max_names: int = 8) -> list[dict]:
    norm_text = normalize(text)
    tokens_text = set(re.findall(r"[A-Z0-9]{3,}", text.upper()))
    best_by_id: dict[str, tuple[float, dict]] = {}
    for ref in refs:
        name = ref.get("name", "")
        if not name:
            continue
        norm_name = normalize(name)
        name_tokens = set(re.findall(r"[A-Z0-9]{3,}", name.upper()))
        if not name_tokens:
            continue
        overlap = len(tokens_text & name_tokens) / len(name_tokens)
        seq = SequenceMatcher(None, norm_name, norm_text).ratio() if norm_name else 0.0
        contained = bool(norm_name and norm_name in norm_text)
        score = max(overlap, seq * 0.82, 1.0 if contained else 0.0)
        key = str(ref.get("catalog_card_id") or ref["card_id"])
        if score >= 0.35 and (key not in best_by_id or score > best_by_id[key][0]):
            best_by_id[key] = (score, ref)
    return [{"score": round(score, 4), "name": ref.get("name"),
             "catalog_card_id": ref.get("catalog_card_id"), "card_id": ref["card_id"]}
            for score, ref in sorted(best_by_id.values(), key=lambda x: -x[0])[:max_names]]


def candidate_references(ocr: dict, refs: list[dict]) -> list[dict]:
    observed_ids = set(ocr["observed_card_id_tokens"])
    raw = ocr["raw_text"]
    name_rank = name_candidates(raw, refs)
    ids = observed_ids | {str(x["catalog_card_id"]) for x in name_rank if x["score"] >= 0.52}
    candidates = [r for r in refs if r["card_id"] in ids or r.get("catalog_card_id") in ids]
    # Search refs sharing the most plausible name strings as well, to keep
    # art variants and repeated card names visible for feature verification.
    likely_names = {normalize(x.get("name", "")) for x in name_rank[:5]}
    candidates += [r for r in refs if normalize(r.get("name", "")) in likely_names]
    unique = {}
    for r in candidates:
        unique[r["reference_path"]] = r
    return list(unique.values()), name_rank


def match_one(query_path: Path, refs: list[dict]) -> list[dict]:
    from PIL import Image
    with Image.open(query_path) as qim:
        qrgb = np.asarray(qim.convert("RGB"))
    matches = []
    for ref in refs:
        rpath = ROOT / ref["reference_path"]
        try:
            with Image.open(rpath) as rim:
                rrgb = np.asarray(rim.convert("RGB"))
            stats = verify_homography(rrgb, qrgb)
        except Exception as exc:
            stats = {"accepted": False, "good_match_count": 0, "inlier_count": 0,
                     "inlier_ratio": 0.0, "reference_coverage": 0.0, "query_coverage": 0.0,
                     "reprojection_error_px": None, "quad_xy": None,
                     "rejection_reasons": [f"matcher_error:{type(exc).__name__}"]}
        matches.append({**ref, "sift_geometry": stats})
    matches.sort(key=lambda x: (x["sift_geometry"]["accepted"],
                               x["sift_geometry"]["inlier_count"],
                               x["sift_geometry"]["reference_coverage"],
                               x["sift_geometry"]["query_coverage"],
                               -((x["sift_geometry"]["reprojection_error_px"] or 9999))), reverse=True)
    return matches


def contact_sheets(proposals: list[dict], out: Path, per_page: int = 9) -> list[str]:
    output = []
    for start in range(0, len(proposals), per_page):
        page = proposals[start:start + per_page]
        cols, pair_w, img_w, img_h, text_h = 3, 560, 270, 330, 108
        rows = (len(page) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * pair_w, rows * (img_h + text_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, proposal in enumerate(page):
            x, y = (i % cols) * pair_w, (i // cols) * (img_h + text_h)
            for j, path_key in enumerate(("crop_path", "proposed_reference_path")):
                path = ROOT / proposal[path_key]
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((img_w - 8, img_h - 8), Image.Resampling.LANCZOS)
                    left = x + j * img_w + (img_w - im.width) // 2
                    top = y + (img_h - im.height) // 2
                    sheet.paste(im, (left, top))
            s = proposal.get("top_match_sift") or {}
            lines = [f"PAIR {proposal['pair_index']:03d} | PROPOSED / UNRESOLVED",
                     f"{proposal['source_group']} · frame {proposal.get('source_frame')} · {proposal['crop_id'][:28]}",
                     f"REF {proposal['proposed_card_id']} · {proposal['proposed_name'][:30]}",
                     f"SIFT inliers {s.get('inlier_count',0)}/{s.get('good_match_count',0)} accepted={s.get('accepted',False)}",
                     f"OCR {proposal.get('observed_card_id_tokens',[])} | {proposal.get('ocr_summary','')[:36]}"]
            draw.multiline_text((x + 3, y + img_h + 3), "\n".join(lines), fill="black", spacing=2)
        path = out / f"paired-contact-{start // per_page + 1:02d}.jpg"
        sheet.save(path, quality=88, optimize=True)
        output.append(str(path.relative_to(ROOT)))
    return output


def run(inventory_path: Path, output: Path, limit: int = 1000) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    inv = json.loads(inventory_path.read_text())
    candidates = []
    for row in inv["crops"]:
        if row.get("source_group") not in TARGET_GROUPS or not row.get("geometry_eligible"):
            continue
        if not row.get("is_contact_sheet_representative"):
            continue
        flags = set(row.get("review_flags", []))
        if flags & EXCLUDED_FLAGS:
            continue
        if row.get("aspect_width_over_height", 1) > 0.95:
            continue
        candidates.append(row)
    candidates = candidates[:limit]
    refs = load_references()
    proposals, ocr_rows, no_match = [], [], []
    for ix, row in enumerate(candidates, 1):
        crop_path = ROOT / row["crop_path"]
        ocr = ocr_text(crop_path)
        possible_refs, name_rank = candidate_references(ocr, refs)
        match_rows = match_one(crop_path, possible_refs) if possible_refs else []
        ocr_row = {"crop_id": row["crop_id"], "crop_path": row["crop_path"],
                   "source_group": row["source_group"], "source_frame": row.get("source_frame"),
                   "raw_ocr": ocr["raw_text"], "ocr_passes": ocr,
                   "observed_card_id_tokens": ocr["observed_card_id_tokens"],
                   "name_candidates": name_rank,
                   "reference_search_count": len(possible_refs),
                   "ranked_matches": match_rows[:5]}
        ocr_rows.append(ocr_row)
        if not match_rows:
            no_match.append(ocr_row); continue
        top = match_rows[0]
        stats = top["sift_geometry"]
        # Keep plausible feature correspondence proposals for the root to
        # verify. Accepted is from the existing conservative helper only.
        if not stats.get("accepted") and stats.get("inlier_count", 0) < 8:
            continue
        proposals.append({
            "pair_index": len(proposals) + 1,
            "status": "proposed_unresolved_root_verification_required",
            "crop_id": row["crop_id"], "crop_path": row["crop_path"],
            "crop_sha256": row["crop_sha256"], "source_group": row["source_group"],
            "source_frame": row.get("source_frame"), "time_seconds": row.get("time_seconds"),
            "source_frame_image": row.get("source_frame_image"),
            "source_frame_sha256": row.get("source_frame_sha256"),
            "source_bbox_xyxy": row.get("bbox"), "crop_width": row["width"],
            "crop_height": row["height"], "sharpness_rank_global": row.get("sharpness_rank_global"),
            "raw_ocr_text": ocr["raw_text"],
            "observed_card_id_tokens": ocr["observed_card_id_tokens"],
            "ocr_name_candidates": name_rank[:5],
            "proposed_reference_path": top["reference_path"],
            "proposed_reference_sha256": top["reference_sha256"],
            "proposed_card_id": top["card_id"],
            "catalog_card_id": top.get("catalog_card_id"),
            "proposed_name": top.get("name", ""),
            "reference_source": top["reference_source"],
            "top_match_sift": stats,
            "ranked_reference_candidates": [
                {"reference_path": x["reference_path"], "reference_sha256": x["reference_sha256"],
                 "card_id": x["card_id"], "catalog_card_id": x.get("catalog_card_id"),
                 "name": x.get("name"), "reference_source": x["reference_source"],
                 "sift_geometry": x["sift_geometry"]} for x in match_rows[:5]
            ],
            "candidate_notes": ["OCR is observed text only; SIFT homography is proposal evidence only.",
                                "Visual art/printing match must be verified against the reference image.",
                                "No label accepted; not promoted to training or evaluation."]
        })

    # Retain one crop per base ID only as a proposal list; alternative printings
    # remain visible in ranked_reference_candidates for exact-art verification.
    proposals.sort(key=lambda p: (bool(p["top_match_sift"].get("accepted")),
                                  p["top_match_sift"].get("inlier_count", 0),
                                  p["top_match_sift"].get("reference_coverage", 0),
                                  p["sharpness_rank_global"] or 10**9), reverse=True)
    unique = []
    used_base = set()
    for p in proposals:
        base = re.sub(r"_p\d+$", "", p["proposed_card_id"])
        if base in used_base:
            continue
        used_base.add(base)
        unique.append(p)
    unique = unique[:50]
    for i, p in enumerate(unique, 1): p["pair_index"] = i
    sheets = contact_sheets(unique, output)
    known_sanity = None
    sanity_query = next((r for r in inv["crops"] if r["crop_id"] == "e1c37c8026df_frame_000000060_p0_det_000"), None)
    if sanity_query:
        target_refs = [r for r in refs if r.get("catalog_card_id") == "OP12-115" or r["card_id"] == "OP12-115"]
        if target_refs:
            sanity_crop = ROOT / sanity_query["crop_path"]
            sanity_ocr = ocr_text(sanity_crop)
            sanity_matches = match_one(sanity_crop, target_refs)
            if sanity_matches:
                best = sanity_matches[0]
                known_sanity = {
                    "status": "root_cued_reference_correspondence_sanity_check_unresolved",
                    "note": "Root supplied the visible OP12-115 cue; this row only reports a reference comparison and does not independently assign or accept that identity.",
                    "crop_id": sanity_query["crop_id"], "crop_path": sanity_query["crop_path"],
                    "crop_sha256": sanity_query["crop_sha256"], "source_group": sanity_query["source_group"],
                    "source_frame": sanity_query.get("source_frame"),
                    "source_frame_image": sanity_query.get("source_frame_image"),
                    "raw_ocr_text": sanity_ocr["raw_text"],
                    "observed_card_id_tokens": sanity_ocr["observed_card_id_tokens"],
                    "reference_path": best["reference_path"],
                    "reference_sha256": best["reference_sha256"],
                    "catalog_card_id_from_root_cue": "OP12-115",
                    "sift_geometry": best["sift_geometry"],
                    "ranked_reference_candidates": [
                        {"reference_path": x["reference_path"], "reference_sha256": x["reference_sha256"],
                         "card_id": x["card_id"], "name": x.get("name"), "sift_geometry": x["sift_geometry"]}
                        for x in sanity_matches[:3]
                    ]
                }
    payload = {
        "schema_version": 1, "artifact_type": "proposed_onepiece_crop_reference_correspondence",
        "inventory_path": str(inventory_path.relative_to(ROOT)),
        "reference_roots": ["datasets/card_recognizer/cards/", "runs/card_recognizer/onepiece_catalog_sync/cards/"],
        "reference_sync_manifest": "runs/card_recognizer/onepiece_catalog_sync/manifest.json",
        "reference_gallery_count": len(refs),
        "source_groups": sorted(TARGET_GROUPS),
        "source_group_filter": "English One Piece source groups requested by root; explicit Japanese collection excluded",
        "candidate_filter": {"geometry_eligible": True, "best_cluster_member_only": True,
                              "excluded_flags": sorted(EXCLUDED_FLAGS),
                              "excluded_aspect": "width/height > 0.95",
                              "short_edge_min": 300,
                              "note": "Advisory prefilter only; no crop is deleted or labeled."},
        "match_method": "tesseract observed-text passes (whole crop + enlarged lower band), catalog name/code shortlist, then scripts.verify_card_geometry.verify_homography SIFT/RANSAC geometry. No DINO evaluation.",
        "identity_policy": "all rows are unresolved proposals; reference art variant must visually match; do not infer or force _pN identity",
        "counts": {"candidate_crops_after_quality_filter": len(candidates),
                   "crops_with_catalog_shortlist": sum(bool(x["name_candidates"] or x["observed_card_id_tokens"]) for x in ocr_rows),
                   "crops_with_sift_proposals": len(proposals), "unique_proposed_card_ids": len(unique),
                   "paired_contact_sheets": len(sheets)},
        "contact_sheets": sheets,
        "proposals": unique,
        "known_reference_sanity_pair": known_sanity,
        "ocr_inventory_all_candidates": ocr_rows,
        "no_reference_shortlist": no_match,
        "known_reference_sanity_case": {
            "crop_id": "e1c37c8026df_frame_000000060_p0_det_000",
            "note": "Root already observed OP12-115 text; excluded from proposed pair count and not assigned here. Its reference may be added as a separate visual sanity pair after matching.",
            "identity_status": "unresolved_by_this_worker",
            "pair_record_path": "known_reference_sanity_pair"
        },
        "training_promotion": "not_promoted"
    }
    (output / "proposals.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "proposals.csv").open("w", newline="") as f:
        fields = ["pair_index", "status", "crop_id", "crop_path", "crop_sha256", "source_group",
                  "source_frame", "time_seconds", "source_frame_image", "source_frame_sha256",
                  "source_bbox_xyxy", "crop_width", "crop_height", "sharpness_rank_global",
                  "raw_ocr_text", "observed_card_id_tokens", "proposed_reference_path",
                  "proposed_reference_sha256", "proposed_card_id", "catalog_card_id",
                  "proposed_name", "reference_source", "top_match_sift"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for p in unique:
            w.writerow({**p, "source_bbox_xyxy": json.dumps(p["source_bbox_xyxy"]),
                        "observed_card_id_tokens": json.dumps(p["observed_card_id_tokens"]),
                        "top_match_sift": json.dumps(p["top_match_sift"])})
    return payload["counts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "runs/dataset_cleanup/inventory/inventory.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/dataset_cleanup/onepiece_reference_pairs")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    inventory = args.inventory if args.inventory.is_absolute() else ROOT / args.inventory
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(run(inventory, output, args.limit), indent=2))


if __name__ == "__main__":
    main()
