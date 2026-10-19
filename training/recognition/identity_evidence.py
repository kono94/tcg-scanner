"""Conservative OCR and retrieval evidence for card identity proposals.

This module does not perform OCR or verify ground truth. OCR strings and catalog
tokens are supplied by callers; outputs are candidate/review states only.
"""
from __future__ import annotations

import re
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


_PATTERNS = (
    ("onepiece", "printed_number", re.compile(r"\b(?:OP|ST|EB|PRB|P)-?\d{1,3}-\d{1,3}\b|\bP-\d{1,3}\b", re.I)),
    ("pokemon", "collector_number", re.compile(r"\b(?:[A-Z]{1,3})?\d{1,3}\s*/\s*(?:[A-Z]{1,3})?\d{1,3}\b", re.I)),
    ("yugioh", "set_code", re.compile(r"\b[A-Z0-9]{2,6}-[A-Z0-9]{2,6}\b", re.I)),
    ("yugioh", "passcode", re.compile(r"(?<!\d)\d{8}(?!\d)")),
)


def extract_printed_codes(text: str) -> list[dict[str, str]]:
    """Extract conservative game-scoped printed tokens from OCR text.

    Pokémon collector numbers retain their denominator (for example ``TG01/30``);
    a bare number is intentionally not treated as an identifying token.
    Yugioh set codes and eight-digit passcodes have distinct ``kind`` values.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    found: list[tuple[int, dict[str, str]]] = []
    for game, kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            token = re.sub(r"\s*/\s*", "/", match.group(0)).upper()
            if game == "onepiece":
                # Preserve the printed set prefix and digits exactly; only normalize case.
                pass
            elif game == "yugioh" and kind == "set_code":
                # These prefixes are also valid One Piece set prefixes; do not double-tag.
                if re.match(r"^(?:OP|ST|EB|PRB|P)-?\d", token):
                    continue
            found.append((match.start(), {"game": game, "kind": kind, "token": token}))
    found.sort(key=lambda item: (item[0], item[1]["game"], item[1]["kind"]))
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for _, token in found:
        key = (token["game"], token["kind"], token["token"])
        if key not in seen:
            seen.add(key)
            unique.append(token)
    return unique


def _normalize_token(value: Any) -> dict[str, str] | None:
    if isinstance(value, str):
        # Catalog metadata must identify token type explicitly; no guessing from ID/name.
        return None
    if not isinstance(value, Mapping):
        return None
    game, kind, token = value.get("game"), value.get("kind"), value.get("token")
    if not all(isinstance(item, str) and item for item in (game, kind, token)):
        return None
    return {"game": game.lower(), "kind": kind.lower(), "token": token.upper()}


def assess_identity_evidence(
    shortlist: Sequence[Mapping[str, Any]],
    ocr_texts: str | Iterable[str],
    catalog_metadata: Mapping[str, Mapping[str, Any]],
    *,
    min_similarity: float = 0.75,
    min_margin: float = 0.05,
) -> dict[str, Any]:
    """Compare distinct-identity retrieval results with exact OCR/catalog tokens.

    Each shortlist row has ``identity`` and ``similarity``; duplicate template rows
    for one identity are collapsed to that identity's maximum similarity. Catalog
    metadata is keyed by identity and supplies ``expected_printed_tokens`` (token
    objects with game/kind/token) plus optional ``printing_variant``. Thresholds are
    configurable but uncalibrated defaults; no result represents verified truth.
    """
    if not 0 <= min_similarity <= 1 or not 0 <= min_margin <= 2:
        raise ValueError("thresholds must be within similarity [0,1] and margin [0,2]")
    if isinstance(ocr_texts, str):
        ocr_texts = [ocr_texts]
    ocr_tokens: list[dict[str, str]] = []
    seen_ocr: set[tuple[str, str, str]] = set()
    for text in ocr_texts:
        for token in extract_printed_codes(text):
            key = (token["game"], token["kind"], token["token"])
            if key not in seen_ocr:
                seen_ocr.add(key)
                ocr_tokens.append(token)

    scores: dict[str, float] = {}
    for row in shortlist:
        identity = row.get("identity")
        score = row.get("similarity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("each shortlist row needs a non-empty identity")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("similarity must be finite")
        score = float(score)
        if score < -1.000001 or score > 1.000001:
            raise ValueError("similarity must be within cosine range [-1,1]")
        score = min(1.0, max(-1.0, score))
        scores[identity] = max(scores.get(identity, float("-inf")), score)
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if not ranked:
        return _result("unresolved/visual_candidate", None, None, ocr_tokens, [], "empty_retrieval_shortlist", [])

    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else None
    margin = best_score - second_score if second_score is not None else None
    gates_pass = best_score >= min_similarity and margin is not None and margin >= min_margin
    meta = catalog_metadata.get(best_id, {})
    expected = [token for value in meta.get("expected_printed_tokens", []) if (token := _normalize_token(value))]
    expected_keys = {(item["game"], item["kind"], item["token"]) for item in expected}
    observed_keys = {(item["game"], item["kind"], item["token"]) for item in ocr_tokens}
    matching = expected_keys & observed_keys
    token_identities: dict[tuple[str, str, str], set[str]] = {}
    for identity, identity_meta in catalog_metadata.items():
        for value in identity_meta.get("expected_printed_tokens", []):
            token = _normalize_token(value)
            if token:
                key = (token["game"], token["kind"], token["token"])
                token_identities.setdefault(key, set()).add(identity)
    ambiguous_tokens = sorted(
        ("/".join(key), sorted(token_identities.get(key, set())))
        for key in matching if len(token_identities.get(key, set())) > 1
    )
    # Yu-Gi-Oh passcodes identify a card face, but do not establish a printing/set.
    printing_matching = {key for key in matching if not (key[0] == "yugioh" and key[1] == "passcode")}
    mismatching_same_type = {
        (item["game"], item["kind"]) for item in expected
    } & {
        (item["game"], item["kind"]) for item in ocr_tokens
    }
    has_conflicting_token = any(
        (item["game"], item["kind"]) in mismatching_same_type
        and (item["game"], item["kind"], item["token"]) not in expected_keys
        for item in ocr_tokens
    )
    limitations: list[str] = []
    variant = str(meta.get("printing_variant", ""))
    onepiece_base_match = any(game == "onepiece" and kind == "printed_number" for game, kind, _ in matching)
    sibling_variant = False
    p1_base_only = onepiece_base_match and "_p1" in best_id.lower()
    if onepiece_base_match and (p1_base_only or variant):
        for identity, sibling_meta in catalog_metadata.items():
            if identity == best_id:
                continue
            sibling_tokens = {
                (tok["game"], tok["kind"], tok["token"])
                for value in sibling_meta.get("expected_printed_tokens", [])
                if (tok := _normalize_token(value))
            }
            sibling_variant = sibling_variant or bool(sibling_tokens & matching)
        limitations.append("One Piece printed base number does not establish the _p1 printing/art variant")

    if has_conflicting_token:
        status, reason = "conflicting_evidence", "readable_ocr_token_disagrees_with_best_identity_catalog_token"
    elif ambiguous_tokens:
        status, reason = "unresolved/visual_candidate", "exact_token_is_shared_by_multiple_catalog_identities"
        limitations.append("Observed printed token is expected by multiple catalog identities")
    elif sibling_variant or p1_base_only:
        status, reason = "unresolved/visual_candidate", "base_number_shared_across_printing_variants"
    elif printing_matching and gates_pass:
        status, reason = "corroborated_candidate", "best_identity_and_exact_expected_token_agree"
    elif not gates_pass:
        status, reason = "unresolved/visual_candidate", "retrieval_similarity_or_distinct_identity_margin_below_configured_gate"
    elif not ocr_tokens or not expected:
        status, reason = "unresolved/visual_candidate", "printed_token_evidence_absent_or_catalog_expectation_missing"
    elif not matching:
        status, reason = "unresolved/visual_candidate", "ocr_token_not_exactly_matched_to_catalog_expectation"
    else:
        status, reason = "unresolved/visual_candidate", "insufficient_evidence"
    if not gates_pass and status == "corroborated_candidate":
        status, reason = "unresolved/visual_candidate", "retrieval_gates_not_met"
    if not matching and ocr_tokens and expected and not mismatching_same_type:
        limitations.append("OCR evidence did not exactly match an expected catalog token")
    if not expected:
        limitations.append("catalog metadata has no expected printed tokens for the retrieval leader")
    if matching and not printing_matching:
        limitations.append("Yu-Gi-Oh passcode alone does not establish the card printing/set code")
    return _result(status, best_id, best_score, ocr_tokens, expected, reason, limitations, margin=margin, gates_pass=gates_pass, ambiguous_tokens=[{"token": key, "identities": identities} for key, identities in ambiguous_tokens])


def _result(status: str, identity: str | None, score: float | None, observed: list[dict[str, str]], expected: list[dict[str, str]], reason: str, limitations: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "best_identity": identity,
        "best_similarity": score,
        "observed_tokens": observed,
        "expected_tokens": expected,
        "reason": reason,
        "limitations": limitations,
        "ground_truth_verified": False,
        "thresholds_calibrated": False,
        **extra,
    }
