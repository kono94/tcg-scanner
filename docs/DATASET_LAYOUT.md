# Curated dataset layout

Where `datasets/curated` sits among the other data folders: [datasets/README.md](../datasets/README.md).

`scripts/curation/curate_datasets.py` copies reviewed inputs into a namespaced layout at
`datasets/curated/`. It does not assign labels, infer missing identity fields,
or claim human verification. Run it only after the root reviewer has prepared
both manifests and inspected the accepted crops. Existing corpora and notebooks
remain untouched. The curated output is derived and can be rebuilt from its
manifests and source files.

## Input manifests

All paths in either manifest are clean relative paths under `--input-root`.
Every image path has a lowercase SHA-256 digest and must decode with Pillow.
Symlinks, traversal, paths outside the input root, and mismatched hashes fail
curation.

Reference manifest (`schema_version: 1`):

```json
{
  "schema_version": 1,
  "references": [{
    "path": "gallery/dp1-51.webp",
    "sha256": "<64 lowercase hex characters>",
    "game": "pokemon",
    "language": "en",
    "card_id": "dp1-51",
    "printing": "unknown",
    "finish": "unknown"
  }]
}
```

Review manifest (`schema_version: 1`) requires `reviewer: "root-sol"`, a
nonempty `review_provenance`, and `queries` and optional `rejected` arrays.
Every accepted query must set `status` to `verified_reference_match`,
`single_complete_card` to `true`, and provide `split` (`train`, `val`, or
`test`), `source_group`, crop/source/reference paths and SHA-256 values, an
explicit `{game, language, card_id}` identity, and nonempty
`annotation_provenance` and `annotation_notes`. The referenced file and digest
must match a row in the reference manifest with the same identity. Extra fields
are retained verbatim in the output sidecar, including printing/finish
uncertainty, crop transforms, rectification provenance, and raw-crop pointers.
Use a source group for the underlying video/session so neighboring frames from
the same source cannot leak across splits. The tool also rejects identical
query bytes across splits and evaluation query bytes identical to any gallery
reference.

Rejected rows are informational unless the explicit deletion option is used.
For deletion, each row needs `status: "rejected"`, `crop_path` relative to the
allowlisted derived-crop root, and `crop_sha256`. The tool validates all paths
and hashes first and removes only those exact image files after curated copies
have been written.

## Output

```text
datasets/curated/
  curation_manifest.json
  recognition/
    manifest.json
    review_audit.json               # exhaustive root decisions/provenance
    exact_reference_aliases.json
    gallery/{game}/{language}/{card_id}/reference-{sha12}.{ext}[.json]
    {train,val,test}/{game}/{language}/{card_id}/query-{sha16}.{ext}[.json]

datasets/staging/recognition/         # excluded from training
  manifest.json
  {source_group}/candidate-{sha24}.{ext}[.json]
```

Staging holds only reviewed exclusions: `reviewed_unresolved` and
`verified_identity_unusable_geometry`, with explicit reasons and any verified
identity. Accepted staging links are retired after curation; bad derived crops
are hash-checked and removed, while their audit records and source frames remain.

Sidecars retain source records and provenance. Recognition identities remain
namespaced by game, language, and catalog ID; physical printing/finish stays
unknown unless the source record explicitly establishes it. Gallery references
are copied to a separate gallery tree, never mixed into query splits. The
output directory must be new or empty, inside the repository, and must not
contain or overlap any input source. Default mode copies files; `--copy-mode
hardlink` is available where the filesystem supports it.
Treat linked image bytes as immutable: modifying a hardlink in place would also
modify the original. Write transformations to a new artifact instead.

## Detector data

Detector training data is not curated here. `scripts/detection/generate_detection_scenes.py`
renders synthetic scenes with exact corners, and `scripts/detection/build_corner_dataset.py`
converts them, together with the AI-reviewed real corners in
`datasets/detection/real-corners/`, to a YOLO-pose dataset under `runs/detection/`. Its YAML
records an absolute dataset root; recreate it after moving the dataset. The dataset behind
a deployed detector is versioned in `datasets/detection/`. Unreviewed real scenes are not
detector ground truth, and partially annotated multi-card scenes are not complete labels.

## Example

Rebuild the current snapshot from its review records (this reproduces `datasets/curated`
exactly; verified 2026-10-19):

```bash
.env/bin/python scripts/curation/curate_datasets.py \
  --input-root . \
  --references-manifest datasets/review/pass2/references_v5.json \
  --review-manifest datasets/review/pass2/sol_review_merged.json \
  --output datasets/curated-next --copy-mode hardlink
```

A new review batch adds its merged manifests under `datasets/review/` and runs the same
command; then replace `datasets/curated`, `dvc add` it and commit the pointer. History and
counts of the executed snapshot: [DATASET_CLEANUP.md](DATASET_CLEANUP.md).
