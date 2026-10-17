# Curated dataset layout

`scripts/curate_datasets.py` copies reviewed inputs into a namespaced layout at
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
  detection/                         # optional synthetic import only
    dataset.yaml
    manifest.json
    {train,val,test}/{images,labels,annotations}/

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

`--synthetic-detector` only imports a synthetic renderer directory named
`synthetic` below `runs/card_detector/<run>/`. The importer validates scene
images, matching labels, counts and dimensions, true renderer quadrilaterals,
and derived OBB corner annotations. It writes YOLO split paths that point to
`train/images`, `val/images`, and `test/images`; only renderer-produced train
and validation scenes are populated. It records
`real_camera_validation: false`. Real unreviewed scenes are not promoted into
detector ground truth, and partially annotated multi-card scenes are not
imported as complete labels.
The generated YAML records an absolute local dataset root so Ultralytics does
not resolve a relative root under its configured download directory. Recreate
the YAML after moving the dataset. Annotation image/label paths are rebased to
the detection root; original renderer annotations remain unchanged.

## Example

From the repository root, after placing the manifests and source images under
`runs/video_mining/<review-batch>/`:

```bash
.env/bin/python scripts/curate_datasets.py \
  --input-root runs/video_mining/<review-batch> \
  --references-manifest runs/video_mining/<review-batch>/references.json \
  --review-manifest runs/video_mining/<review-batch>/review.json \
  --synthetic-detector runs/card_detector/synthetic
```

`datasets/curated/` is ignored because it contains local derived data. The current
executed snapshot and its limits are recorded in [DATASET_CLEANUP.md](DATASET_CLEANUP.md).
