# Card data collection and annotation

Status: data contract with an implemented automatic synthetic bootstrap. `scripts/generate_detection_scenes.py` (renderer `training/card_detector/scene_renderer.py`) produces scenes with exact renderer corners; `scripts/build_corner_dataset.py` converts them, plus AI-reviewed real corners, to YOLO-pose labels. Synthetic scenes do not constitute real-camera validation. The v1 generator `generate_synthetic_cards.py` is archived.

## Variable card count

Capture scenes with **any number of cards, including zero**. Nine-card binder pages are an important use case and benchmark slice, not a fixed model input, required layout, annotation length, collection quota per image, or maximum supported count. Do not stage every image as a full 3 × 3 grid.

Cover single cards, loose groups, scattered/overlapping cards, partially filled binder pages, different pocket layouts, and scenes containing more than nine cards. Include empty backgrounds, backs, hands, sleeves, pocket seams, and other non-card objects for detection/rejection testing. Vary card size, angle, lighting, blur, glare, device, and capture distance independently of card count.

Use count buckets such as 0, 1, 2–4, 5–8, 9, and more than 9 to inspect coverage and failures, not as restrictions on acceptable captures. Preserve useful natural captures even when their count is not a target bucket.

## Three different data units

| Unit | Contents | Card-count contract |
| --- | --- | --- |
| Scene capture for localization / end-to-end evaluation | Full source image/frame with dimensions, orientation, session/source provenance, and annotations | A variable-length list of actual visible card instances; zero is valid |
| Recognition training/evaluation query | One card crop derived from an annotated instance, with source capture and identity/visibility labels | One card per query crop; any number of crops may come from a scene |
| Enrolled gallery reference | A canonical or curated image of one identity, linked to metadata/provenance | One identity per reference; multiple references per identity are optional |

The encoder's tensor mini-batch size is a resource/training setting. It is unrelated to how many cards a source photograph contains. A fixed-size image tensor does not imply a fixed number of detected objects. Never pad a scene to nine fake cards or truncate its annotations at nine.

## Instance annotations

Store an instance ID, geometry, optional exact catalog identity/game, visibility/occlusion and quality flags for each actual card. Two physical copies of one catalog identity receive separate instance IDs. Preserve original frames and geometry so localization and cropping can be evaluated separately from retrieval.

Use explicit coordinates. The illustrative schema below uses oriented source-image pixels, top-left origin, with four ordered corners around the card. This example has two instances; the list may be empty or any other length. A bounding box can be stored when true corners are not annotated, but must not be treated as corner ground truth.

```json
{
  "capture_id": "capture-0123",
  "session_id": "session-a",
  "width": 1200,
  "height": 900,
  "coordinate_space": "oriented_source_pixels_top_left",
  "instances": [
    {
      "instance_id": "instance-a",
      "card_id": "onepiece:EB01-001",
      "corners_tl_tr_br_bl": [[100, 100], [350, 100], [350, 500], [100, 500]],
      "identity_label_status": "verified",
      "occlusion": "none"
    },
    {
      "instance_id": "instance-b",
      "card_id": null,
      "corners_tl_tr_br_bl": [[600, 200], [850, 180], [880, 570], [620, 600]],
      "identity_label_status": "unlabeled",
      "occlusion": "none"
    }
  ]
}
```

An unlabeled identity is not a verified not-enrolled example. Keep label status, known exact identity, and membership in a particular versioned gallery separate. Annotate partial/occluded cards explicitly; define visibility eligibility and ignore regions before evaluation rather than dropping hard examples afterward. Do not invent hidden corners as if directly observed.

For detection/rectification training, distinguish physical card edges from sleeve/pocket edges and preserve true perspective quadrilaterals. A minimum-area rotated rectangle may be derived for an OBB experiment, but retain the original corners: OBB fitting loses perspective geometry. Existing axis-aligned annotations cannot be promoted to corner ground truth by expanding their boxes. Document the rounded-edge/intersection convention and per-corner visibility/confidence. Record the corner-index convention and upright artwork orientation separately; `tl/tr/br/bl` must have a defined meaning for rotated/upside-down cards.

## Collection and splitting

- Start implementation using automatic synthetic compositing of existing card fronts onto procedural binder/desk backgrounds, with known transformations and camera effects. No manual labeling or new captures are required to generate bootstrap data. Preserve true quads separately from derived OBB rectangles; hold source identity/hash groups out of training validation. Model-generated backgrounds or real-domain pseudo-label proposals can be added later, with provenance and uncertainty rather than treating an LLM guess as geometry truth.
- Expand independent real-domain validation alongside the synthetic bootstrap, using automated source ingestion, teacher proposals, and consistency checks where possible. Missing real captures do not block implementation or synthetic training.
- Collect diverse natural scenes and readable close-up queries; full-page images alone do not supply all the detail needed for exact variant training/evaluation.
- Label all relevant card instances in a scene, not only the successfully recognized ones. Record empty pockets as scene context when useful; they are not card instances.
- Keep source capture/session IDs on every derived crop. All crops, augmentations, and adjacent video frames from one source session stay in one split. Check exact/near duplicates between encoder training, calibration, and locked tests.
- Keep independent camera queries separate from gallery references. Hold identities out of task-specific encoder training, enroll their references for retrieval evaluation, and retain genuinely not-enrolled identities for rejection evaluation.
- For nine-card binder evaluation, deliberately include full and partial pages and duplicate physical copies. Also collect other counts/layouts so the detector does not learn that cards only occur in nine fixed slots.

## Runtime and evaluation consequences

Detection returns a variable-length instance list. Schedule a bounded amount of work over that list, using chunks or lazy crop extraction where necessary. A runtime resource cap is explicit/configurable and must produce a truthful partial-result/closer-capture message; it must not silently discard cards above nine.

UI counts come from actual detections and states, never an assumed grid size. `N detected, K ready` does not prove that the detector found every physical card; optional expected counts remain user hints. Handle zero-card captures as a normal no-cards result.

Report localization recall, recognition precision/coverage, unknown handling, time/memory versus actual card count and source pixels per card, and all-eligible-instances-correct scene rate. Include missed ground-truth instances in end-to-end denominators according to the predefined visibility policy. Report nine-card binder performance as one named slice alongside the other counts/layouts.
