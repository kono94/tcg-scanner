# Stable-frame scanning policy

Status: default capture architecture since 2026-10-15, implemented in the browser scanner (`web/src/capture/`; batch workflow in `web/src/batch/`, see `web/README.md`). Not yet verified on a physical phone. Applies to both single cards and nine-card binder pages.

Scene card count is variable, including zero and more than nine. Nine-card pages are one important test case. Capture eligibility, detection output, result lists, and collected annotations must not require nine instances or a fixed pocket arrangement. See [the data contract](DATA_COLLECTION.md).

## Decision

Prefer stable-frame capture for the first scanner: wait for a sufficiently steady image, capture one high-resolution frame, detect all visible cards, rectify their crops, recognize each independently, and review results on that captured image. Continuous per-card object tracking is optional future work, not a prerequisite.

This separates camera feedback from expensive inference. It removes cross-frame card association and moving-overlay alignment from the first implementation. It does not remove the need for good localization, resolution, unknown rejection, bounded asynchronous work, or page/save identity.

## Cycle

```text
live preview + lightweight scene-stability sampling
  → steady dwell / conditional countdown
  → immutable high-resolution capture
  → detection on that capture
  → per-card quality check + corner warp
  → queued / recognizing / verifying / confirmed or needs-attention
  → review captured image + independent card results
  → save once / retry capture / next page
```

All boxes, corners, crops, and results in a batch refer to the same capture. Number physical instances once in capture reading order. Two identical cards in different pockets remain two copies. The UI switches to a clearly labeled captured image for processing/review; it must not draw snapshot boxes on a changing live preview.

Keep camera/session handling responsive. Whether the camera stream stays active during review is a device/resource policy; the displayed captured image remains fixed. Camera movement after capture does not invalidate that image or cancel useful recognition. Explicit cancel/reset/next-page actions invalidate the relevant batch generation.

## User controls and proposed starting values

These are initial tunable settings, not validated device thresholds.

| Setting | Initial value | Meaning |
| --- | --- | --- |
| Trigger mode | Auto when steady | One capture, then stop for review; no endless automatic rescanning |
| Other modes | Manual; Repeat while steady | Manual snapshot or optional periodic replacement captures of the active page |
| Steady duration | 0.8 seconds, adjustable 0.4–2 seconds | Continuous acceptable scene stability before capture; countdown resets on motion |
| Repeat interval | 3 seconds, adjustable 1–10 seconds | Minimum cooldown after the previous batch settles; never an inference-completion promise |
| Motion tolerance | Normal; optional sensitive/relaxed presets | Calibrated image-motion threshold, not camera-sensor-only movement |
| Manual capture | Always available in live preview | Bypasses the automatic stability dwell, not later quality validation |
| Pause auto-scan | Available | Disarms auto triggers; explicit cancel controls in-flight work separately |

An optional repeat cycle is eligible only when the previous batch has settled, the cooldown has elapsed, the scene has been steady for the configured dwell, the page is active, and the app is visible/unpaused. Use monotonic deadlines. Do not enqueue missed timer events or catch up with multiple captures after a slow batch.

Top-bar examples:

- `Hold steady` when motion prevents acquisition.
- `Scan in 0.8 s` during the steady dwell; movement resets it.
- `Next scan in 2.4 s` in armed repeat mode, conditional on remaining steady; the remaining delay is the maximum of cooldown and steady-dwell requirements.
- `Capturing`, `Finding cards`, then `Checking 4 of 9` while actual work runs. Do not show an expired countdown as if a second batch will start during processing.
- `8 ready · 1 needs attention` when the batch settles. Stop activity indicators; no invented time estimate for unresolved work.
- `Review page` after the default one-shot auto capture; no next-scan countdown until the user rearms capture or selects repeat mode.

Countdown labels are conditional capture eligibility, not a promise that camera acquisition or recognition finishes at zero. At expiry, recheck the latest sample and snapshot quality before accepting the capture. If pause, app hiding, missing/stale frames, changed configuration, or camera interruption occurs, disarm/reset the dwell and require fresh samples when resumed.

## Stability is a cheap scene test

Sample a downscaled image at a modest rate (initial experiment: 5–10 samples/second). Compare recent scene motion/appearance and assess blur/exposure. This can use frame/block differences or global motion estimates; it does not require detecting or tracking each card. Use continuous timestamps rather than counting samples as if cadence were guaranteed.

Image evidence is essential: the phone can be stationary while a binder page or hand moves. Gyroscope/accelerometer signals alone cannot establish a still scene and are optional supporting signals, not a required browser permission. Test autoexposure/focus changes, lighting flicker, subtle hand movement, textured sleeves, and stale/frozen preview frames. Do not interpret a repeated old frame as a new stable observation.

Being steady does not imply that nine cards are sharp, large enough, unoccluded, or free of glare. Those per-card checks happen after detection. The countdown is not a quality guarantee. Empty scenes can pass the stability test; detection must return a truthful no-card result and allow another attempt.

## Bounded work and batch identity

- At most one capture batch is active, with no queued future batches. The stability monitor keeps only a bounded low-resolution window/latest samples.
- Retain one immutable high-resolution source and lazily extract/process crops from its variable-length detection list under a memory cap. Use one recognition worker initially; compare bounded batches only after measuring cost. Nine-card support is an initial benchmark, not a nine-instance allocation limit. Larger counts use bounded chunks; report resource-limited partial results explicitly.
- Tag results with session/page generation, capture/batch ID, instance ID, and model/index versions. Callbacks may update only their existing compatible batch; they cannot create a new page or resurrect a cancelled batch.
- Within the batch, instances finish independently and fairly. Timeouts become explicit needs-attention states so a stuck verifier cannot block the next cycle forever. Recognition confirmation can use calibrated retrieval/detail evidence from the snapshot; repeated inference on the same pixels is not independent temporal evidence.
- Save only by an explicit idempotent page/capture revision policy. A periodic scan is another observation of the active page, not nine additional physical cards. Saving the same page twice does not increase quantities.
- A repeat capture replaces the active page's candidate result revision atomically. Do not union old and new detections or match pockets by raw screen position. Keep earlier saved revisions inspectable; do not silently overwrite a saved page. Updating a saved page is explicit.
- Freeze automatic replacement while the user reviews details or edits a result; require an explicit return to scanning. An explicit next-page boundary creates a new page generation. Scene motion alone does not establish a new page.

## Batch workflow (browser pilot, implemented 2026-10-17)

An optional "review at the end" workflow for binders: each capture is a new page. After detection the
straightened crops of every card are stored and the capture is released at once, so acquisition of
the next page starts immediately and nothing capture-sized is retained between pages. Recognition
runs later from the stored crops, one card at a time, and results are written into the saved page.
Still one active capture; the background queue is bounded by admission control before a capture
(backlog limit), persisted with the saved pages and resumed after a reload. In auto mode the next
automatic capture waits for a page turn (scene motion followed by a steady view that differs from
the previous capture); manual capture stays available. Review happens per card afterwards. See
`web/README.md`; thresholds are uncalibrated and the workflow is not device-verified.

## Retries without tracking

Retry recognition/verification on the same crop when appropriate, or recapture the whole active page as a new candidate revision. For a closer/detail capture, first select the target instance; bind that capture to the selected pocket explicitly and require review if other identities conflict. Do not automatically merge a new close-up into old page coordinates. Cross-capture automatic reassociation is deferred with continuous tracking.

## Acceptance and measurements

- Deterministic timestamped stability samples verify dwell, motion reset, cooldown, setting changes, manual mode, repeat mode, pause/resume, and stale-frame rejection.
- Delayed/failing fake detection/recognition verifies one active batch, no timer backlog, independent progress for nine cards, timeout settlement, and generation guards on cancel/reset/next-page.
- UI tests verify truthful countdown/processing/review transitions and geometry tied exclusively to its captured image. Motion after capture must not move snapshot boxes or corrupt results.
- Repeated scans and saves of a page containing duplicate card IDs preserve physical quantities without accumulating duplicates. Partial results and blank pockets remain explicit.
- Benchmark capture eligibility delay separately from snapshot acquisition, detection, first confirmed result, all-instance confirmation, and partial-scene settlement, grouped by actual card count/layout (including nine-card pages). Cooldown is not inference latency; report both total user wait and pipeline timing.
- Physical-device tests measure false triggers, failure to trigger while reasonably steady, image quality, retained memory, and battery/thermal behavior on single cards and binder pages.

## Technical references

- [MDN video-frame callbacks](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback): per-frame metadata/timing is available, but callbacks do not guarantee precise synchronization. Feature-detect and handle fallback/stale samples.
- [MDN track settings](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/getSettings): inspect actual selected camera settings rather than assuming requested resolution or cadence was granted.
- [OpenCV motion analysis](https://docs.opencv.org/4.13.0/d7/df3/group__imgproc__motion.html): global motion estimation is an available implementation technique; it is not a requirement to ship OpenCV in the browser.

The policy, defaults, and expected engineering simplifications are proposals. They have not yet been benchmarked or wired into either client.
