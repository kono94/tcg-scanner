# Scanner interaction and overlay specification

Status: updated 2026-10-16; implemented in the browser scanner pilot (`web/src/ui/`), verified only on desktop Chrome with a test-video mock. It does not claim device-verified camera behavior or measured recognition accuracy.

## Primary use cases

Single cards and multi-card scenes are equally important from the first implementation. A typical nine-card binder page is a required scenario, not a fixed input or maximum. Support zero, one, and variable multiple detections, other pocket layouts, partially filled pages, loose cards, and more than nine cards under explicit resource budgets. A grid is an optional framing aid, not ground truth or a recognition shortcut. Collection/annotations follow `DATA_COLLECTION.md`.

The user should always understand what to point at, whether to hold still, which cards are ready, what needs attention, and when to turn the page. Show the card faces throughout. The default mode waits for scene stability, captures once, and processes/reviews that fixed image without continuous card tracking. Keep controls responsive while recognition runs; other card results remain available. See [the stable-frame policy](CAPTURE_CYCLE.md) for configuration and lifecycle.

## Screen composition

- **Top:** compact scanner title, page/single mode choice, and truthful status: `Hold steady`, `Scan in 0.8 s`, `Finding cards`, or `Review page`. Expose trigger mode, steady duration, and optional repeat interval in settings. Put technical telemetry behind an explicit diagnostics control; no default frame sizes, raw scores, track IDs, or FPS labels.
- **Camera area:** live framing guides and steady/countdown cues before capture; detection starts after capture, so do not invent live card boxes. Show thin white outlines on actual detections in the fixed captured image, clearly labeled as captured. Prefer corner brackets or light quadrilateral strokes over heavy filled boxes. A restrained white scan-line traverses the captured region once when a usable capture is accepted. It represents acquisition, not model execution progress.
- **Per-card status:** small anchored badge with a physical-instance number and icon: queued, working, check, or attention. Confirmed outlines may use a restrained success accent; pair it with a check so meaning survives grayscale. Keep names, prices, and match percentages out of nine-card overlays.
- **Bottom:** one short instruction and real page counts, such as `6 ready · 2 checking · 1 needs attention`. A compact activity indicator appears only while work is pending. Show one contextual primary action: capture, review, retry selected, or next page.
- **Selection/review:** tapping a card opens a compact detail area with its crop, name, exact ID, state, and relevant retry/skip control. Page review lists each physical copy once, grouped quantities optionally. Prices are secondary and load independently.

Use a dark camera scrim only behind instructions/chrome where needed for contrast. Do not dim or blur all card faces while processing. Keep labels inside the visible preview and resolve collisions; show a selected card's details rather than stacking nine wide labels.

## Preferred language, value ranges and review

Follow [the match and value policy](MATCH_AND_VALUE_POLICY.md). Preferred card
language is separate from interface language. Use it when observed language is
unknown, labeled `English assumed` or equivalent; clear observed evidence wins.
Preserve plausible language/printing/finish alternatives, rather than silently
confirming a default-language ID.

Each instance always has a compact value/review status. Keep price ranges in
selected-card details and page review so binder faces remain visible. Show the
preferred estimate separately from the plausible-versions range, coverage,
assumptions, market/currency and update time. Missing/incomplete prices remain
unknown. A compatible materially valuable alternative triggers `Check variant`
even if it differs from the preferred language. Explain the distinguishing mark
and offer `Inspect this card` for a closer capture of that same instance.
Recognition progress does not wait for prices. Page totals distinguish confirmed
values, provisional ranges and unpriced instances.

## Per-card state contract

| State | Visible treatment | Instruction / transition |
| --- | --- | --- |
| Locating | No invented boxes; optional page framing guide | `Fit the page in view` / `Point at a card` |
| Stabilizing | Live framing guide and conditional countdown; no fabricated detections | `Hold steady` / `Scan in 0.8 s`; motion resets the dwell |
| Queued | Instance number plus waiting mark, no running spinner | `Waiting to check`; bounded scheduler owns admission |
| Recognizing | White active outline and small working indicator | `Checking card`; corresponds to an executing job |
| Verifying | Compact checking badge | `Confirming match`; retrieval/detail evidence is being evaluated, with additional capture only if needed |
| Confirmed | Check badge and quiet outline | Identity remains inspectable while other cards progress |
| Needs attention | Attention icon and restrained amber outline | Specific reason: `Glare`, `Move closer`, `Not found`, or `Match unclear`; retry/skip available |
| Retired | Remove transient geometry | Late work cannot revive the card; confirmed capture remains in its page record |

These states must be explicit fields, independent of display strings and raw confidence. Presentation consumes immutable state snapshots. A queued card does not pretend to be running, and a nearest neighbor does not become confirmed merely because inference returned.

## Page capture and wait/review cycle

1. **Frame:** `Fit the page in view`. Page mode supports nine cards immediately; single mode uses the same lifecycle with one instance. Optional expected pocket count is user configuration, not inferred card presence.
2. **Acquire:** use a configurable continuous steady dwell (initial default 0.8 seconds) and show `Scan in X s`; reset on scene movement. Accept one immutable high-resolution capture, then run detection and extract crops from it. Show one white scan-line sweep. A low-resolution stability sample must not become the recognition source by accident. Manual capture remains available; later quality checks still apply.
3. **Process:** switch the main display to the clearly labeled captured image. Show `Finding cards`, then cards progress independently through queued → recognizing → verifying. Keep controls and per-card interactions responsive. All boxes and crops belong to this snapshot; no continuous object tracking is required. No new batch is admitted while it runs.
4. **Settle:** completed cards show checks immediately. If work remains, `7 ready · 2 checking` conveys the wait cycle. When no work is pending, stop all working indicators. An unresolved card becomes needs-attention after a documented retry/time budget; no endless spinner or artificial post-result wait.
5. **Review:** derive summaries from actual instance states, such as `3 ready`, `8 ready · 1 needs attention`, or `12 cards found`. A nine-card summary is only one example. `All detected cards ready` means the detected list is resolved, not proof that no physical card was missed. Never require a count of nine or convert missing/unknown cards into success. Expected-versus-found hints remain visible. The user can retry one instance, mark a pocket empty, or explicitly skip unresolved results.
6. **Continue:** save the confirmed instances once and select `Next page`. A partial save is labeled partial; skipped/empty positions are recorded separately and are never recognized IDs. Advance page generation before accepting new work. The next page starts acquisition immediately; callbacks from the previous page cannot alter it.

Automatic page-turn detection may later suggest a new page, but it must not silently create duplicate counts. The first implementation uses an explicit next-page boundary. Reopening/reviewing or saving the same page again is idempotent. An explicit rescan updates that page rather than making another copy.

The default auto-when-steady mode stops for review after one capture. Optional repeat-while-steady mode has a configurable cooldown (initial default 3 seconds after the previous batch settles). `Next scan in X s` is conditional on stability and availability; while busy show processing, not a second capture countdown. Repeated captures replace the active page's candidate revision, never add quantities. Pause automatic replacement during detail review/editing. See `CAPTURE_CYCLE.md` for the exact eligibility and save rules.

## Multi-card identity and scheduling

- Maintain separate IDs for scan session, page instance, physical card instance/pocket, transient track, and catalog card. Two pockets with the same catalog ID are two copies. Grouping by catalog ID may change display, never silently delete a physical copy. The old duplicate-card setting must not suppress instances in a binder-page scan.
- Preserve stable display numbers within a captured page; use reading order once acquisition is stable rather than renumbering on every frame. Map crop/recognition results to the captured instances. Do not draw a stale snapshot box over a moved live page.
- With one recognition worker, retain the bounded crops of one immutable capture under a total memory cap. Nine-card support means nine independent progress states, not nine uncontrolled simultaneous model invocations. Unresolved instances receive fair opportunities; one expensive or ambiguous crop cannot starve the rest. Tracking modes may coalesce live crops later; snapshot mode does not maintain a live per-track crop queue.
- Favor recent sharp crops and known source transforms. Invalid/too-small crops produce a concrete quality cue. Partial or occluded cards are not stretched into falsely confident full-card matches.
- Preserve confirmed page results when the user requests a closer capture of a selected unresolved instance. Bind it explicitly to that instance and require review on conflicting evidence; do not automatically reassociate moving page coordinates. Whole-page recapture creates a replacement revision, not a union of old and new detections.
- Page-level progress counts physical instances, including repeated catalog IDs. `Ready` means confirmed; `checked` does not imply identified. Completion means no pending work, with any unknown/attention/skip states stated separately.

## Crop-flight timing (implemented in `web/src/ui/presenter.ts`)

The rectified crop's flight (photo → loupe → filmstrip) decorates real work and adapts to its speed:

- Results apply immediately: badges, counts and details update when a result arrives, not when a flight lands.
- The loupe shows a card only if its work is still running about 200 ms after it started. Fast cards skip it; slow ones (for example a hard verification) take it, so the loupe means "this one is taking a while".
- Cards that finish without the loupe fly straight from the photo to their slot. A flight starts within 150 ms of its trigger or not at all, with at most four in the air; the rest appear in their slot directly. No animation backlog builds up.
- A card that finishes mid-flight is redirected to its slot from where it is. Flights start from the detected quad, so the un-tilting shows the actual perspective correction.
- Reduced motion: no flights.

Measured in desktop Chrome/WebGPU (Apple-silicon Mac, test-video mock, 2026-10-16): detection about 40 ms, about 80 ms per card, nine cards settled about 1 s after capture. Phone timings are not measured yet.

## Motion, responsiveness, and accessibility

Use one short capture sweep triggered by actual acquisition, not a perpetual scanner effect. Small wait indicators exist only for real running/verifying work. Use subtle state transitions with hysteresis so detector jitter does not flash the overlay. Honor reduced motion with static outlines and text; do not rely on animation or color to convey state. Avoid false percentage progress when model work is indeterminate.

Keep essential labels readable at a 320px viewport and provide touch targets of at least 44px where possible without overlapping neighboring cards. Put full details in review when an on-card target would become too small. Screen-reader announcements summarize meaningful count/state changes, not every frame or tracking update. Error states include an action, and pause/background/stop ends wait indicators and releases camera/frame resources.

## First acceptance scenarios

- One card and nine readable known cards: independent truthful state changes, no overlapping labels, one capture sweep, correct ready counts, and an idempotent page save.
- Nine-card page containing two identical copies: nine physical instances remain represented, including both copies.
- Eight known cards plus one unknown: eight can finish; unknown remains retryable/skippable; no all-success label or infinite wait.
- Empty pocket, glare, clipped edge, page movement, and pocket seams: no fabricated detections; specific guidance and truthful partial review.
- Page turn during inference, stop/restart, background/resume, and repeated capture: no old-page result, duplicate save, frozen indicator, or stale box over a moving preview.
- Delayed recognition and a slow device: bounded queue/crop memory, fair progress across all nine, responsive preview, distinct queued/running states.
- Moving scene, expired countdown while processing, changed settings, repeat scans, and manual override: no motion-triggered false capture, overlapping batches, timer backlog, stale live boxes, or duplicate accumulation. Stillness alone must not mark blur/glare as acceptable recognition quality.
- At 320px and with reduced motion: readable instructions, selectable details/review, no essential color-only information, no clipped controls.

Benchmark time to first confirmed result, time for all eligible actual instances to confirm, time to settled partial results, all-instance-correct scene rate, per-card precision/coverage, unknown handling, and physical-copy counts. Report by actual card count/layout, including zero-card negatives and nine-card binder pages. Use independent capture sessions; adjacent frames/crops from one session must not cross evaluation splits. The timing goals in `../ROADMAP.md` are proposals, not measured performance.
