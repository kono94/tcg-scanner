# Phone feedback, sessions, camera lifecycle and capture intake

Date: 2026-10-17. Status: core session/camera implementation and local raw intake
complete; physical-phone checks and dataset review/training remain pending. The
investigation below describes the original behavior; changes and checks are
recorded in the implementation section.

## User requirements and observations

- The scanner worked well overall on an iPhone 13 mini, but three Pokémon cards
  were not found. Battery was below 15%. Low Power Mode, iOS/browser versions,
  thermal state, failure causes and the total number of scanned cards are unknown.
- Batch mode was useful; the app felt slow/laggy, and the camera appeared active
  after scanning and in Settings. This is user-reported physical-device feedback,
  not a controlled benchmark or measured recognition accuracy.
- Repeated `Page 1` labels across runs are confusing. Discussed alternatives:
  global numbering, game-based numbering, named sessions and `Red Binder`.
  The user has not selected a naming option yet.
- Avoid repeated camera permission prompts: the requested experience is one
  initial permission interaction, not a prompt on each capture/navigation.
- Capture uploads were enabled at `https://192.168.178.73:5173/`; the user wants
  the scans considered for training.
- Document all work in Markdown for this task. The user clarified that this is
  a special case and must not be added to AGENTS.md; the earlier addition was
  removed in full before implementation.

## Investigation performed

Read the repository guide, root README, [development priorities](DEVELOPMENT.md),
[browser README](../web/README.md), [capture cycle](CAPTURE_CYCLE.md),
[scanner UX](SCANNER_UX.md), [data collection](DATA_COLLECTION.md),
[image sources](IMAGE_SOURCES.md) and [deployment README](../deploy/README.md).
Inspected camera, navigation, stability sampling, capture, rendering, worker
initialization, batch scheduling/persistence, collection storage/presentation and
upload code. Read-only Git checks found `main` at `3f42add`, one worktree at the
repository root, and a clean working tree before these documentation edits.
No separate worktree or sub-agents were created.

External primary documentation consulted:

- [Apple Low Power Mode](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/EnergyGuide-iOS/LowPowerMode.html):
  CPU/GPU performance may be reduced; this does not establish the cause here.
- [MDN track stopping](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/stop)
  and [track disabling](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/enabled):
  disabling frame delivery and ending a track are different operations.
- [MDN camera permissions](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia):
  browser policy/user settings control persistent grants. Exactly one browser
  prompt across reloads, revoked permission and all configurations is not an app
  guarantee.
- [WebKit WebRTC permissions](https://webkit.org/blog/7763/a-closer-look-into-webrtc/):
  historical Safari context, not verification of the user's current browser.

## Implemented behavior found

- [Camera source](../web/src/camera/source.ts): `setActive(false)` disables video
  tracks; `stop()` ends them. Requested ideal resolution is 2560 × 1440; the
  browser chooses actual settings.
- [Application](../web/src/main.ts): single-page capture disables tracks and
  pauses video after the snapshot is copied. Batch processing keeps the camera
  running. Bottom navigation disables it only in the live phase; the Settings
  shortcut only changes the overlay. Camera policy is spread across handlers.
- The sampling timer checks capture phase rather than explicitly rejecting all
  covered screens before pixel reads. Freezing capture triggers does not stop
  sampling. Single-page rendering continues on animation frames while processing
  even if no presenter animation is active; actual phone costs are unmeasured.
- Hidden-document handling stops camera tracks and pauses batch scheduling.
- [Inference worker](../web/src/inference/worker.ts): iOS currently uses
  single-threaded WASM, int8 encoder and compact gallery to limit memory; it does
  not attempt WebGPU. CPU contention is plausible, not profiled here.
- [Collection storage](../web/src/collection/store.ts): session IDs use a
  three-hour activity gap. [Batch sessions](../web/src/batch/session.ts) have a
  separate active batch and page counter. These are not one explicit user-facing
  session. Finishing a batch can leave recognition of saved crops running.
- [Upload code](../web/src/feedback.ts) and [Vite handlers](../web/vite.config.ts)
  are development-only. Uploading captures does not move recognition to a
  backend; the static production server has no capture receiver.

## Local capture inventory

Read-only inspection found the uploads in `runs/captures/2026-10-17/`:

| Item | Result |
| --- | --- |
| Full JPEG captures with JSON sidecars | 16 pairs |
| Metadata camera type / automated-test flag | All `camera` / `false` |
| Single-page / batch captures | 10 / 6 |
| Detector proposals over all captures | 131 |
| JPEG dimensions | All 1440 × 1604 |
| Client timestamp range (UTC) | 16:00:37.951–16:09:24.015 |
| Corresponding local time (CEST) | 18:00:37.951–18:09:24.015 |
| Bundle | `270473320b53d5b4` |

Parsed every sidecar and used Pillow to inspect every JPEG's dimensions. Card
artwork and corners were not visually reviewed. `runs/feedback/` also contains
older correction examples, not audited by this task.

131 means detector proposals across repeated captures, not distinct physical
cards, admitted crops or verified labels. Two single-page captures carry page
number 9; numbering cannot establish unique pages. Geometry is explicitly marked
`detector_proposals_unverified`. Uploads do not supply verified exact identities
or independent true-edge annotations. Originals are unchanged local ignored
artifacts, not yet a curated dataset or verified remote backup.

## Recommended session design (proposal)

Use a persistent named scan session containing numbered captures and physical card
instances. `Red Binder` can continue tomorrow and use either workflow. Changing
day or workflow does not create a session. This is a scan group, not proof of one
unique binder.

Show `Continue Red Binder` and `New session` on the start screen when applicable,
without a modal on every visit. Provide explicit finish and later resume. Only a
new session restarts numbering; rescans revise the same capture without adding
physical copies. Migrate existing saves without changing crops or quantities.

Recommend a date/time default name plus renaming. A stable random name is another
presentation option. Recognized-game summaries (`Pokémon`, `One Piece`, `Mixed`)
can be secondary labels. Majority-game recognition should not own IDs/counters:
results may be pending, mixed or corrected later.

## Revised camera and performance plan (proposal)

The first discussion suggested closing after each single capture. The user's
permission requirement changes the starting recommendation to stream reuse for
short foreground transitions; reopening must not be assumed prompt-free.

1. Centralize camera state across overlays, processing, repeat mode and visibility.
   Disable tracks, pause video and stop sampling/frame callbacks when pixels are
   unnecessary. Keep models loaded independently.
2. Reuse the stream during foreground scanning, including short Settings/review
   visits, subject to physical Safari checks of camera indicators and resource
   behavior. Stop tracks on explicit stop/finish and when hidden. Do not claim
   measured power savings from track disabling without evidence.
3. Request camera access on an explicit scan/resume action, coalesce overlapping
   opens and discard obsolete opens after navigation. Never loop permission
   requests. Browser-controlled grant persistence remains a limit of exactly one
   prompt across restarts.
4. Measure startup to usable/focused frames, detection, crop storage, per-card
   recognition, UI stalls and sustained batches. Require fresh stability samples
   after resume; preserve batch scene-change gating.
5. Remove unnecessary sampling/rendering before changing models/resolution. Keep
   recognition bounded; consider a visible queue pause for CPU contention. No
   provider/model change is selected by this investigation.

## Capture review and data intake plan (proposal)

1. Preserve/hash originals and create a staging manifest. Keep this visit and
   repeated physical views together for splitting; uploads lack source-session IDs.
2. Automatically propose all visible instances, including detector misses. Review
   true card edges separately from sleeve/pocket edges. Detector output is not
   corner ground truth.
3. Root agent visually compares eligible complete crops with actual reference
   artwork before exact-ID assignment. Keep unresolved/partial/poor-detail cards
   unlabeled; require no user transcription or screening.
4. Diagnose geometry, retrieval and catalog gaps separately. Do not assume which
   three cards failed without evidence. Future uploads should include stable
   session/capture IDs and optional failure breadcrumbs.
5. Assign dataset roles before fitting. Keep a suitable independent phone session
   for validation and obtain other independent sessions for training; do not put
   repeated views from this visit in both training and locked tests.
6. Promote only reviewed eligible examples, version changed artifacts with DVC,
   and report counts/provenance. No training or threshold selection has run.

## Initial documentation stage

Added this log, a development-priorities link and the continuing Markdown rule
in the agent guide. No application, hosting, model, dependency or dataset edits;
no commit or deployment. One initial combined documentation patch failed context
validation against DEVELOPMENT.md and applied no changes; the corrected patch
was then applied.

Checks: Git status/worktree inspection; JSON/Pillow capture inventory; all 19
relative links in this log and DEVELOPMENT.md resolved; `git diff --check`
passed for tracked edits; the new log also passed whitespace checking with
`git diff --no-index --check /dev/null docs/PHONE_FEEDBACK_2026-10-17.md`.
Reviewed the documentation changes.
Application tests/training were not run for documentation-only changes. Phone
permission, camera indicator, startup, latency, memory and thermal checks remain
unperformed. Naming choice, implementation, automated capture review/intake and
controlled device checks remain open. Update this log or a linked dated Markdown
record as work progresses.

## Implementation work

- Restored AGENTS.md exactly to its original tracked content; the documentation
  requirement stays in this task's log only. Working on the existing main
  checkout; no commit or deployment requested or performed.
- Implemented explicit persisted scan sessions with date/time defaults, custom
  names, finish/resume, and a shared capture counter. Legacy groups migrate as
  metadata without rewriting original pages/card quantities/crop keys. Repeat
  saves retain original ownership and numbering. Deleted capture numbers are
  not reused. Day changes no longer expire sessions.
- Added Home controls for new/continue/rename/finish, Collection resume actions
  and capture terminology. Both workflows use the persistent session counter.
- Added centralized camera ownership and acquisition policy, coalesced opens,
  obsolete-result cleanup and stream reuse for short foreground navigation.
  Processing/covered screens stop frame delivery, video playback, sampling and
  frame callbacks. Explicit finish/background releases tracks and the video
  reference. No automatic permission retry loop was introduced.
- Removed unconditional per-animation-frame rendering while inference is busy;
  actual presenter animations retain their frame loop. Fixed capture-cycle
  freezing to apply in live mode as well as repeat-review mode.
- Added regression tests for session continuity/migration/numbering/idempotence,
  failed storage, camera request coalescing, permission denial, obsolete opens
  and acquisition eligibility. Typecheck passed; first unit run: 15 suites,
  89 tests passed, including parity fixtures (no skipped suites).
- Browser checks exposed a real startup/resume race: capture could be admitted
  against a stale black preview. Added a fresh, nonblank-pixel gate, with fresh
  frame-callback evidence or changed pixel evidence on the WebKit fallback. Manual
  capture and automatic triggering wait for readiness. First-preview diagnostics
  are not autofocus measurements. The failing session test then passed.
- Suspended batch delivery can miss the motion of a page turn. The gate now
  accepts a fresh steady image that differs from the capture reference after that
  delivery gap. Unchanged views remain blocked. Added regression coverage; the
  existing difference threshold remains uncalibrated.
- Protected batch finish while a capture is being stored, so a late save cannot
  recreate a batch the user just retired. Saves are coalesced and navigation to
  another session preserves the foreground revision first. Storage failures do
  not mark revisions as saved. Added the finish-during-capture regression test.
- New development uploads carry UUID capture, persistent scan-session and
  document-visit IDs, user agent and execution provider. Recognition remains on
  the phone. Existing uploads were not rewritten.
- Avoided reading/parsing all saved captures in each animation render to display
  the counter; it refreshes at collection/batch/session boundaries instead.
- Updated browser README, capture-cycle and scanner-UX documentation. The second
  combined documentation patch also failed a context check and applied nothing;
  the corrected patch succeeded. Source changes were unaffected.

## Raw phone data intake artifact

Created [the local intake manifest](../runs/capture_intake/iphone13mini-20261017/manifest.json)
using the existing Python environment and Pillow. It records SHA-256 hashes for
all 16 original JPEGs and all 16 JSON sidecars, verifies decoded dimensions against
the upload metadata, retains all 131 detector proposals and assigns the
conservative source group `iphone13mini-20261017-reported-visit`. All 16 image
hashes are distinct; this does not establish distinct physical cards or rule out
near duplicates. Device model provenance is the user's report; OS/browser versions
remain unknown for those older uploads.

All entries are explicitly `unreviewed`, identities `unlabeled`, geometry
`detector_proposals_unverified` and split `null`. Automated tests/video mocks were
excluded. No originals were moved or changed. This ignored local manifest is an
intake record, not accepted training data, a DVC upload or an independent locked
benchmark. Full geometry/artwork review, split assignment and training remain
separate pending work; no model, gallery, threshold or accepted dataset changed.

## Implementation verification and limits

- TypeScript typecheck passed.
- Unit tests: 15 suites, 93 tests passed, no skipped suites. Golden parity
  fixtures were present. Earlier intermediate runs passed 89 and 92 tests as
  regressions were added.
- Full default Chrome end-to-end suite: 4 tests passed. Covers original single
  capture/check/save and persisted batch flows, plus named sessions across
  workflow switches/reload/finish/resume and stream reuse across navigation.
- The camera test substitutes `getUserMedia` with a canvas stream. It observed
  one request through Home/Settings/continue, disabled tracks and paused video
  behind covered screens, and ended tracks/detached video on session finish.
  This is browser lifecycle evidence, not hardware permission-dialog evidence.
- At 320 CSS pixels, checked Home for horizontal overflow and visually inspected
  `web/test-results/lifecycle-camera-suspends--2dac7-rant-and-releases-on-finish/home-320.png`.
  Also inspected the session Collection screenshot at
  `web/test-results/lifecycle-named-sessions-k-9de79-ws-reload-finish-and-resume/collection.png`.
  Initial screenshot capture landed during a fade; final screenshot capture
  disables animations. Adjusted the session heading to keep its name, date,
  totals and Continue action readable.
- Production build passed. No deployment or commit has been performed.
- Test tooling emitted Node experimental/deprecation notices and ONNX provider
  placement warnings; these did not fail the passing checks. No phone timing,
  thermal, memory, autofocus, battery saving or accuracy claim follows from them.

Remaining: controlled iPhone/Safari lifecycle/performance checks; visual review
and exact-reference matching of uploaded instances; diagnosis of the reported
three misses where source evidence permits; independently grouped train/evaluation
intake and any subsequent training. The implemented naming policy is date/time
defaults with editable names, not random names or majority-game counters.

Final checkpoint: after the readiness/counter refinements, all 93 unit tests
passed again and the two new browser lifecycle/session tests passed. A final
small provenance change freezes camera/mock type and workflow before async
encoding; opening a source also synchronizes an empty capture's number with its
session. Typecheck, both lifecycle/session browser tests and the production build
passed on that final source. The full four-test browser suite passed before those
small refinements. All 24 relative documentation links resolve; all 32 original
JPEG/sidecar hashes still match the intake manifest. Tracked and new-file
whitespace checks passed, and AGENTS.md has no diff from the original commit.
The supplied LAN development URL responded HTTP 200 when checked with its local
self-signed certificate accepted by curl. Reload that development app to try the
changes. Tests and build do not constitute a VPS deployment or device benchmark.

## Follow-up: batch frozen preview, pacing, duplicate quarantine and deletion

The user reported a black batch preview after capture, repeated automatic saves
of the same page, rushed auto timing, and requested deletion of a whole collection.
This follow-up was implemented in the same main checkout, preserving the
previous changes and keeping AGENTS.md unchanged.

Investigation found that batch processing muted its camera stream without first
making a display copy. Disabled tracks can display black pixels. The scene gate
also cleared its reference and the capture cycle latched the first changed view,
so a temporary movement could admit the original page later. Fresh sensor pixels
were only a camera-readiness check, not proof of a different physical page.

Implemented changes:

- Copy a bounded (1024 px wide at most) frozen batch display before disabling
  camera delivery. Retain it through extraction and preview warm-up, then switch
  back only when fresh nonblank camera pixels are available. No model-sized
  capture is retained between batches; only this one display copy is kept.
- Keep checking the scene against the last capture right up to admission rather
  than consuming a transient change. Centre luma differences to reject uniform
  exposure changes. Add a separate configurable batch auto cooldown (default
  3 seconds), independent of steady dwell and repeat-mode cooldown. Manual Scan
  now bypasses auto timing, but not probable-duplicate admission.
- Compute versioned low-resolution colour appearances from rectified crops in
  the inference worker. Compare card appearances one-to-one, preserving repeated
  physical copies and variable card counts. At least 85% of the larger detected
  list must match, with a starting mean absolute descriptor difference of at most
  12/255. These are uncalibrated heuristics for a *possible* repeated page, not
  catalog identities, physical-card proof or measured duplicate accuracy.
- Hold a probable duplicate outside saved pages, quantities and recognition
  queues. Persist its reduced crops and pending metadata for reload-safe
  `Keep anyway`; Dismiss/new capture/finish removes unkept candidate crops.
  Explicit overrides record the suspected original capture and time. New pages
  store signatures; legacy crops are fingerprinted lazily with at most 48 old
  card decodes per capture. Missing legacy crops cannot establish duplication.
- Add Delete collection for each named group, including empty groups. Confirm
  in the app, remove that group's saved pages, metadata and IndexedDB crops, and
  retire queued work. Late recognition results check that the card still exists.
  Other groups and original training uploads remain untouched. Roll back the
  page-list removal if session metadata cannot be updated.

Checks so far: typecheck passed and 102 unit tests passed in 16 suites. Two old
scene-gate fixtures initially failed because they modeled only a uniform brightness
shift as a page turn; replaced them with patterned scenes and added an explicit
exposure-only rejection case. Added synthetic appearance invariance/one-to-one/
variable-count tests, reload/override/discard queue tests and collection-deletion/
storage-failure tests. Existing four browser flows pass; a new controlled static
replay checks frozen processing pixels, an observation window longer than the
cooldown, manual duplicate quarantine, reload/override and complete crop deletion.
Final verification: typecheck passed; all 102 unit tests passed in 16 suites,
including available parity fixtures; all five default Chrome browser tests passed
(34.5 seconds); production build passed. The first new browser test reached
quarantine successfully but timed out selecting Trigger with an exact label
locator; corrected the test to use the combobox's accessible role/name. Its
targeted rerun passed, followed by the full five-test suite. This was a test
locator failure, not a successful phone verification.

Visually inspected the frozen-processing screenshot at
`web/test-results/batch-safety-batch-holds-i-cebf2-eep-and-collection-deletion/batch-frozen.png`:
the actual captured binder image remained visible with `Saving capture 1` while
the video was paused and the track disabled. The controlled replay uses a real
test-video frame redrawn with tiny pixel changes, not a physical camera; it
observed the unchanged page for 6.5 seconds (longer than auto cooldown and dwell),
kept one saved capture, quarantined a subsequent manual capture, reloaded that
pending candidate, explicitly kept it once, then deleted the collection and
verified zero remaining crop records in IndexedDB and no resurrection on reload.

Additional safeguards: probable-repeat metadata persistence failures report a
storage error and remove the unadmitted candidate's crops; already-kept candidates
remain idempotent after interrupted metadata updates. Batch page/count summaries
can rebuild from saved `batchId` rows as well as the active record. Optional
legacy fingerprint decode failures skip that evidence unless the worker itself
failed. New truncated captures do not publish a complete-page appearance signature.
The descriptor cache is bounded to 256 entries and comparisons yield between
groups of eight old captures to keep the UI responsive.

Updated browser README, capture-cycle and scanner-UX descriptions for these
changes. Documentation link/whitespace checks and unchanged-AGENTS checks passed.
No original uploads, accepted dataset, model, gallery or deployment changed;
no commit was made. Controlled iPhone/Safari verification and calibration of
duplicate evidence remain pending. Identical visible designs cannot establish
whether a physically different page was intended, hence the explicit override.
Checked leftover Node/Vite processes from the earlier timed-out test run
(test server on port 5179). They exited between inspection and a targeted cleanup
attempt, which reported `no such process`; no process was killed by that attempt.
The user's LAN development server on port 5173 was left running and returned HTTP 200.

### Collection rename follow-up

The user also requested renaming collections. Added **Rename** beside Continue
and Delete collection for every collection, including empty and finished groups.
It opens a browser name prompt prefilled with the current name; Cancel leaves it
unchanged. Reuses the existing persisted session rename operation (trimmed,
nonblank, at most 80 characters), preserving capture numbers, saved cards, crops
and queue IDs. Updated the browser README and scanner UX description. Extended
the named-session browser flow to check Cancel, rename with whitespace trimming,
stored metadata, unchanged captures and continuing at Capture 3 under the new name.
Verification: `npm run typecheck` passed; both targeted lifecycle browser tests
passed (13.1 seconds, `E2E_PORT=5184 npm run e2e -- e2e/lifecycle.spec.ts`), including
the extended collection rename flow. `git diff --check` passed and AGENTS.md is
unchanged. No deployment, commit, dataset or model changes were made for this follow-up.

## Review follow-up (2026-10-17)

A review of the work above found the default Chrome suite failing: `e2e/batch-flow.spec.ts`
reproducibly (2/2 runs) held its second capture as `Possible repeat of Capture 1`, because
the test video still showed the first page's cards. The earlier "all five passed" result
did not hold for the final source.

Changes:

- A held possible repeat is no longer discarded implicitly by the next capture or by
  finishing the batch. `canCapture()` and `finish()` refuse while one is held; the scanner
  shows `Possible repeat · Keep it or dismiss it to continue scanning`, Scan now and Done
  are disabled, and batch review offers Keep anyway/Dismiss. Switching to or finishing
  another session asks for the decision first; continuing the same session is allowed.
  Deleting the collection that owns the candidate discards it.
- Unnamed sessions that never saved a capture are dropped when another session starts,
  resumes or the session is finished, so empty `Scan · …` collections no longer accumulate.
  Named sessions (at creation or by rename) are always kept.
- Legacy migration activates the old session only if it was used within its former
  three-hour window, instead of resuming a days-old session indefinitely.
- `batch-flow.spec.ts` accepts a held repeat for its second capture, checks that Scan now
  is disabled, and keeps it explicitly.

Fingerprint sanity check (not a calibration): on 600 random English catalog references per
game, distinct-card pairs within the 12/255 threshold were 3/179,700 (Pokémon) and
10/179,700 (One Piece); median distance about 42. Real sleeved photos were not measured.

Verification: typecheck passed; 104 unit tests in 16 suites passed; all five default
Chrome e2e tests passed. Not addressed: the extra full-collection parse in
`nextCaptureNumber()` per batch change, and phone timing of the paused-camera batch cycle.
No commit, deployment, model or dataset change.
