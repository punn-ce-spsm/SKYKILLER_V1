# SKYKILLER V1 — Project State

Last updated: 2026-08-31 (build 2 + demo console + JV association)

## What this is

A demo-scale counter-UAS (anti-drone) demonstrator modelled on skylocksys.com's service. Multi-sensor detection feeding one fused air picture, an operator console that owns the decision, and a graduated ladder of effects.

**Core design constraint: real sensing, simulated effects.** Nothing in the build transmits an effect. This is deliberate — jamming and GNSS interference are unlawful without specific authority nearly everywhere, and all the interesting engineering (fusion, track continuity, classification, the ROE gate) sits upstream of the transmitter.

## Current state

**Build 1 complete: the L2 visual tracking lane runs.** Specification complete.
**Build 2 complete (phases A–D): two-post triangulation, cross-site
association, the filtered air picture with cooperative IFF, the effector handoff
behind a human ROE gate, and the terrain-masking model with a measured
demonstration.** Cross-site association now solves as a linear assignment
problem, so the stated envelope of ten drones in view fits inside the frame
budget. Lanes L1a, L1b, L3 and the effect ladder are still unbuilt.

**The detector is still the placeholder.** Retraining it on the real image
library is the highest-value item on the board and needs the dataset first —
see the task queue at `~/.claude/plans/i-want-to-build-curious-perlis.md`.

Run the demonstration:
- `.venv/bin/python -m skykiller demo` — the numbers, as text
- `.venv/bin/python -m skykiller console` — the same run as an operator console,
  one self-contained HTML file (~530 KB) that opens by double-click, needs no
  server and no network, and can be handed to the customer to keep

| Artefact | Location |
|---|---|
| Visual spec board (10 sections, 5 SVG figures) | `skykiller-spec-board.html` → https://claude.ai/code/artifact/03293450-05c6-4557-b4c3-c1b16809aa47 |
| Core shopping list, Thailand, paste-ready for ChatGPT | `docs/product/shopping-list.md` |
| Human-only actions | `ACTION.md` |
| Plan file | `~/.claude/plans/i-want-to-build-curious-perlis.md` |
| L2 lane code | `skykiller/` — run `.venv/bin/python -m skykiller` |
| Build 2 fusion code | `skykiller/{triangulate,sites,associate,air_picture,effector,scenario}.py` |
| L2 engineering notes | `docs/engineering/L2-visual-tracking.md` |

## Architecture, in one line

Four sensor lanes → one MQTT `Detection` bus → associate / UKF track / classify / score → C2 console with a human ROE gate → three simulated effect tiers. CoT export to TAK/ATAK.

| Lane | Hardware | Status |
|---|---|---|
| L1a RF energy | 2 × Atheros AR9380, ath9k spectral scan, directional 2.4/5.8 GHz panels — presence **and** bearing | to build |
| L1b Remote ID | Wi-Fi NAN adapter + ESP32-S3 (BT5 Long Range) | to build — **verify the target broadcasts first, see ACTION.md** |
| L2 EO/IR | webcam now; 1080p60 + varifocal on pan-tilt later. YOLO11x drone weights + BoT-SORT | **built (build 1)** |
| L3 Acoustic | 4-mic USB array, GCC-PHAT + log-mel CNN | to build |
| L4 Radar | not built — architectural slot, fed by a plot simulator | deliberate |

Effect tiers T1 (RF link denial), T2 (GNSS takeover), T3 (net interceptor) are all **simulated**. T3 is the only tier a private builder could lawfully fly, and is deferred to optional phase P6.

## Build plan

P0 bench → P1 lane 1 → P2 fusion + console → P3 cue + classify → P4 effect ladder + interop → P5 demonstration. 13 weeks, one falsifiable exit test per phase. See §9 of the spec board.

## Decisions locked

- Hybrid demo: real sensing hardware, simulated effects.
- Sensors limited to what is achievable in software; radar stays a slot.
- Sourcing in **Thailand**. Core build only.
- User already owns the compute machine and the target drone → those lines are excluded, bringing the buy to ≈ USD 1,040 / ≈ THB 37,500.

## Corrections made

**2026-08-29 — RF lane was wrong in rev A.** The spec board originally had the RTL-SDR Blog V4 scanning 2.4/5.8 GHz. Its R828D tuner stops at **1766 MHz**; that is a hardware limit and it cannot see those bands.

Still not covered by anything in the build: the upper 5.8 GHz analogue FPV video band, above ~5825 MHz, which sits above where Wi-Fi cards tune. Full coverage would need a HackRF-class transceiver, which breaks the receive-only rule. Deferred deliberately.

**2026-08-29 — and once corrected, the RTL-SDR did not belong in the build at all.** Tracing the P1 exit test against the user's actual target exposed the real problem: the target is a DJI, DJI OcuSync transmits only on 2.4 and 5.8 GHz, so a sub-GHz receiver would sit silent through the entire demonstration. The RF lane is no longer split — it is 2 × AR9380 on 2.4/5 GHz, and the RTL-SDRs moved to extensions ($140), to be bought when the threat model widens to ExpressLRS / Crossfire / analogue FPV.

**2026-08-29 — a single spectral-scan card would have broken the cue chain.** One card gives energy against frequency and no bearing. With no bearing from L1, nothing cues the camera until the target is inside acoustic range (~100 m) — shorter than the camera's own ~150 m reach, so the slew-to-cue loop in FIG. 1 would have silently done nothing. Two matched cards on directional antennas restore bearing from the amplitude ratio. This is why item 1 on the shopping list is quantity two.

**2026-08-29 — the no-transmit claim was overstated.** Rev A said "there is nothing in the box that can radiate." The Wi-Fi cards and the ESP32 are transmit-capable radios. Corrected claim: no wideband transmitter and no RF amplifier is in the build; the transmit-capable parts are narrowband, low-power, and run only in passive receive modes; nothing in the build can perform T1 or T2 and no effect code path opens a radio.

## Build 1 — L2 visual lane, as built

Target is a **DJI Mini 4 Pro**. Runs on the laptop webcam or a video file, emits
the section-2 `Detection` contract as JSONL, viewer optional.

- **Environment: Python 3.12**, not the system 3.14 — torch has no 3.14 wheels.
  Venv at `.venv/`. `requirements.txt` is authoritative.
- **Device is MPS**, not CUDA. This machine is an Apple M5. `resolve_device`
  prefers MPS, then CUDA, then CPU; selecting on CUDA alone would have run the
  whole pipeline on the CPU.
- **Measured: 4.7 fps** (213 ms/frame) with YOLO11x at imgsz 1280 on MPS.
  Adequate for a hovering or walking-pace target, marginal above ~10 m/s.
  Biggest available win is fine-tuning a `yolo11n`/`s` instead of the `x`;
  CoreML export is the untried second option. This closes ACTION.md item 5.
- **Detector is swappable by config**, which was the whole point: drop a `.pt`
  in `models/`, change `model.weights` in `configs/l2.yaml`. Currently
  `doguilmak/Drone-Detection-YOLOv11x` (MIT, single class, ~1,000 training
  images) — a placeholder that proves the pipeline, not a fielded detector.
  Falls back to COCO YOLO11n filtered to airplane/bird/kite, marked
  `provisional: true`, so a fresh clone still demonstrates.
- **Home test mode added.** `configs/hometest.yaml` tracks household objects
  (person/bottle/cup/phone) via stock COCO YOLO11n, so the lane is verifiable
  indoors with no drone. Runs ~13 fps vs 4.7 for the drone model. Procedure and
  acceptance checks are in ACTION.md item 0. Enabled by two new config options,
  `model.classes` (a filter that applies to any weights, not just the fallback)
  and `model.weights: null` (choose COCO deliberately, distinct from a missing
  file, which still warns).
- **Verified end to end** on a synthetic clip built from a real quadcopter photo:
  60/60 frames detected, one stable track id, azimuth monotonic through the
  boresight once unwrapped. 36 unit tests pass.

### The bug worth remembering: `imgsz`

`imgsz=640` against a 1280-wide capture detected the target in **0 of 60 frames**
at any confidence down to 0.01. At `imgsz=1280`, 60 of 60. Detection does not
degrade gracefully with input size — it falls off a cliff, because a distant
drone *is* the pixels that downscaling discards.

| imgsz | ms/frame | fps | detections |
|---:|---:|---:|---:|
| 640 | 56 | 17.8 | 0/60 |
| 800 | 91 | 11.0 | 5/60 |
| 960 | 116 | 8.6 | 29/60 |
| 1280 | 213 | 4.7 | 60/60 |

`model.imgsz` therefore defaults to `auto` (match the capture width) and the lane
warns loudly if configured below it. Train any replacement model at the imgsz you
intend to infer at.

## Open questions

- Does the user's existing drone broadcast Remote ID? Thailand has no FAA-style broadcast mandate. If it does not, lane L1b is untestable and $50 of receiver should not be bought yet. No longer architecturally blocking — the two AR9380s give presence and bearing without it — but identity, drone position and operator position are all lost.
- Which DJI model is it? The whole RF lane is specced on the assumption of OcuSync at 2.4/5.8 GHz. An older or enterprise model could differ (some DJI enterprise links use 1.4 GHz).
- **Not yet flown.** The build-1 exit criterion — a Mini 4 Pro holding one track
  id for 30 s at 15–20 m — needs a real flight, which needs ACTION.md items 2
  and 3 (CAAT/NBTC registration, site permission) done first.
- Will 4.7 fps hold a track on a real drone crossing at speed? Untested against
  a real aircraft; the synthetic clip moved slowly.
- Bearing accuracy is unmeasured. The 65 deg FOV is a guess until someone runs
  ACTION.md item 0 check 3 against the actual webcam. That is now a ten-minute
  indoor task with no drone required, and it gates the accuracy of every bearing
  the system will ever report.


## Build 2 — phases A, B and C, as built

The chain is `Detection` (per site) → `associate` → `triangulate` → `AirPicture`
→ IFF verdict → `RoeGate` → `EffectRequest`. 220 tests pass. Nothing here
transmits; the receive-only guarantee from build 1 is untouched, and
`EffectRequest` refuses to be constructed with `simulated=False`.

| Module | Does |
|---|---|
| `triangulate.py` | N bearings → ENU position + covariance ellipse |
| `sites.py` | surveyed post positions; bus-to-solver bridge; siting maths |
| `associate.py` | pairs site A's contacts with site B's (Jonker-Volgenant), and flags when it cannot |
| `air_picture.py` | Kalman track store, track merging, cooperative IFF |
| `effector.py` | aim geometry from the effector, beam coverage, the ROE gate |
| `scenario.py` | synthetic scenarios that drive the real pipeline |
| `masking.py` | terrain line-of-sight — the reason the observer is airborne |
| `harness.py` | measures a run against truth: warning, dwell, accuracy, failures |
| `record.py` | drives the real pipeline and writes down what happened, frame by frame |
| `console.py` | injects a recorded run into `assets/console.html` |
| `assets/console.html` | the operator console: three views, playback, the ROE gate |

### Deployment numbers that came out of the maths

These are the actionable results. All measured against the real solver at
σ = 0.5°, not asserted.

- **Baseline sets usable range, not the camera.** 100 m fixes every frame at
  1 km, 91% at 2 km, 73% at 2.5 km. 200 m clears the whole early-warning band.
  `SiteNetwork.max_range_m`.
- **Accuracy at jammer range is the requirement, and it is met.** 22 m at 500 m
  with a 100 m baseline, against ±44 m affordable for the narrowest (10°) beam.
  The 2 km figure is ~350 m and that is fine — at 2 km the job is early warning,
  not aiming.
- **The masts must fly above the traffic they watch.** This is the one that
  would have been discovered in the field. Two targets at different ranges are
  pushed apart in elevation as the mast rises, and when they subtend the *same*
  elevation the cross-site pairing is exactly degenerate — true and crossed
  pairings intersect equally well and the winner is floating-point noise. Twelve
  seeds of the two-aircraft scenario, same software and baseline throughout:

  Since the track filter and merge landed, poor siting no longer *invents*
  targets — it degrades, which is the right shape. What it still costs, over
  eight seeds of the two-aircraft ingress:

  | Siting | Median error on the inbound track | Hostile declared |
  |---|---|---|
  | 100 m baseline, 120 m masts | 30.0 m | 15.4 s |
  | 200 m baseline, 120 m masts | 14.7 m | 8.7 s |
  | 200 m baseline, 200 m masts | 14.3 m | 3.0 s |

  Twelve seconds of delay is 185 m of standoff at 15 m/s — most of a jammer
  envelope. `SiteNetwork.elevation_separation_deg` computes the mast-height
  number, so a site can be checked on paper. **This belongs in the siting brief
  for any demonstration.**

### IFF, and the two rules that make it safe

Identity is decided by correlation against a feed only our side produces. The
rule stated at the outset — "if the frequency is not ours, it is theirs" — does
not hold, because both sides fly DJI-class aircraft on the same bands. Friends
cooperate; hostiles do not.

1. **HOSTILE needs a healthy feed.** A dead feed makes every track stop
   correlating, which read naively declares the whole sky hostile at the moment
   we have lost the ability to tell. Dead feed ⇒ UNKNOWN. This is why the feed
   must heartbeat an explicit "nothing airborne": silence and an empty report
   are opposite messages.
2. **No FRIENDLY → HOSTILE transition.** It goes via UNKNOWN and serves a hold,
   so one dropped packet cannot re-label our own aircraft as a target.

Plus track-before-declare: HOSTILE needs 3 confident frames, so a ghost that
appears once cannot be shot at.

### The demonstration, measured

`python -m skykiller demo`. Same target, same trajectory, same software, same
20 m treeline 200 m out — only the observer's altitude differs. A 3 m ground
camera behind that treeline cannot see below 88 m at 1 km; the target flies
at 50 m.

| | Ground cameras at 3 m | Tethered observers at 200 m |
|---|---|---|
| First held | 63.2 s, at 563 m | **0.0 s, at 1504 m** |
| Declared hostile | 66.2 s, at 519 m | **3.0 s, at 1459 m** |
| Warning before the jammer envelope | 4.4 s | **67.6 s** |
| Error at the envelope edge | 4.5 m | **1.7 m** |
| Error over the whole run (p50) | 3.3 m | 11.9 m |
| Ghosts declared hostile | 0 | 0 |
| Own aircraft called hostile | never | never |

**Fifteen times the warning.** Two readings of this table are wrong and the
harness guards against both:

- The whole-run p50 makes the airborne pair look *worse*. It is not — it holds
  the target for 95 s of mostly long range, where a bearings-only fix is
  honestly poor, while the ground camera only ever sees it inside 600 m. At the
  envelope edge, the moment an effector cares about, the airborne pair is better.
- Envelope timings are taken from **truth**, never from our own estimate, so a
  bad fix cannot flatter the warning figure. Tested by wrecking the sensing to
  5° and checking the entry time does not move.

Remove the treeline and the ground pair performs like the airborne one — the
product case rests on terrain, and that is deliberately falsifiable.

### The effector handoff

Two things in it are load-bearing:

- **The bearing is computed from the effector, not the observer.** They are
  hundreds of metres apart and the angle between them at a target is tens of
  degrees when the target is close — exactly when someone wants to shoot.
  Re-projecting is the only reason fusion needs a *position* rather than a
  bearing; a single-post system could never do it.
- **The error ellipse must fit inside the beam**, projected across the boresight
  only. A bearings-only fix is elongated *down-range*, and down-range error
  moves the target within the same cone rather than out of it. Counting it would
  reject good solutions for an error that cannot cause a miss.

The ROE gate takes two deliberate human acts and continuously revokes them: if
the track stops being HOSTILE, leaves the envelope, is lost, or the operator
takes too long, the decision is withdrawn without anyone remembering to withdraw
it. It refuses on identity but reports on range — shooting at our own aircraft
is the failure it exists to prevent; the effector's finite range is a fact the
record states plainly.

### Corrections made in build 2

- **The miss gate was the wrong shape.** A fixed 150 m threshold, where the miss
  from honest bearing noise scales with range (p99 1.9 m at 120 m, 32 m at
  2 km) — so 78× too loose up close. Now dimensionless.
- **No gate can catch azimuth mis-association with two rays.** A 20° azimuth
  error puts the fix 75 m out with a 4 m miss and a confident σ; the same error
  in elevation misses by 22 m and is caught. Two rays that cross produce a zero
  miss wherever they cross. A third post fixes it.
- **Nothing paired site A's contacts with site B's.** Found by ducking the chain
  end to end. Two aircraft fused into one position that was neither, and a
  crossed pairing reported a *tighter* covariance than the real target.
- **Association by last position instead of predicted position** split one
  aircraft into a new track every frame — and did it more readily the better
  the fix was.
- **Then a raw two-point velocity made long range worse.** It divides position
  noise by dt, so a 100 m fix at 5 Hz yields a 700 m/s phantom velocity that the
  code trusted. Velocity now carries its own σ and is only used when it beats
  not predicting at all.
- **The track position was the raw last fix.** So track-to-fix distance was the
  difference of two noisy measurements, and a 99% association gate applied 800
  times spawned duplicates by construction. Replaced with the constant-velocity
  Kalman filter the spec board always called for: median error on the inbound
  track fell from ~100 m (single fix) to 15 m, and it cured the ghost-HOSTILE
  problem that siting alone had been covering.
- **Even filtered, each aircraft grew two tracks** 10–18 m apart, each nearest
  to the fix on alternate frames so neither aged out. A duplicate of the hostile
  raised a second ARM prompt; a duplicate of the friendly was declared HOSTILE.
  Fixed by merging statistically indistinguishable tracks.
- **The friendly correlation gate ignored report age.** A feed that merely went
  quiet — still inside its freshness window — left our own aircraft 105 m from
  its last report at 21 m/s, outside the gate, through the hold, and declared
  HOSTILE at 23 s. Reports are now inflated by how far the aircraft could have
  flown since.
- **Covariance is conservative by ~1/cos(el)** — isotropic azimuth term, under
  2% below 10° elevation where this operates. Left deliberately; conservative is
  the safe direction for a number that gates an effector.

### Known and deliberately unsolved

- **A hostile inside a friendly's correlation gate reads FRIENDLY.** Inherent to
  position-correlation IFF. Closing it needs Remote ID serials from L1b, not a
  smaller threshold. Tested so nobody finds it in the field.
- **Cross-site association does not use track feedback.** Existing track
  predictions are strong evidence about which pairing is right, and feeding them
  back into `associate` is the proper software answer to residual ghosts. Not
  built — it is a real design change, not a tweak.
- **`associate` handles exactly two posts.** A third raises `NotImplementedError`
  rather than silently mishandling it.
- **A swarm past 30 contacts a post is refused, not slowed.** The cap is set by
  the cost matrix, not the solver. Beyond it the answer is bearing-space
  pre-gating so most pairs are never triangulated -- a different algorithm.
- **The masking model is a screen, not a DEM.** An obstacle is a crest line
  with an altitude. It ignores refraction, earth curvature (centimetres under
  3 km), partial vegetation, and anything below the crest — all of which make
  the real picture *better* than predicted, which is the right direction for a
  claim about what the customer cannot see. A real siting study needs terrain
  data; this needs a map and a tape measure.
- **The scenario models no range or detector limit.** Every aircraft in clear
  line of sight is seen. Sensing is therefore optimistic; the association load
  is pessimistic. Build 1 measured the real optical limits (webcam 26 m, 25 mm
  lens 128 m, 100 mm 601 m) and those are not wired into the scenario yet.
- **Nothing is flight-tested.** Every number above comes from the real solver
  against synthetic geometry.


## Build 3 — the assignment solver (2026-08-31)

`associate` enumerated every partial pairing of post A's contacts with post B's.
Factorial, and it broke **inside** the stated envelope of ten drones in view,
not at some far-off swarm limit: ten contacts a post is 234,662,231 matchings,
and eight — the old cap — took **1.27 s** against a 200 ms frame budget at 5 Hz.

Now one `scipy.optimize.linear_sum_assignment` (a modified Jonker-Volgenant)
over a square-padded cost matrix. `scipy` is a new dependency.

| contacts/post | total | cost matrix | solve + second-best |
|---:|---:|---:|---:|
| 2 | 0.4 ms | 0.2 ms | 0.01 ms |
| 10 | 5.0 ms | 4.6 ms | 0.06 ms |
| 20 | 19.4 ms | 18.7 ms | 0.36 ms |
| 30 | 44.1 ms | 43.9 ms | 0.65 ms |

**The solver is now under 2% of the call at every size.** What costs is building
the cost matrix — one `triangulate` per candidate pair at a measured 56 µs,
quadratic. So `MAX_CONTACTS_PER_SITE` rose 8 → 30 on the geometry, not the
assignment, and the next speed-up if it is ever needed is bearing-space
pre-gating, not a faster solver.

### The three things that could have broken quietly

- **The padding has to reproduce a *partial* matching.** Forcing every contact
  to pair is exactly what makes a tracker invent targets when one post is
  occluded. Each real row gets a dummy column to retire into and each real
  column a dummy row, both priced at `unpaired_price`, dummy-dummy free — which
  reproduces `sum(pairs) + (n+m−2k)·price` term for term. The old `_matchings`
  now lives in `tests/test_associate.py` as the oracle: 8 shapes × 40 trials
  including infeasible cells, same total, same pairing wherever the optimum is
  unique.
- **`margin_m` needs the second-best assignment, and JV returns only the
  optimum.** Dropped in naively, `ambiguous` becomes permanently False — which
  silently removes the gate that stops a ghost reaching HOSTILE, the exact
  defect phase B was written to fix. Recovered from Murty's first step (forbid
  each winning pair, re-solve) plus the cheapest single addition, because a
  rival either drops one of the winner's pairs or keeps them all and adds one.
- **`linear_sum_assignment` raises on infinities it cannot assign around.**
  Infeasible pairs take a finite sentinel bigger than the entire all-unpaired
  assignment, so one can never be chosen.

Checked beyond the unit tests: **2000 frames across five geometries**, old
scoring path against new — zero pairing differences, zero ambiguity-verdict
differences, margins agreeing to 5e-14. `demo` prints the same 4.4 s / 67.6 s
and 4.5 m / 1.7 m, and the regenerated console is byte-identical.

### The defect the duck found, and the lesson

A test for the unmeasurable-price path passed **with the code it was testing
deliberately removed**. `_gate_scale` prices an unpaired contact off the first
contact at each post, and returns infinity when those two rays are too close
to parallel to triangulate — reachable past ~2.9 km on a 100 m baseline. The
test asserted outcomes that happened to hold for a different reason.

What the substitution actually prevents is a **`ValueError: cost matrix is
infeasible` crash mid-frame** when the two posts see different numbers of
contacts — because an infinite price makes retiring anyone infinitely
expensive, so only a perfect pairing is finite, and with unequal counts none
exists. What makes the fallback safe is that `ambiguity_floor` comes from the
same unmeasurable scale, so it is infinite too: every fix in such a frame reads
ambiguous and cannot create a track. The solver does produce a ghost there
(measured, at (96, 212, 78)); it just cannot be believed.

**Mutation-testing each branch is what caught it.** Three other mutations —
deleting the Murty loop, deleting the addition term, shrinking the sentinel —
all died against the oracle. That one lived, and it was the one covering a
crash.

## The demo console

`python -m skykiller console` records the pipeline twice — 3 m masts and 200 m
masts, identical in every other respect — and writes one HTML file. Three views
on one clock: **Console** (the FIG. 4 operator surface), **Comparison** (the two
runs side by side), **Accuracy** (reported position against truth).

It exists because of a specific gap: `RoeGate.arm()` and `RoeGate.authorise()`
had **no callers outside `tests/`**. The two-human-act gate — the thing that
makes this defensible — was fully built, fully tested, and had never been
operated by a person. Everything build 2 produced reached the operator as text
from a `print` statement.

### How the ROE gate survives the trip into a browser

The gate is tested Python and is **not** reimplemented in JavaScript. At record
time, for every frame where a track is `PROMPTED`, a `copy.deepcopy` of the live
gate is armed and authorised and the genuine `EffectRequest` is stored against
that frame. The browser holds no ROE logic: it offers ARM only where the
recording says the real gate said `PROMPTED`, and displays what Python decided.

A fresh `RoeGate` would *not* work here and the reason is worth keeping. `update()`
prompts inside `envelope_m` (500 m) but `_revoked()` only releases past
`envelope_m × EXIT_HYSTERESIS` (550 m). A track in that 50 m band is legitimately
prompted in the live gate and a fresh gate refuses to prompt it — so AUTHORISE
would have had nothing behind it. The deepcopy carries the real engagement over.

**One field is substituted in the browser.** The recorder arms and authorises on
the same frame, so the stored `armed_at` equals `authorised_at`, which reads as a
one-click engagement — exactly what the two-act gate exists to prevent. Nothing
in the aiming solution depends on when the arm happened, so the browser replaces
that single field with the operator's own ARM time and says so on screen.

### Corrections found while building it

- **`json.dumps` emits bare `Infinity`.** Invalid JSON, but *valid JavaScript*,
  so it would have arrived in the page as a `NaN` SVG coordinate and made
  geometry vanish with nothing logged. Reachable: `lowest_visible_alt` returns
  `-inf` with no obstacles, which is what `--treeline-m 0` asks for. Non-finite
  values now become `null`, `allow_nan=False` asserts it, and the page guards
  coordinates before they reach the DOM.
- **`Track` objects are mutated in place** and `Frame.tracks` is a new list of
  the *same* objects — holding a reference and reading it later gives the final
  position for every frame. The recorder serialises inside the loop; a test pins
  it, because the failure mode is a frozen map with no error.
- **The error chart was plotted by sample index.** The ground run holds the
  target for only the last third, so its 167 samples were stretched across the
  full width — making a blind sensor look accurate. Now plotted against time.
- **The comparison read "1 track held" for the ground pair**, which is true and
  misleading: it holds the *friendly*, which orbits close and above the treeline,
  while being blind to the inbound target. It now says whether the **target** is
  held.
- **Recording and measuring drove the pipeline twice** (76 s). `harness.measure`
  gained an `on_frame` hook so one pass does both — 4.5 s, and the console and
  the text demo now cannot quote different numbers.

### Deliberately absent

No threat score. Zone scoring is specified and not built, and FIG. 4's
illustrative `82` would be a number the pipeline never produced. The same rule
removed FIG. 4's `BUS 41 msg/s` and `AUDIT 1 204 events`: **nothing on the screen
is a figure the code did not compute.** Lanes L1a/L1b/L3/L4 are shown greyed and
labelled *not fitted*.
