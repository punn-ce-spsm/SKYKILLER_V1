# SKYKILLER V1 — Project State

Last updated: 2026-08-29 (build 2, phases A-B)

## What this is

A demo-scale counter-UAS (anti-drone) demonstrator modelled on skylocksys.com's service. Multi-sensor detection feeding one fused air picture, an operator console that owns the decision, and a graduated ladder of effects.

**Core design constraint: real sensing, simulated effects.** Nothing in the build transmits an effect. This is deliberate — jamming and GNSS interference are unlawful without specific authority nearly everywhere, and all the interesting engineering (fusion, track continuity, classification, the ROE gate) sits upstream of the transmitter.

## Current state

**Build 1 complete: the L2 visual tracking lane runs.** Specification complete.
**Build 2 phases A and B complete: two-post triangulation, cross-site
association, the air picture and cooperative IFF.** Phases C (effector handoff)
and D (demo instrumentation) are unbuilt, as are lanes L1a, L1b, L3 and the
effect ladder.

| Artefact | Location |
|---|---|
| Visual spec board (10 sections, 5 SVG figures) | `skykiller-spec-board.html` → https://claude.ai/code/artifact/03293450-05c6-4557-b4c3-c1b16809aa47 |
| Core shopping list, Thailand, paste-ready for ChatGPT | `docs/product/shopping-list.md` |
| Human-only actions | `ACTION.md` |
| Plan file | `~/.claude/plans/i-want-to-build-curious-perlis.md` |
| L2 lane code | `skykiller/` — run `.venv/bin/python -m skykiller` |
| Build 2 fusion code | `skykiller/{triangulate,sites,associate,air_picture}.py` |
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


## Build 2 — phases A and B, as built

The chain is `Detection` (per site) → `associate` → `triangulate` → `AirPicture`
→ IFF verdict. 141 tests pass. Nothing here transmits; the receive-only
guarantee from build 1 is untouched.

| Module | Does |
|---|---|
| `triangulate.py` | N bearings → ENU position + covariance ellipse |
| `sites.py` | surveyed post positions; bus-to-solver bridge; siting maths |
| `associate.py` | pairs site A's contacts with site B's, and flags when it cannot |
| `air_picture.py` | track store, cooperative IFF, track-before-declare |

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

  | Masts | Elevation separation | Outcome |
  |---|---|---|
  | 120 m | 0.19° | ghost declared HOSTILE in 10/12 runs |
  | 200 m | 3.34° | 0/12, real hostile declared in 3.0 s |

  `SiteNetwork.elevation_separation_deg` computes it, so a site can be checked
  on paper. **This belongs in the siting brief for any demonstration.**

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
- **Nothing is flight-tested.** Every number above comes from the real solver
  against synthetic geometry.
