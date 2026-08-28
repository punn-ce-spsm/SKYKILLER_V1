# SKYKILLER V1 — Project State

Last updated: 2026-08-29

## What this is

A demo-scale counter-UAS (anti-drone) demonstrator modelled on skylocksys.com's service. Multi-sensor detection feeding one fused air picture, an operator console that owns the decision, and a graduated ladder of effects.

**Core design constraint: real sensing, simulated effects.** Nothing in the build transmits an effect. This is deliberate — jamming and GNSS interference are unlawful without specific authority nearly everywhere, and all the interesting engineering (fusion, track continuity, classification, the ROE gate) sits upstream of the transmitter.

## Current state

Specification and planning complete. **No code written yet.**

| Artefact | Location |
|---|---|
| Visual spec board (10 sections, 5 SVG figures) | `skykiller-spec-board.html` → https://claude.ai/code/artifact/03293450-05c6-4557-b4c3-c1b16809aa47 |
| Core shopping list, Thailand, paste-ready for ChatGPT | `docs/product/shopping-list.md` |
| Human-only actions | `ACTION.md` |
| Plan file | `~/.claude/plans/i-want-to-build-curious-perlis.md` |

## Architecture, in one line

Four sensor lanes → one MQTT `Detection` bus → associate / UKF track / classify / score → C2 console with a human ROE gate → three simulated effect tiers. CoT export to TAK/ATAK.

| Lane | Hardware | Status |
|---|---|---|
| L1a RF energy | RTL-SDR v4 ×2 (433/868/915 MHz, 1.2 GHz) + Atheros ath9k spectral scan (2.4/5 GHz) | to build |
| L1b Remote ID | Wi-Fi NAN adapter + ESP32-S3 (BT5 Long Range) | to build — **blocked on a Remote-ID-capable target, see ACTION.md** |
| L2 EO/IR | 1080p60 + varifocal on 2-axis pan-tilt, YOLO11n + BoT-SORT | to build |
| L3 Acoustic | 4-mic USB array, GCC-PHAT + log-mel CNN | to build |
| L4 Radar | not built — architectural slot, fed by a plot simulator | deliberate |

Effect tiers T1 (RF link denial), T2 (GNSS takeover), T3 (net interceptor) are all **simulated**. T3 is the only tier a private builder could lawfully fly, and is deferred to optional phase P6.

## Build plan

P0 bench → P1 lane 1 → P2 fusion + console → P3 cue + classify → P4 effect ladder + interop → P5 demonstration. 13 weeks, one falsifiable exit test per phase. See §9 of the spec board.

## Decisions locked

- Hybrid demo: real sensing hardware, simulated effects.
- Sensors limited to what is achievable in software; radar stays a slot.
- Sourcing in **Thailand**. Core build only.
- User already owns the compute machine and the target drone → those lines are excluded, bringing the buy to ≈ USD 1,070 / ≈ THB 38,500.

## Corrections made

**2026-08-29 — RF lane was wrong in rev A.** The spec board originally had the RTL-SDR Blog V4 scanning 2.4/5.8 GHz. Its R828D tuner stops at **1766 MHz**; that is a hardware limit and it cannot see those bands. The lane is now split: RTL-SDR covers 433/868/915 MHz and 1.2 GHz (where ExpressLRS, Crossfire and analog FPV video live), and an Atheros AR9380 in `ath9k` spectral-scan mode covers 2.4/5 GHz. The upper 5.8 GHz FPV video band (above ~5825 MHz) is **not covered** by any receive-only part in the build — full coverage would need a HackRF-class transceiver, which breaks the receive-only rule. Deferred deliberately.

**2026-08-29 — the no-transmit claim was overstated.** Rev A said "there is nothing in the box that can radiate." The Wi-Fi cards and the ESP32 are transmit-capable radios. Corrected claim: no wideband transmitter and no RF amplifier is in the build; the transmit-capable parts are narrowband, low-power, and run only in passive receive modes; nothing in the build can perform T1 or T2 and no effect code path opens a radio.

## Open questions

- Does the user's existing drone broadcast Remote ID? Thailand has no FAA-style broadcast mandate. If it does not, lane L1b is untestable and $50 of receiver should not be bought yet.
- Is the existing GPU machine adequate for YOLO11n at 60 ms/frame alongside the other lanes? Unverified.
