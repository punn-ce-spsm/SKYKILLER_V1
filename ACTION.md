# ACTION.md — things only you can do

Ordered by what blocks what. **Item 0 needs nothing you do not already have — do it now.** Items 2 and 3 block every field phase from P1 onward. Item 1 is free, takes ten minutes, and decides whether two lines of the shopping list are worth buying.

---

## 0. Test the tracker at home, today, with no drone

**Why:** the visual lane is built and you can verify almost all of it indoors
right now. Instead of looking for drones, it tracks a person, a bottle, a cup or
a phone — the detector is swapped by config, no code changes.

### Setup, once

```bash
cd ~/Desktop/SKYKILLER_V1
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python **3.12**, not 3.14 — torch has no 3.14 wheels and `python3` on this Mac
is 3.14.

**macOS will block the camera the first time.** When nothing appears, open
System Settings → Privacy & Security → Camera and enable your terminal app, then
restart the terminal. Nothing in the code can grant this for you.

### Run it

```bash
.venv/bin/python -m skykiller --config configs/hometest.yaml
```

A window opens. Hold up a cup, a bottle or your phone, or just stand in frame.
You should see a box, a label, `T-<id>`, and a live `az` / `el` readout. Press
`q` to quit. The first run downloads a 5 MB model.

Detections also stream to your terminal as JSON, one per line.

---

### Check 1 — does it track, and does the id hold?

Stand in frame and move slowly left to right.

**Pass:** one box follows you, and the `T-` number **stays the same** for the
whole crossing. That number is the track id, and its stability is the thing the
whole lane exists to produce.

**A new id every few frames** means the tracker is dropping you. Move more
slowly, improve the lighting, or raise `model.conf` in the config to cut false
detections that steal the association.

Expect roughly **13 fps** on your M5. The drone model is slower — about 4.7 fps —
because the only public drone weights are the heavy `x` variant.

### Check 2 — is the bearing maths right?

This is the check that matters most, because it is the number the camera cue
will eventually depend on.

Hold an object at the **far left** of frame, then the **centre**, then the **far
right**, pausing at each. Watch the `az` value.

| Position | Expected `az` |
|---|---|
| Far left edge | about **327°** (that is −32.5°, wrapped) |
| Centre | about **0°** — flickers between 359 and 1 |
| Far right edge | about **+32.5°** |

Azimuth is degrees clockwise from north and is normalised to `[0, 360)`, so it
wraps through zero at the centre. That is correct, not a glitch.

**Pass:** the numbers sweep smoothly and are roughly symmetric about the centre.
The exact end values depend on your webcam's real field of view, which check 3
measures. `el` should go positive when you raise the object and negative when
you lower it.

### Check 3 — measure your webcam's real field of view

The config ships with a **guessed** 65°. Replacing that guess with a measurement
is the single thing that makes every bearing this system ever reports accurate.

1. Put an object of known width flat-on to the camera at a **measured** distance.
   A sheet of A4 held landscape is 0.297 m. A 330 ml can is 0.066 m wide. Use a
   tape measure for the distance — 2 m is convenient.
2. Run headless and capture the output:
   ```bash
   .venv/bin/python -m skykiller --config configs/hometest.yaml \
       --no-show --jsonl ~/Desktop/fov.jsonl
   ```
   Hold the object still for a few seconds, then press Ctrl-C.
3. Read the box width — the third number in `bbox_xywh`:
   ```bash
   tail -5 ~/Desktop/fov.jsonl | python3 -c "import sys,json;[print(json.loads(l)['extra']['bbox_xywh'][2]) for l in sys.stdin]"
   ```
4. Feed that pixel width in, with your real measurements:
   ```bash
   .venv/bin/python -m skykiller calibrate \
       --width-px 1280 --object-px <the number above> \
       --object-m 0.297 --distance-m 2.0
   ```
5. Put the printed value into `camera.hfov_deg` in **both** `configs/l2.yaml` and
   `configs/hometest.yaml`.

**Record the measured FOV in `MEMORY.md`.** It is currently listed as unmeasured.

### Check 4 — does range estimation work?

Only meaningful after check 3, because range is derived from the field of view.

Set `--target-width-m` to your object's true width and hold it at a measured
distance:

```bash
.venv/bin/python -m skykiller --config configs/hometest.yaml --target-width-m 0.066
```

**Pass:** the `r ~Xm` on the box reads within about 10% of your tape measure.

If it is consistently wrong **by the same factor**, your field of view is still
off — redo check 3. If it reads `r` as absent, `--target-width-m` was not set;
that is deliberate, because a range derived from an assumed object size is a
guess, and the system reports nothing rather than a confident wrong number.

### What this proves, and what it does not

**Proves:** detection, tracking, track-id stability, the bearing maths, the range
maths, the message contract, both output sinks, the viewer, and your real frame
rate.

**Does not prove:** that the *drone* detector finds a *real drone at range*. A
close, large object never exercises small-target detection, which is the hard
part and the thing that actually fails in the field. Only item 1 plus a flight
tests that.

---

### Optional — make it track only *your* face

Instead of tracking any person, enrol a photo of yourself and the lane emits a
`Detection` only for you. Other people stay visible in the viewer, drawn thin and
grey — they are real contacts, just not the target. Hiding them would tell you
less, which is the wrong behaviour for a console.

**1. Take a photo.** Front-on, well lit, face a decent fraction of the frame. A
selfie is fine. Save it as `me.jpg` in the project folder.

**2. Enrol it.**

```bash
.venv/bin/python -m skykiller enroll --image me.jpg --name me
```

First run downloads two OpenCV models (227 KB and 37 MB).

Sideways photos are handled — a portrait phone shot whose EXIF orientation tag
was stripped stays sideways in the pixels, and the face detector is not rotation
invariant. Enrolment tries all four orientations and picks the most confident,
telling you if it had to rotate.

**If it still says no face was found**, see exactly what the detector saw:

```bash
.venv/bin/python tools/diagnose_enroll.py me.jpg
```

That reports the file type, what OpenCV loaded, whether EXIF rotation was
applied, brightness, and how many faces are found at each orientation and
confidence threshold — then tells you whether it is the photo or the code.

**3. Run with the identity filter on.**

```bash
.venv/bin/python -m skykiller --config configs/face.yaml
```

**Pass:** a green box on you labelled `T-<id> me 0.9x`, and JSON appearing in the
terminal only while you are in frame. Get someone else in shot, or hold up a
photo of another face: they get a thin grey box labelled `not-target` and **no
JSON is emitted for them**. The HUD shows `tracks 2  emitting 1`.

Measured separation on a reference pair is **0.91–0.95 for the enrolled face
against 0.07–0.08 for a different person**, so the 0.363 threshold has wide
margins either side. If your own score sits near the threshold, re-enrol with a
better photo rather than lowering `identity.threshold`.

Two different thresholds live in `configs/face.yaml` and it is worth not mixing
them up. `detect_threshold` (0.5) decides whether something **is a face**; lower
it if enrolment cannot find your face at all. `threshold` (0.363) decides whether
two faces are **the same person**; lower it only if you are being rejected as
yourself, and raise it if someone else is being accepted as you.

**Turn your head away.** The box should stay green. Identity belongs to the
*track*, not the frame — once confirmed, the tracker's motion association carries
it while no face is visible, and it re-verifies every 15 frames.

**About the enrolment file.** `models/identity/me.npy` is a face embedding, which
is biometric data. It never leaves this machine, nothing uploads it, and the
whole `models/` directory is gitignored so it cannot be committed by accident.
Delete the file to revoke it. If you demo this to anyone else, enrol *them* only
with their say-so.

**This is the same gate the Remote ID lane will use.** `identity.enabled` filters
tracks against a known identity; faces are just the implementation that can be
tested indoors. L1b swaps in serial numbers and the lane code does not change.

### Switching back to drone mode

```bash
.venv/bin/python -m skykiller fetch-model     # 109 MB, one time
.venv/bin/python -m skykiller                 # uses configs/l2.yaml
```

To track different household objects, edit `model.classes` in
`configs/hometest.yaml`. The COCO ids are listed in the comments there —
`0` person, `39` bottle, `41` cup, `67` cell phone, `32` sports ball, `73` book.
Fewer classes means fewer false positives.

---

## 1. Check whether your drone broadcasts Remote ID — free, ten minutes

**Why it matters:** lane L1b (Remote ID) is the highest-value capability in the demonstrator — it hands you the drone's serial, its position, and the operator's position from a passive beacon. It is also **$50 of hardware you should not buy** if your aircraft does not broadcast. Thailand has no FAA-style broadcast Remote ID mandate, so this is genuinely uncertain.

This no longer blocks the architecture — the two AR9380 cards give presence and bearing on their own — but without Remote ID you lose identity, drone position and operator position, which is the most impressive thing the system does.

**How to do it, no hardware needed:**

1. Install a Remote ID scanner app on an Android phone with Bluetooth 5 — **"OpenDroneID Receiver"** or **"Drone Scanner"** (both free, both on the Play Store). iPhone will not work for this; Apple restricts the Wi-Fi NAN access needed.
2. Put the phone within ~30 m of the drone, in the open.
3. Power the drone on and let it acquire GPS. Do not fly.
4. Open the app and watch for a detection for 2 minutes.

**Record the result in `MEMORY.md`:**

- **Detected** → note the serial and whether it reported an operator position. Buy items 3 and 4 on the shopping list. L1b is go.
- **Nothing detected** → check the drone's settings for a Remote ID toggle and its firmware version, then retry once. If still nothing, **do not buy items 3 and 4 yet**. Note the model and firmware in `MEMORY.md` and we will re-plan lane L1b around it.

**Either way, write down the exact model.** The whole RF lane assumes DJI OcuSync on 2.4 / 5.8 GHz. If it is an older or enterprise aircraft the bands may differ, and the antenna choice changes with them.

---

## 2. Register the drone with CAAT and NBTC

**Why it blocks:** flying an unregistered drone in Thailand is an offence, and every field phase from P1 onward requires flight.

Thailand requires drone registration with the **Civil Aviation Authority of Thailand (CAAT)** and, for aircraft carrying a camera or radio transmitter, with the **NBTC**. Both registrations are normally required regardless of weight when a camera is fitted.

**How to do it:**

1. Go to the CAAT drone registration portal — start from `caat.or.th` and follow the UAV / drone registration path. Registration is online.
2. Have ready: your ID or passport, the drone's make, model and serial number, and photographs of the aircraft.
3. Complete the NBTC registration for the radio equipment. CAAT's portal links to it, or apply via `nbtc.go.th`.
4. Expect insurance to be required — third-party liability cover is normally a condition of registration.

**Record in `MEMORY.md`:** registration numbers and expiry dates.

> Regulations change. Treat the above as a starting point and confirm the current requirements on the CAAT site rather than relying on this file.

---

## 3. Secure the demo site

**Why it blocks:** phases P1 through P5 all need a field, and the site geometry in §7 of the spec board assumes roughly 600 × 600 m.

**What you need:**

- **Private land, with written permission from the owner.** A text message saying yes is enough — keep it.
- Confirmation the site is **not inside restricted or controlled airspace**. Check before each run, not once. CAAT publishes drone no-fly zones; there are also 9 km exclusion radii around airports and restrictions near government and royal sites.
- Space for a ~300 m detection ring with clear line of sight from the mast position.
- A **safety pilot** — a second person on the sticks for the whole run, whose only job is to fly and abort. Not you; you will be at the console.

**Record in `MEMORY.md`:** site coordinates, owner contact, and the date permission was given.

---

## 4. Buy the hardware

**Blocked by:** item 1 only, and only for lines 3 and 4 of the list. Everything else can be ordered now.

1. Open `docs/product/shopping-list.md`.
2. Copy the block between `▼ PASTE FROM HERE ▼` and `▲ PASTE TO HERE ▲`.
3. Paste it into ChatGPT with web search enabled.
4. Review what it returns against the **"Requirement — do not substitute past this"** column. That column exists because several lines have a specific technical requirement that a cheaper-looking product will silently fail:
   - You need **two matched AR9380 cards**, not one. One card cannot produce a bearing, and without a bearing nothing aims the camera. See Note B in the list.
   - The **AR9380 must actually support `ath9k` spectral scan** — this is the hardest line to source and may need a compromise. See Note A in the list.
   - The **antennas must be directional**, not omni. The bearing comes from comparing power between two of them.
   - The **Wi-Fi adapter must support NAN on Linux**, not just monitor mode.
   - The **mic array must expose 4 raw channels**, not a mixed-down single channel.
   - The **camera must be UVC** — no proprietary driver.
5. Order. Expect 2–4 weeks for AliExpress, days for Lazada/Shopee local warehouses.

**Record in `MEMORY.md`:** what you actually bought, with links and prices, so the BOM in the spec board can be corrected to reality.

---

## 5. Confirm the GPU machine is adequate

**Why it matters:** the $900 mini-PC was dropped from the BOM on the basis that you already have a GPU machine. If it cannot hold the frame rate, that line comes back.

**How to check,** once you have Python and PyTorch installed:

```
pip install ultralytics
yolo predict model=yolo11n.pt source=0 show=True
```

Watch the reported inference time. **You want under ~25 ms per frame** on a 1080p webcam feed, which leaves headroom for the three other lanes and the fusion stack running alongside. Note the GPU model and the measured figure in `MEMORY.md`.

---

## Standing rules — these are not one-off actions

- **Never add a transmit-capable wideband SDR** (HackRF, PlutoSDR, bladeRF, LimeSDR) or any RF amplifier to this build. It changes the legal posture of the entire project and invalidates the spec board.
- **Never point any part of this at a third party's drone.** The target is your own aircraft, flown by your own safety pilot, with consent.
- **Remote ID captures a real person's live location.** Retain the logs for the run, then delete them. Do not publish console screenshots containing a real operator position.
