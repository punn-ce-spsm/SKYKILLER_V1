# ACTION.md — things only you can do

Ordered by what blocks what. Nothing in the build proceeds past P1 until items 1–3 are done.

---

## 1. Check whether your drone broadcasts Remote ID — do this first, it is free

**Why it blocks:** lane L1b (Remote ID) is the highest-value capability in the whole demonstrator — it hands you the drone's serial, its position, and the operator's position from a passive beacon. It is also worth **$50 of hardware you should not buy** if your aircraft does not broadcast. Thailand has no FAA-style broadcast Remote ID mandate, so this is genuinely uncertain.

**How to do it, no hardware needed:**

1. Install a Remote ID scanner app on an Android phone with Bluetooth 5 — **"OpenDroneID Receiver"** or **"Drone Scanner"** (both free, both on the Play Store). iPhone will not work for this; Apple restricts the Wi-Fi NAN access needed.
2. Put the phone within ~30 m of the drone, in the open.
3. Power the drone on and let it acquire GPS. Do not fly.
4. Open the app and watch for a detection for 2 minutes.

**Record the result in `MEMORY.md`:**

- **Detected** → note the serial and whether it reported an operator position. Buy items 4 and 5 on the shopping list. L1b is go.
- **Nothing detected** → check the drone's settings for a Remote ID toggle and its firmware version, then retry once. If still nothing, **do not buy items 4 and 5 yet**. Note the model and firmware in `MEMORY.md` and we will re-plan lane L1b around it.

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

**Blocked by:** item 1 only, and only for lines 4 and 5 of the list. Everything else can be ordered now.

1. Open `docs/product/shopping-list.md`.
2. Copy the block between `▼ PASTE FROM HERE ▼` and `▲ PASTE TO HERE ▲`.
3. Paste it into ChatGPT with web search enabled.
4. Review what it returns against the **"Requirement — do not substitute past this"** column. That column exists because several lines have a specific technical requirement that a cheaper-looking product will silently fail:
   - The **RTL-SDR must be a genuine Blog V4**, not a clone.
   - The **AR9380 card must actually support `ath9k` spectral scan** — this is the hardest line to source and may need a compromise. See Note A in the list.
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
