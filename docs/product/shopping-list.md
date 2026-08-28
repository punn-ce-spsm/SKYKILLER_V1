# SKYKILLER V1 — Core Build Shopping List

**Tier:** core only (runs the full demo)
**Ship to:** Thailand
**Excluded:** compute (you have a GPU machine) and the target drone (you have one)
**Budget:** ≈ USD 1,040 / ≈ THB 37,500 at ~36 THB/USD

Copy everything between the two rules below and paste it into ChatGPT.

---

## ▼ PASTE FROM HERE ▼

You are sourcing parts for a hobby counter-drone **detection** demonstrator (receive-only — no transmitters, no jamming). I am in **Thailand**. Use web search — I need current, in-stock listings, not remembered prices.

**For each line item below, give me:**

1. A specific product (exact model, not a category)
2. A direct buy link — prefer, in this order: **Lazada TH / Shopee TH (local warehouse) → AliExpress (ships to TH) → Thai electronics retailers (e.g. ThaiEasyElec, Gravitech, ArduinoAll) → international with TH shipping**
3. Price in **THB**, plus shipping cost and estimated delivery time
4. A one-line note if you are substituting something different from what I asked for, and what the substitution costs me in capability

**Then give me:** a total in THB, and a flag on any line where the part is commonly counterfeited or where cheap sellers ship a materially worse item.

**Hard constraints:**

- **Receive-only.** Do not suggest transmit-capable SDRs (HackRF, PlutoSDR, bladeRF, LimeSDR) for any line. This is a deliberate design constraint, not a budget one.
- Everything must work on **Linux**. Driver support is a hard requirement, especially for the Wi-Fi cards — check it before recommending.
- I already have the compute machine and the target drone. Do not include those.

### Line items

| # | Item | Qty | Requirement — do not substitute past this | Target USD |
|---|------|-----|-------------------------------------------|-----------|
| 1 | Wi-Fi card — spectral scan | **2** | **Atheros AR9380 / AR9382** mini-PCIe (or AR9280), plus USB or M.2 enclosures. Must support `ath9k` **spectral scan** on Linux for 2.4 **and** 5 GHz. Two cards, not one — see note B. This is the hard-to-source line, see note A. | 80 |
| 2 | Directional antennas + coax | 2 + coax | **2.4 / 5.8 GHz dual-band directional panel or patch** antennas, ~8–12 dBi, RP-SMA or SMA with pigtails to match the cards. Include ~5 m of low-loss coax (LMR-195/240 class). | 70 |
| 3 | Wi-Fi card — Remote ID | 1 | **MediaTek MT7921** or **Intel AX210** USB/M.2 adapter. Must support **monitor mode and Wi-Fi NAN** on Linux. | 35 |
| 4 | ESP32-S3 dev board | 1 | Must have **Bluetooth 5 Long Range (Coded PHY)**. ESP32-S3 or ESP32-C3. USB-C preferred. | 15 |
| 5 | Camera | 1 | **1080p60 USB (UVC)** camera with a **CS/C-mount 6–60 mm varifocal** lens. Varifocal matters more than sensor size. Must be UVC — no proprietary drivers. | 150 |
| 6 | Pan-tilt head | 1 set | 2-axis bracket + **2 × serial-bus servos** (Feetech STS3215 / Dynamixel-class) + driver board + 12 V PSU. Must hold the camera above steady with repeatable absolute positioning — hobby PWM servos are not good enough. | 130 |
| 7 | Microphone array | 1 | **4-mic USB array** — ReSpeaker 4-Mic USB, miniDSP UMA-8, or equivalent. Must expose **4 raw channels** over USB, not a single mixed-down channel. | 80 |
| 8 | Mast + tripod | 1 set | **4 m telescoping mast**, heavy tripod base, guy-line kit, ground stakes. Must stay rigid in wind — camera bearing accuracy depends on it. | 160 |
| 9 | Field power + network | 1 set | **~500 Wh portable power station** (LiFePO4 preferred), small PoE or unmanaged switch, 10 m outdoor Cat6, USB extension cables. | 320 |

### Note A — the one line that needs a decision

Item 1 is genuinely awkward to source in 2026 because AR9380 cards are old stock. Please give me **three options**, ranked:

- **A1:** an AR9380 mini-PCIe card + enclosure (best: real spectral scan on 2.4 and 5 GHz)
- **A2:** an AR9271 USB adapter (2.4 GHz only — I lose 5 GHz energy detection)
- **A3:** any current-production card you can verify does spectral scan on Linux

If none is cleanly available in Thailand, say so plainly rather than recommending something that will not work.

### Note B — why two spectral-scan cards, not one

One card gives me energy against frequency. It cannot tell me **which direction** the signal came from. Two cards, each on its own directional antenna pointed to overlap, let me compare received power between them and derive a coarse bearing from the amplitude ratio. Bearing is what aims the camera, so a single card would leave the camera with nothing to point at. Both cards must be the same model so their receive chains match.

### Also note — what I deliberately left off

**RTL-SDR dongles are not on this list.** They top out at 1.766 GHz, which covers 433 / 868 / 915 MHz and 1.2 GHz analog FPV video — real bands, used by ExpressLRS, TBS Crossfire and analog FPV builds. But my target drone is a DJI, and DJI OcuSync transmits only on 2.4 and 5.8 GHz. A sub-GHz receiver would detect **nothing** during my own demo. Worth buying later to broaden the threat model; worth nothing to me now. Do not add them back.

### Also tell me

- Anything on this list that is **restricted to import into Thailand** or needs NBTC clearance. These are all receive-only devices, so I expect few issues, but confirm rather than assume.
- Which items are worth buying locally at **Ban Mo (บ้านหม้อ)** electronics market in Bangkok instead of ordering online.

## ▲ PASTE TO HERE ▲

---

## Notes for me, not for ChatGPT

### Two corrections to the spec board, found while costing this

**1. The RTL-SDR cannot see 2.4 or 5.8 GHz.** The spec board originally had it scanning those bands. The RTL-SDR Blog V4's R828D tuner tops out at **1766 MHz** — a hardware limit of the chip.

**2. And once corrected, it turned out not to belong in this build at all.** The sub-GHz bands the RTL-SDR *does* cover — 433 / 868 / 915 MHz and 1.2 GHz — carry ExpressLRS, TBS Crossfire and analog FPV video. Real threat bands. But **my target drone is a DJI, and DJI OcuSync transmits only on 2.4 and 5.8 GHz.** A sub-GHz receiver would sit silent through my entire demonstration.

So the RF lane is not split. It is one lane, on the band the target actually uses:

| Band | Hardware | Sees |
|------|----------|------|
| 2.4 / 5 GHz | 2 × Atheros AR9380, `ath9k` spectral scan, directional antennas | DJI OcuSync and Wi-Fi-class control links — presence **and** coarse bearing |
| 2.4 GHz + BT5 | MT7921 (Wi-Fi NAN) + ESP32-S3 (BLE Coded PHY) | ASTM F3411 Remote ID — serial, drone position, operator position |

RTL-SDRs are deferred to the extensions list, worth about $140 with antennas. Buy them when the threat model needs to include FPV and long-range builds — not for this demo.

### Why the second AR9380 is not optional

A single spectral-scan card gives energy against frequency and **no direction**. Bearing is what aims the camera, and the camera is what confirms and classifies. With one card, nothing cues L2 until the target is inside acoustic range (~100 m) — which is *shorter* than the camera's own ~150 m reach, so the whole slew-to-cue mechanism in FIG. 1 quietly stops doing anything.

Two matched cards on two directional antennas restore bearing from the amplitude ratio, and the cue chain works again independently of whether Remote ID is present.

### The receive-only claim needed softening

The spec board said "there is nothing in the box that can radiate." That was **overstated** — items 1, 3 and 4 are Wi-Fi and Bluetooth radios, transmit-capable in principle.

The accurate claim: *no wideband transmitter and no RF amplifier is in the build. The transmit-capable parts are commodity, low-power, narrowband devices used strictly in passive receive modes (monitor, spectral scan, NAN listen, BLE scan). Nothing in the build can perform T1 or T2, and no effect code path touches a radio.*

Still a strong guarantee. Just a true one.

### Before you buy items 3 and 4 — $50 of Remote ID kit

**Confirm your drone actually broadcasts Remote ID.** Thailand has no US-style FAA broadcast mandate. Many DJI models broadcast DroneID / ASTM F3411 regardless of region, but it depends on model, firmware and market. See `ACTION.md` item 1 for how to check with just a phone.

If it does not broadcast, defer these two lines. The demo still works — the two AR9380s give presence and bearing on their own — but you lose identity, drone position and operator position, which is the single most impressive capability in the system.

### What is still not covered

The top of the 5.8 GHz analog FPV video band (roughly 5825–5945 MHz) sits above where Wi-Fi cards tune. Full coverage needs a HackRF-class SDR, which is transmit-capable and breaks the receive-only rule. Deferred deliberately.

### Running total

| | USD | THB (~36) |
|---|---|---|
| Original core build | 2,290 | 82,400 |
| Less compute (you have it) | −900 | −32,400 |
| Less target drone (you have it) | −350 | −12,600 |
| RF lane rework (drop RTL-SDR + sub-GHz antennas, add 2nd AR9380 + directional antennas) | 0 | 0 |
| **To buy** | **1,040** | **≈ 37,500** |

Prices are estimates. The paste block asks ChatGPT for real ones.
