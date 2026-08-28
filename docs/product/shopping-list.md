# SKYKILLER V1 — Core Build Shopping List

**Tier:** core only (runs the full demo)
**Ship to:** Thailand
**Excluded:** compute (you have a GPU machine) and the target drone (you have one)
**Budget:** ≈ USD 1,070 / ≈ THB 38,500 at ~36 THB/USD

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
| 1 | RTL-SDR receiver | 2 | **RTL-SDR Blog V4** specifically (R828D tuner, TCXO, SMA). Beware clones. Covers 500 kHz–1.766 GHz. | 80 |
| 2 | Antennas + coax | 1 set | Whips or Yagis for **433 MHz, 868/915 MHz**, plus a **1.2 GHz** patch. SMA. Include ~5 m of low-loss coax (LMR-195/240 class). | 60 |
| 3 | Wi-Fi card — spectral scan | 1 | **Atheros AR9380 / AR9382** mini-PCIe (or AR9280), plus a USB or M.2 enclosure to connect it. Must support `ath9k` **spectral scan** on Linux for 2.4 **and** 5 GHz. This is the hard-to-source line — see note A. | 40 |
| 4 | Wi-Fi card — Remote ID | 1 | **MediaTek MT7921** or **Intel AX210** USB/M.2 adapter. Must support **monitor mode and Wi-Fi NAN** on Linux. | 35 |
| 5 | ESP32-S3 dev board | 1 | Must have **Bluetooth 5 Long Range (Coded PHY)**. ESP32-S3 or ESP32-C3. USB-C preferred. | 15 |
| 6 | Camera | 1 | **1080p60 USB (UVC)** camera with a **CS/C-mount 6–60 mm varifocal** lens. Varifocal matters more than sensor size. Must be UVC — no proprietary drivers. | 150 |
| 7 | Pan-tilt head | 1 set | 2-axis bracket + **2 × serial-bus servos** (Feetech STS3215 / Dynamixel-class) + driver board + 12 V PSU. Must hold the camera above steady with repeatable absolute positioning — hobby PWM servos are not good enough. | 130 |
| 8 | Microphone array | 1 | **4-mic USB array** — ReSpeaker 4-Mic USB, miniDSP UMA-8, or equivalent. Must expose **4 raw channels** over USB, not a single mixed-down channel. | 80 |
| 9 | Mast + tripod | 1 set | **4 m telescoping mast**, heavy tripod base, guy-line kit, ground stakes. Must stay rigid in wind — camera bearing accuracy depends on it. | 160 |
| 10 | Field power + network | 1 set | **~500 Wh portable power station** (LiFePO4 preferred), small PoE or unmanaged switch, 10 m outdoor Cat6, USB extension cables. | 320 |

### Note A — the one line that needs a decision

Item 3 is genuinely awkward to source in 2026 because AR9380 cards are old stock. Please give me **three options**, ranked:

- **A1:** an AR9380 mini-PCIe card + enclosure (best: real spectral scan on 2.4 and 5 GHz)
- **A2:** an AR9271 USB adapter (2.4 GHz only — I lose 5 GHz energy detection)
- **A3:** any current-production card you can verify does spectral scan on Linux

If none is cleanly available in Thailand, say so plainly rather than recommending something that will not work.

### Also tell me

- Anything on this list that is **restricted to import into Thailand** or needs NBTC clearance. These are all receive-only devices, so I expect few issues, but confirm rather than assume.
- Which items are worth buying locally at **Ban Mo (บ้านหม้อ)** electronics market in Bangkok instead of ordering online.

## ▲ PASTE TO HERE ▲

---

## Notes for me, not for ChatGPT

### Correction from the spec board

The published spec board says the RTL-SDR scans 2.4 / 5.8 GHz. **That is wrong.** The RTL-SDR Blog V4's R828D tuner tops out at **1766 MHz** — a hardware limit of the chip. It physically cannot see the 2.4 or 5.8 GHz bands.

The RF lane is therefore split:

| Band | Hardware | What lives there |
|------|----------|------------------|
| 433 / 868 / 915 MHz, 1.2 GHz | RTL-SDR v4 ×2 | ExpressLRS, TBS Crossfire, long-range control links, 1.2 GHz analog FPV video |
| 2.4 / 5 GHz | Wi-Fi card, `ath9k` spectral scan | DJI OcuSync and Wi-Fi-class control links, energy detection only |

This is not a downgrade. Sub-GHz is where most long-range and FPV control traffic actually lives, and it is the band a serious intruder is *more* likely to use, not less.

**What you still lose:** the top of the 5.8 GHz FPV video band (5645–5945 MHz) sits above where Wi-Fi cards tune. Full coverage there needs a HackRF-class SDR, which is transmit-capable and breaks the receive-only rule. Deferred deliberately — revisit only if the demo shows it matters.

### The receive-only claim needs softening

The spec board says "there is nothing in the box that can radiate." That is **overstated**. Items 3, 4 and 5 are Wi-Fi and Bluetooth radios and are transmit-capable in principle.

The accurate claim: *no wideband transmitter and no RF amplifier is in the build. The transmit-capable parts are commodity, low-power, narrowband devices used strictly in passive receive modes (monitor, NAN listen, BLE scan). Nothing in the build can perform T1 or T2, and no effect code path touches a radio.*

Still a strong guarantee. Just a true one.

### Before you buy items 4 and 5 — $50 of Remote ID kit

**Confirm your drone actually broadcasts Remote ID.** Thailand has no US-style FAA broadcast Remote ID mandate. Many DJI models broadcast DroneID / ASTM F3411 regardless of region, but it depends on model, firmware and market.

Check first, on the drone you already own. If it does not broadcast, lane L1b has nothing to receive, the demo's strongest single capability is untestable, and you should defer these two lines until you have a target that does.

### Running total

| | USD | THB (~36) |
|---|---|---|
| Original core build | 2,290 | 82,400 |
| Less compute (you have it) | −900 | −32,400 |
| Less target drone (you have it) | −350 | −12,600 |
| RF lane correction (antennas + 2nd Wi-Fi card) | +30 | +1,100 |
| **To buy** | **1,070** | **≈ 38,500** |

Prices are estimates. The paste block asks ChatGPT for real ones.
