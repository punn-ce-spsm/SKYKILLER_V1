# SKYKILLER V1

A demo-scale counter-UAS demonstrator: multi-sensor detection feeding one fused
air picture, an operator console that owns the decision, and a graduated ladder
of effects.

**Real sensing, simulated effects.** Nothing in this build transmits. That is
deliberate rather than incidental — jamming and GNSS interference are unlawful
without specific authority nearly everywhere, and all the interesting
engineering (fusion, track continuity, classification, the ROE gate) sits
upstream of the transmitter. `EffectRequest` refuses to be constructed with
`simulated=False`, and no code path opens a radio.

---

## Five minutes, no hardware

You need **Python 3.12** — not 3.13 or 3.14, because torch has no wheels for
them. On macOS `python3` is often already newer, so name the version explicitly.

```bash
git clone https://github.com/punn-ce-spsm/SKYKILLER_V1.git
cd SKYKILLER_V1
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then, in order — each of these should work on a laptop with no camera, no
drone and no network:

```bash
.venv/bin/python -m pytest            # 221 passed, 10 skipped
.venv/bin/python -m skykiller demo    # the demonstration, as numbers
.venv/bin/python -m skykiller console # the same run as an operator console
```

**The 10 skips are expected on a fresh clone**, not a failure. They are the
identity tests, which need OpenCV's face models — those arrive the first time
you run `python -m skykiller enroll`, after which the suite reports 231 passed.
Everything else is hermetic: no camera, no weights, no network.

`console` writes `skykiller-console.html` (~540 KB). **Open it by
double-clicking** — no server, no network, works on a plane.

### Numbers your clone should reproduce

The scenarios are seeded and the recordings are deterministic, so a healthy
clone prints exactly this. If yours differs, something in your environment
changed the pipeline and that is worth understanding before you build on it.

| | Ground cameras at 3 m | Tethered observers at 200 m |
|---|---|---|
| First held | 63.2 s, at 563 m | 0.0 s, at 1504 m |
| Declared hostile | 66.2 s | 3.0 s |
| Warning before the jammer envelope | 4.4 s | **67.6 s** |
| Error at the envelope edge | 4.5 m | 1.7 m |
| Ghosts declared hostile | 0 | 0 |

Same target, same trajectory, same software, same 20 m treeline — only the
observer's altitude differs. **Altitude is the product.**

### Attack the demo, because a reviewer will

The case rests entirely on line of sight, and that is deliberately falsifiable:

```bash
.venv/bin/python -m skykiller demo --treeline-m 0    # no terrain: the advantage vanishes
.venv/bin/python -m skykiller demo --sigma-deg 2.0   # poor bearings
.venv/bin/python -m skykiller demo --baseline-m 100  # posts closer together
.venv/bin/python -m skykiller demo --mast-m 120      # observers lower
```

Being able to break your own case is worth more than the headline number.

---

## Running the sensing lane

The L2 visual lane needs a webcam. It tracks whatever you configure, so it is
verifiable indoors against a cup:

```bash
.venv/bin/python -m skykiller --config configs/hometest.yaml
```

**macOS blocks the camera on first run and says nothing.** If no window
appears: System Settings → Privacy & Security → Camera, enable your terminal,
restart the terminal. Nothing in the code can grant this.

For the drone detector instead of household objects:

```bash
.venv/bin/python -m skykiller fetch-model   # 114 MB, once
.venv/bin/python -m skykiller              # uses configs/l2.yaml
```

Two environment facts worth knowing before you debug something that is not
broken. **The device is MPS**, not CUDA, on Apple silicon — selecting on CUDA
alone silently runs everything on the CPU. And **`imgsz` does not degrade
gracefully**: at `imgsz 640` against a 1280-wide capture the reference clip
detected the target in 0 frames of 60; at 1280, 60 of 60. A distant drone *is*
the pixels that downscaling discards. Train any replacement model at the size
you intend to infer at.

`ACTION.md` has the full bench procedure, including the field-of-view
measurement that scales every bearing the system reports.

---

## Layout

| Path | What |
|---|---|
| `skykiller/l2_visual.py`, `geometry.py`, `identity.py` | the visual lane: detect, track, bearing, identity gate |
| `skykiller/triangulate.py`, `sites.py`, `associate.py` | bearings → position; pairing one post's contacts with the other's |
| `skykiller/air_picture.py` | Kalman track store, track merging, cooperative IFF |
| `skykiller/effector.py` | aim geometry, beam coverage, the two-act ROE gate |
| `skykiller/masking.py`, `scenario.py`, `harness.py` | terrain line of sight; synthetic scenarios; measurement against truth |
| `skykiller/record.py`, `console.py`, `assets/console.html` | recording a real run into one self-contained HTML console |
| `configs/` | `l2.yaml` (drone), `hometest.yaml` (indoors), `face.yaml` (identity filter) |
| `MEMORY.md` | project state, every measured number, and every correction made |
| `ACTION.md` | the things only a human can do — registration, site, bench procedure |
| `skykiller-test-map.html` | what is still unverified and how to verify it |
| `skykiller-spec-board.html` | the design the whole thing serves |

`MEMORY.md` is the one to read before changing anything. It records not just
what the code does but what was tried and rejected, and why — several of those
entries exist because a plausible-looking change was measured and turned out to
be wrong.

---

## Contributing

### Before you push

```bash
.venv/bin/python -m pytest        # 231 passed, or 221 + 10 skipped without face models
```

No failures, and no *new* skips. A skip count above 10 means a dependency
silently went missing.

**If you touched anything in the fusion pipeline, regenerate the console:**

```bash
.venv/bin/python -m skykiller console
git diff --stat skykiller-console.html
```

Recordings are byte-identical for a given seed, so that diff is empty unless
you changed real behaviour. **An unexpected diff is a finding, not noise** —
find out what moved before committing it.

### Never commit

- **Photographs of people.** `.gitignore` default-denies `*.jpg`, `*.png`,
  `*.jpeg`, `*.heic` because `me.jpg` was committed once and had to be scrubbed
  from history before this repo was ever pushed. A legitimate image can be
  added with `git add -f`, but read the file first and be certain nobody is in
  it.
- **Face embeddings** (`models/identity/*.npy`). Biometric data. It never
  leaves the machine that made it.
- **Model weights** (`*.pt`, `*.onnx`). `models/` is ignored; weights are
  fetched, not versioned.

### Working style

Branch, don't push to `main` directly:

```bash
git checkout -b your-change
# ... work, test ...
git push -u origin your-change
```

Two conventions that are load-bearing here rather than decorative:

- **Measure, don't assert.** Numbers in commit messages, comments and
  `MEMORY.md` are measurements taken on this machine. If you write one down,
  say what produced it. Several bugs in this repo's history were found because
  a claim in a docstring was checked and turned out to be false.
- **Explain the change until the explanation could be wrong.** Narrating a
  mechanism catches what tests miss, because a test written from a wrong mental
  model agrees with the bug. The most recent example: a test for the assignment
  solver passed *with the code it was testing deleted* — it asserted outcomes
  that held for a different reason. Deleting a branch and checking a test goes
  red is cheap and it works.

Update `MEMORY.md` when a change lands, and `ACTION.md` when something becomes
a human-only step.

---

## Hard rules

These are not style preferences. They define the legal posture of the project
and they hold regardless of what any individual change would be convenient.

- **Never add a transmit-capable wideband SDR** — HackRF, PlutoSDR, bladeRF,
  LimeSDR — or any RF amplifier. It invalidates the entire spec board.
- **Never point any part of this at a third party's drone.** The target is your
  own aircraft, flown by your own safety pilot, with consent.
- **Remote ID captures a real person's live location.** Retain the logs for a
  run, then delete them. Never publish a console screenshot containing a real
  operator position.
- **Nothing transmits.** If a change would open a radio for an effect, it does
  not belong in this repo.

Flying requires registration with CAAT and the NBTC in Thailand, and written
permission for the site. See `ACTION.md` — those block every field phase and
neither compresses when you need it.
