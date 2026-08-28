# L2 — Visual tracking lane

Build 1. Camera in, `Detection` messages out. This is the only lane that exists
so far; it runs standalone and knows nothing about fusion or the console.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m skykiller fetch-model          # 109 MB, one time
.venv/bin/python -m skykiller                      # webcam, viewer on
```

| Command | What it does |
|---|---|
| `python -m skykiller` | Webcam 0, viewer, detections to stdout |
| `python -m skykiller --source clip.mp4` | Run against a recording |
| `python -m skykiller --no-show --jsonl out.jsonl` | Headless, log to a file |
| `python -m skykiller --target-width-m 0.35` | Enable range estimates for a Mini 4 Pro |
| `python -m skykiller calibrate ...` | Measure the true horizontal FOV |
| `python -m skykiller fetch-model` | Download the drone weights |

Press `q` or `Esc` to quit the viewer.

## The one number that matters: `imgsz`

**Measured on a 60-frame clip with a 150 px target in a 1280×720 frame:**

| `imgsz` | ms/frame (M5, MPS) | fps | frames with a detection |
|---:|---:|---:|---:|
| 640 | 56 | 17.8 | **0 / 60** |
| 800 | 91 | 11.0 | 5 / 60 |
| 960 | 116 | 8.6 | 29 / 60 |
| **1280** | 213 | **4.7** | **60 / 60** |

Detection does not degrade gracefully as `imgsz` falls — it falls off a cliff. A
distant drone is only a few dozen pixels; downscaling the frame before detection
throws away the pixels the target is made of. At `imgsz=640` the model found
nothing at **any** confidence threshold, down to 0.01.

So `model.imgsz` defaults to `auto`, meaning *match the capture width*. The lane
prints a warning if it is ever configured below the frame width, because the
symptom of getting this wrong is not an error — it is a system that quietly sees
half as far.

`quantize`/`half` precision made no measurable difference on MPS.

## Known limits

**Frame rate is 4.7 fps** with YOLO11x at 1280 on an M5. Fine for a hovering or
walking-pace target, marginal for a fast crossing one — at 4.7 fps a drone doing
15 m/s moves ~3 m between observations. Two ways out, in order of payoff:

1. **Fine-tune a `yolo11n` or `yolo11s` on drone imagery.** The public weights
   are the `x` variant, which is 10–20× heavier than needed for a single class.
   This is the largest available win and it also improves accuracy.
2. **Export to CoreML** (`model.export(format="coreml")`) to reach the Apple
   Neural Engine. Not yet tried.

**Bearing is approximate.** Pinhole model, no lens distortion term. With the
default guessed 65° FOV expect ±2–3° absolute. Relative motion within a track is
much better than that, which is what a camera cue actually needs. Run
`calibrate` to replace the guess with a measurement.

**Range is an assumption, not a measurement.** `range_from_width` inverts
apparent size against a *declared* target width. It is correct for the aircraft
you told it about and wrong for anything else, so `lane.target_width_m` is unset
by default and `r` is emitted as `null`. Fusion must treat `null` as "no range",
never as zero.

**Effective range is ~20 m on a laptop webcam.** A Mini 4 Pro is ~0.35 m across
and needs roughly 20 px to detect. That arrives with the varifocal lens in build 2:

| Optics | ~20 px at |
|---|---|
| Laptop webcam, 65° | ~20 m |
| 6 mm lens, 50° | ~25 m |
| 60 mm lens, 5.3° | ~150 m |

The 5.3° figure is why the architecture has slew-to-cue. At range the camera
cannot search; it can only be pointed.

## Swapping in a better detector

The whole point of the config layout. To hand this system real drone data:

1. Train: `yolo detect train model=yolo11n.pt data=<dataset>.yaml imgsz=1280 epochs=100`
2. Drop the resulting `best.pt` into `models/`
3. Change one line in `configs/l2.yaml`:
   ```yaml
   model:
     weights: "models/your-model.pt"
   ```

Nothing in `skykiller/` changes. Train at the `imgsz` you intend to infer at.

**Current detector:** `doguilmak/Drone-Detection-YOLOv11x` (MIT), single `drone`
class, ~1,000 training images. It scores 0.849 on a clear photo of a quadcopter,
but a model trained on a thousand images will be markedly worse against a Thai
sky and treeline than its published mAP suggests. It is a placeholder that proves
the pipeline.

**Fallback:** if those weights are absent, the lane drops to stock COCO YOLO11n
filtered to `airplane`/`bird`/`kite`, relabelled `uav-candidate`. Every emitted
detection carries `provisional: true` and the viewer says so on screen. This
exists so a fresh clone demonstrates something; it is not a drone detector.

## Message contract

Section 2 of the spec board, unchanged:

```json
{"t_utc":1787945187.32,"src":"L2","az":342.13,"el":1.59,"r":null,"conf":0.62,
 "raw_id":"1","extra":{"label":"drone","bbox_xywh":[316.2,330.7,152.0,70.4],
 "provisional":false}}
```

`raw_id` is the tracker's id for this lane — the same field that will carry a
Remote ID serial from L1b. Azimuth is degrees clockwise from north, normalised to
`[0, 360)`, so a track crossing the boresight reads `358 → 0 → 2`; unwrap before
differencing. Elevation is degrees above the horizon.

## Tracker

BoT-SORT, ReID off, camera-motion compensation on. A drone at range is a few
dozen grey pixels, so appearance re-identification has nothing to work with and
motion association is what holds the track. CMC costs little now and earns its
place when the camera goes on a pan-tilt head. Requires `ultralytics >= 8.4.63`.

## Environment

Python **3.12** — torch has no 3.14 wheels yet, and 3.14 is this machine's
default `python3`. Device selection is `auto`: MPS on Apple silicon, then CUDA,
then CPU. Checking only for CUDA, as the spec board originally implied, would
have put the whole pipeline on the CPU on this machine.
