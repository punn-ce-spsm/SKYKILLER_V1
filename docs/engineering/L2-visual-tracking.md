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
| `python -m skykiller --config configs/hometest.yaml` | **Home test** — track household objects, no drone needed |

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

## Home test mode

`configs/hometest.yaml` points the lane at stock COCO YOLO11n filtered to a few
household classes, so the whole pipeline can be exercised indoors with no drone,
no flight and no permissions. **ACTION.md item 0** is the step-by-step procedure
and what each check proves.

It runs at ~13 fps rather than the drone model's 4.7, because YOLO11n is far
lighter than the YOLO11x the public drone weights use — which is also the
clearest evidence for the fine-tuning recommendation above.

Two config options make this work without code changes:

- **`model.classes`** — a class-id filter applied to *whatever* model is loaded.
  Previously the filter existed only on the fallback branch, so narrowing an
  explicit model was impossible.
- **`model.weights: null`** — "use the COCO model on purpose". Distinct from a
  named-but-missing file, which still warns loudly. Both mark detections
  `provisional: true` and put a banner in the viewer, because neither is a drone
  detector.

`model.label_override` forces every detection to one label; it is how the
airplane/bird/kite fallback reports `uav-candidate` instead of repeating COCO's
guess as though this system believed it. Left null during the home test so a cup
reads as "cup".

## Identity filtering

`identity.enabled` gates emission on a track matching a known identity. Tracks
that fail the gate are still produced by `stream()` and still drawn — they are
just never emitted. `should_emit()` in `l2_visual.py` is the only place that
decides, and it **fails closed**: with the gate on, a track carrying no verdict
is not emitted.

The stage is written against an abstract matcher because *"emit only the aircraft
whose Remote ID serial is on this list"* is the same operation. Faces are the
implementation that can be tested indoors today; L1b will supply another.

**Faces** use OpenCV's YuNet detector and SFace embedder — no new dependencies,
227 KB + 37 MB of ONNX, auto-downloaded on first use. Note the download must come
from `media.githubusercontent.com`: opencv_zoo keeps weights in Git LFS, and the
`raw.githubusercontent` URLs return a 131-byte pointer file that loads as a
corrupt ONNX model.

Faces are matched *inside* YOLO person boxes rather than detected independently,
which reuses the whole existing tracking path and means a confirmed track
survives the subject turning away.

**Verdicts are cached per track id** and re-checked every `recheck_every` frames
(default 15). Three distinct situations are kept distinct, which is the part
worth not collapsing:

| Situation | Behaviour |
|---|---|
| Face visible, scores above threshold | match |
| Face visible, scores below | not a match |
| **No face visible this frame** | **keep the previous verdict** — turning your head is not evidence of anything |
| No face visible, no previous verdict | not a match (fails closed) |

Measured on Ultralytics' bundled `zidane.jpg`: same face 0.912–0.950, different
person 0.072–0.077, against a 0.363 threshold. `tests/test_identity.py` asserts
that separation stays wide, and skips itself if the models are absent.

**Enrolment tries all four orientations** and picks the one whose best face has
the highest *confidence*. Two measured facts drive that:

- YuNet is not rotation invariant. A 90-degree rotated image yields **0** faces
  where the upright one yields 2. Phone portrait shots with a stripped EXIF tag
  are exactly this, so the first version of this code blamed the user's photo for
  a code limitation.
- A 180-degree image yields **3** confident-looking detections on a 2-face
  photo. So taking the *first* orientation that returns anything enrols garbage
  that then matches nobody — which is what the first fix did, and why the
  selection is by confidence rather than by ordering.

The live path deliberately does not do this: camera frames arrive upright, and
four detections per crop per frame would be pure cost.

### Two thresholds, doing different jobs

| Setting | Default | Question it answers |
|---|---|---|
| `identity.detect_threshold` | **0.5** | Is this region a face at all? |
| `identity.threshold` | 0.363 | Are these two faces the same person? |

Detection was lowered from 0.7 to 0.5 to accept marginal photos — poor light,
off-angle. Measured cost on a reference image containing exactly two faces:

| detect_threshold | detections | scores | false |
|---:|---:|---|---:|
| 0.9 | 2 | 0.94, 0.90 | 0 |
| 0.7 | 2 | 0.94, 0.90 | 0 |
| **0.5** | **3** | 0.94, 0.90, **0.61** | **1** |
| 0.3 | 4 | 0.94, 0.90, 0.61, 0.38 | 2 |

Discrimination is unaffected — same person 1.000, different person 0.029 — because
the two thresholds are independent. What does change is that spurious boxes now
exist, so live face selection takes the largest face **within 80% of the best
confidence in the crop** rather than the largest outright. Without that band a
big 0.61 artefact could outrank the real subject, embed as nobody, and flip a
confirmed track to not-target.

`tools/diagnose_enroll.py` reports every step of a failed enrolment — file type,
what loaded, EXIF handling, brightness, and faces per orientation per threshold.

An enrolled embedding is biometric data. `models/` is gitignored; nothing
uploads.

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
