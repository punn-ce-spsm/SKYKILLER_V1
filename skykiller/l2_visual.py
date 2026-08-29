"""The L2 visual lane: camera in, `Detection` messages out.

Ultralytics owns capture, detection and tracking, so this module is the glue
that turns a tracked box into the bus contract: pixel centre to bearing, box
width to an optional range, tracker id to `raw_id`.

Nothing here knows about fusion, the console, or any other lane. That is the
point of the bus -- see section 2 of the spec board.
"""

from __future__ import annotations

import itertools
import sys
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import Config, REPO_ROOT, resolve_device
from .geometry import Camera
from .schemas import Detection, SRC_VISUAL


@dataclass(slots=True)
class LoadedModel:
    """A detector plus the label policy that goes with it."""

    model: object
    label_of: dict[int, str]
    class_filter: list[int] | None
    is_fallback: bool
    weights: str


def load_model(cfg: Config) -> LoadedModel:
    """Load the configured detector and the label policy that goes with it.

    Three cases, deliberately distinguished:

    * `weights` names a file that exists -- use it, no warning.
    * `weights` is null -- the caller chose the COCO model on purpose (the home
      test does this). A note, not a warning.
    * `weights` names a file that is missing -- the caller expected a drone
      detector and has not got one. Warn loudly, and mark every detection
      provisional so nothing downstream mistakes it for a drone claim.
    """
    from ultralytics import YOLO  # noqa: PLC0415 -- heavy import, deferred

    mc = cfg.model
    path = None
    if mc.weights:
        path = Path(mc.weights)
        if not path.is_absolute():
            path = REPO_ROOT / path

    if path is not None and path.exists():
        model, weights, is_fallback = YOLO(str(path)), str(path), False
        classes, label_override = mc.classes, mc.label_override
    else:
        if path is None:
            print(f"[l2] using {mc.fallback_weights} by configuration.", file=sys.stderr)
        else:
            print(
                f"[l2] {path} not found -- falling back to {mc.fallback_weights}.\n"
                f"[l2] This demonstrates the pipeline but is NOT a drone detector. "
                f"Run `python -m skykiller fetch-model` for the real weights.",
                file=sys.stderr,
            )
        model, weights, is_fallback = YOLO(mc.fallback_weights), mc.fallback_weights, True
        # An explicit `classes` wins; otherwise fall back to the airplane/bird/kite
        # guess. The label is only overridden when the caller has not chosen
        # classes -- during the home test, "cup" should read as "cup".
        classes = mc.classes if mc.classes is not None else list(mc.fallback_classes)
        label_override = mc.label_override
        if label_override is None and mc.classes is None:
            label_override = mc.fallback_label

    label_of = dict(getattr(model, "names", {}) or {0: "object"})
    if label_override:
        label_of = dict.fromkeys(label_of, label_override)
    if classes:
        kept = ", ".join(f"{i}:{label_of.get(i, i)}" for i in classes)
        print(f"[l2] class filter -> {kept}", file=sys.stderr)

    return LoadedModel(model, label_of, list(classes) if classes else None, is_fallback, weights)


def _crop(frame, cx: float, cy: float, w: float, h: float):
    """Clamped box crop, for handing a detection to the identity matcher."""
    fh, fw = frame.shape[:2]
    x1, y1 = max(int(cx - w / 2), 0), max(int(cy - h / 2), 0)
    x2, y2 = min(int(cx + w / 2), fw), min(int(cy + h / 2), fh)
    return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def _detections_from_result(
    result, cam: Camera, cfg: Config, loaded: LoadedModel,
    verdicts=None, frame_no: int = 0,
) -> list[Detection]:
    """Convert one frame of tracked boxes into bus messages.

    Returns *every* track, identity verdict attached. Filtering happens at the
    emit step, not here, so the viewer can still show contacts that are not the
    target -- a console that hides other traffic is the wrong thing for a C2
    system.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []

    # .xywh is centre-x, centre-y, width, height in pixels.
    xywh = boxes.xywh.cpu().tolist()
    ids = boxes.id.int().cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    clss = boxes.cls.int().cpu().tolist()

    # Ultralytics reindexes results to the tracked subset, so these are the same
    # length. If a future version stops doing that, drop the ragged tail rather
    # than raising -- a sensor lane that dies mid-flight is worse than one that
    # reports less for a frame.
    n = min(len(xywh), len(ids), len(confs), len(clss))
    if not (len(xywh) == len(ids) == len(confs) == len(clss)):
        print(f"[l2] ragged boxes: xywh={len(xywh)} id={len(ids)} conf={len(confs)} "
              f"cls={len(clss)}; using {n}", file=sys.stderr)

    out: list[Detection] = []
    for (cx, cy, w, h), tid, conf, cls in zip(xywh[:n], ids[:n], confs[:n], clss[:n]):
        if conf < cfg.lane.min_conf_emit:
            continue
        az, el = cam.bearing(cx, cy)
        rng = None
        if cfg.lane.target_width_m:
            rng = cam.range_from_width(w, cfg.lane.target_width_m)

        extra = {
            "label": loaded.label_of.get(cls, str(cls)),
            "bbox_xywh": [round(v, 1) for v in (cx, cy, w, h)],
            "provisional": loaded.is_fallback,
        }
        if verdicts is not None:
            crop = _crop(result.orig_img, cx, cy, w, h)
            is_match, score = verdicts.verdict(str(tid), crop, frame_no)
            extra["identity"] = {
                "name": cfg.identity.name if is_match else None,
                "match": is_match,
                "score": None if score is None else round(score, 3),
            }

        out.append(
            Detection(src=SRC_VISUAL, az=az, el=el, conf=float(conf),
                      r=rng, raw_id=str(tid), extra=extra)
        )

    if verdicts is not None:
        # Every track the tracker is holding, including ones filtered out above
        # for low confidence. Forgetting on the emitted set would drop a
        # confirmed target's verdict the moment its detection confidence dipped
        # for one frame -- and if its face were turned away just then, it would
        # come back as not-target.
        verdicts.forget({str(i) for i in ids[:n]})
    return out


def should_emit(det: Detection, cfg: Config) -> bool:
    """Identity gate. With identity off, every held track goes on the wire."""
    if not cfg.identity.enabled:
        return True
    return bool(det.extra.get("identity", {}).get("match"))


def _probe_source(source) -> tuple[int, int] | None:
    """Read the capture size without consuming the stream, for imgsz=auto."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(source)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return (w, h) if w > 0 and h > 0 else None


def _resolve_imgsz(cfg: Config, source) -> int:
    """Pick the detector input size, defaulting to the full capture width."""
    if cfg.model.imgsz != "auto":
        return int(cfg.model.imgsz)
    probed = _probe_source(source)
    if probed is None:
        print("[l2] could not probe source size; imgsz falling back to 1280", file=sys.stderr)
        return 1280
    # Ultralytics reads a single int as the longest side; round to a multiple of 32.
    return max(320, round(max(probed) / 32) * 32)


def stream(cfg: Config) -> Iterator[tuple[object, list[Detection], Camera]]:
    """Yield `(result, detections, camera)` per frame. The lane's whole job.

    Kept as a generator so the viewer, the tests and any future caller share one
    code path -- there is no second, subtly different pipeline for headless runs.
    """
    loaded = load_model(cfg)
    verdicts = None
    if cfg.identity.enabled:
        from . import identity  # noqa: PLC0415 -- only when the feature is on

        matcher = identity.load(cfg.identity.reference, cfg.identity.name,
                                cfg.identity.threshold, cfg.identity.detect_threshold)
        verdicts = identity.TrackVerdicts(matcher, cfg.identity.recheck_every)
        print(f"[l2] identity filter on: emitting only '{cfg.identity.name}' "
              f"(threshold {cfg.identity.threshold})", file=sys.stderr)

    device = resolve_device(cfg.model.device)
    source = int(cfg.source) if str(cfg.source).isdigit() else cfg.source
    imgsz = _resolve_imgsz(cfg, source)

    print(f"[l2] weights={loaded.weights} device={device} tracker={cfg.tracker} imgsz={imgsz}",
          file=sys.stderr)

    cam: Camera | None = None
    results = loaded.model.track(
        source=source,
        stream=True,
        persist=True,
        tracker=cfg.tracker,
        device=device,
        imgsz=imgsz,
        conf=cfg.model.conf,
        iou=cfg.model.iou,
        classes=loaded.class_filter,
        verbose=False,
    )

    try:
        first = next(results)
    except StopIteration:
        print(f"[l2] source produced no frames: {cfg.source!r}", file=sys.stderr)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[l2] could not open source {cfg.source!r}: {exc}", file=sys.stderr)
        return

    frame_no = 0
    for result in itertools.chain([first], results):
        frame_no += 1
        h, w = result.orig_img.shape[:2]
        if cam is None or (cam.width, cam.height) != (w, h):
            # Build the camera from the frame we actually got, not from config.
            # A webcam that negotiates 640x480 when asked for 1280x720 would
            # otherwise put every bearing out by the ratio of the two -- and a
            # source whose size changes mid-stream would keep using the stale
            # focal length, which is wrong silently rather than loudly.
            if cam is not None:
                print(f"[l2] frame size changed {cam.width}x{cam.height} -> {w}x{h}; "
                      f"rebuilding camera", file=sys.stderr)
            cam = Camera(
                width=w,
                height=h,
                hfov_deg=cfg.camera.hfov_deg,
                az_deg=cfg.camera.az_deg,
                el_deg=cfg.camera.el_deg,
            )
            print(
                f"[l2] frame {w}x{h} hfov={cam.hfov_deg:.1f} vfov={cam.vfov_deg:.1f} "
                f"f={cam.focal_px:.0f}px",
                file=sys.stderr,
            )
            if imgsz < w:
                print(
                    f"[l2] WARNING: imgsz={imgsz} is below the capture width {w}. "
                    f"The frame is downscaled {w / imgsz:.1f}x before detection, so a "
                    f"distant drone loses that fraction of its pixels and effective "
                    f"range drops by about the same factor. Set model.imgsz: auto.",
                    file=sys.stderr,
                )
        yield result, _detections_from_result(result, cam, cfg, loaded, verdicts, frame_no), cam


def run(cfg: Config) -> int:
    """Run the lane to completion, emitting to the configured sinks."""
    from . import sinks, viewer  # noqa: PLC0415 -- viewer pulls cv2

    sink = sinks.build(cfg)
    view = viewer.Viewer(cfg) if cfg.show else None
    frames = 0
    t0 = time.perf_counter()
    recent: deque[float] = deque(maxlen=30)
    last = t0

    try:
        for result, dets, cam in stream(cfg):
            frames += 1
            now = time.perf_counter()
            recent.append(now - last)
            last = now
            for det in dets:
                if should_emit(det, cfg):
                    sink.emit(det)
            if view is not None:
                # Mean over the last 30 frames, not since launch -- an average
                # since t0 keeps reporting the warm-up cost forever.
                fps = 1.0 / max(sum(recent) / len(recent), 1e-9)
                if not view.show(result, dets, cam, fps):
                    break
    except KeyboardInterrupt:
        print("\n[l2] stopped", file=sys.stderr)
    finally:
        sink.close()
        if view is not None:
            view.close()

    elapsed = time.perf_counter() - t0
    if frames:
        print(
            f"[l2] {frames} frames in {elapsed:.1f}s -- "
            f"{frames / elapsed:.1f} fps, {1000 * elapsed / frames:.1f} ms/frame",
            file=sys.stderr,
        )
    return 0
