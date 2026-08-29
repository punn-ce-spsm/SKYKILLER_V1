"""Command line for the L2 lane.

    python -m skykiller                       # webcam, viewer on
    python -m skykiller --source clip.mp4     # a recording
    python -m skykiller --no-show > dets.jsonl
    python -m skykiller fetch-model           # download the drone weights
    python -m skykiller calibrate --object-px 412 --object-m 0.9 --distance-m 5
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from . import config as cfgmod

#: Public single-class drone detector, MIT licence. Note the `weight/` prefix --
#: the model card names the file `best.pt` but it is not at the repo root.
WEIGHTS_URL = "https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/resolve/main/weight/best.pt"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skykiller", description="SKYKILLER L2 visual tracking lane")
    sub = p.add_subparsers(dest="command")

    p.add_argument("--config", help="path to a YAML config (default configs/l2.yaml)")
    p.add_argument("--source", help="webcam index (e.g. 0) or a video file path")
    p.add_argument("--hfov", type=float, help="horizontal field of view, degrees")
    p.add_argument("--target-width-m", type=float,
                   help="true target width in metres, enables range estimates (Mini 4 Pro = 0.35)")
    p.add_argument("--jsonl", help="also append detections to this file")
    p.add_argument("--no-show", action="store_true", help="headless; detections to stdout only")
    p.add_argument("--device", help="mps | cuda | cpu (default auto)")

    sub.add_parser("fetch-model", help="download the pretrained drone weights")

    enr = sub.add_parser("enroll", help="store a face embedding to track only that person")
    enr.add_argument("--image", required=True, help="a clear, front-on photo of the subject")
    enr.add_argument("--name", default="me", help="what to call this identity (default: me)")
    enr.add_argument("--out", help="where to write it (default models/identity/<name>.npy)")

    cal = sub.add_parser("calibrate", help="compute horizontal FOV from one reference photo")
    cal.add_argument("--width-px", type=int, default=1280, help="frame width in pixels")
    cal.add_argument("--object-px", type=float, required=True, help="object width as measured in the image")
    cal.add_argument("--object-m", type=float, required=True, help="true object width in metres")
    cal.add_argument("--distance-m", type=float, required=True, help="distance to the object in metres")
    return p


#: Where fetch-model writes when the active config has `weights: null`.
DEFAULT_WEIGHTS_PATH = "models/drone-yolo11x.pt"


def _cmd_fetch_model(cfg: cfgmod.Config) -> int:
    # `weights` is null in configs that deliberately use COCO (the home test),
    # so fetch-model still needs somewhere sensible to put the download.
    dest = cfgmod.REPO_ROOT / (cfg.model.weights or DEFAULT_WEIGHTS_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already present: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return 0
    print(f"downloading {WEIGHTS_URL}\n         -> {dest}")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, dest)  # noqa: S310 -- constant https URL
    except Exception as exc:  # noqa: BLE001
        print(f"download failed: {exc}\nThe lane still runs on the fallback detector.", file=sys.stderr)
        return 1
    print(f"done: {dest.stat().st_size / 1e6:.0f} MB")
    return 0


def _cmd_enroll(args: argparse.Namespace) -> int:
    from . import identity

    out = args.out or f"models/identity/{args.name}.npy"
    try:
        path = identity.enroll(args.image, args.name, cfgmod.REPO_ROOT / out)
    except ValueError as exc:
        print(f"enrol failed: {exc}", file=sys.stderr)
        return 1

    print(f"enrolled '{args.name}' -> {path}")
    print("\nThis file is a face embedding: biometric data. It stays on this")
    print("machine and models/ is gitignored. Turn the filter on with:\n")
    print(f"  python -m skykiller --config configs/face.yaml")
    print(f"\nor set identity.enabled: true and identity.reference: {out}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .geometry import hfov_from_reference

    hfov = hfov_from_reference(args.width_px, args.object_px, args.object_m, args.distance_m)
    print(f"measured horizontal FOV: {hfov:.2f} deg")
    print(f"put this in configs/l2.yaml:\n\ncamera:\n  hfov_deg: {hfov:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "enroll":
        return _cmd_enroll(args)

    cfg = cfgmod.load(args.config)
    if args.command == "fetch-model":
        return _cmd_fetch_model(cfg)

    if args.source is not None:
        cfg.source = args.source
    if args.hfov is not None:
        cfg.camera.hfov_deg = args.hfov
    if args.target_width_m is not None:
        cfg.lane.target_width_m = args.target_width_m
    if args.jsonl:
        cfg.sink.jsonl_path = args.jsonl
    if args.device:
        cfg.model.device = args.device
    if args.no_show:
        cfg.show = False

    # Only nag when the config asked for weights that are not there. A null
    # `weights` is a deliberate choice, not a missing download.
    if cfg.model.weights and not (cfgmod.REPO_ROOT / cfg.model.weights).exists():
        print("[skykiller] hint: run `python -m skykiller fetch-model` for the drone detector.",
              file=sys.stderr)

    from . import l2_visual  # noqa: PLC0415 -- defers the torch import

    return l2_visual.run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
