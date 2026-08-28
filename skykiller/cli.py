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

    cal = sub.add_parser("calibrate", help="compute horizontal FOV from one reference photo")
    cal.add_argument("--width-px", type=int, default=1280, help="frame width in pixels")
    cal.add_argument("--object-px", type=float, required=True, help="object width as measured in the image")
    cal.add_argument("--object-m", type=float, required=True, help="true object width in metres")
    cal.add_argument("--distance-m", type=float, required=True, help="distance to the object in metres")
    return p


def _cmd_fetch_model(cfg: cfgmod.Config) -> int:
    dest = cfgmod.REPO_ROOT / cfg.model.weights
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

    if not (cfgmod.REPO_ROOT / cfg.model.weights).exists():
        print("[skykiller] hint: run `python -m skykiller fetch-model` for the drone detector.",
              file=sys.stderr)

    from . import l2_visual  # noqa: PLC0415 -- defers the torch import

    return l2_visual.run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
