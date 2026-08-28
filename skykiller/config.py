"""Configuration for the L2 lane. YAML over dataclass defaults.

Everything a demonstration might need to change lives here rather than in code.
The one that matters most is `model.weights`: handing this system a better
detector is a matter of dropping a .pt file in `models/` and editing one line.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "l2.yaml"


@dataclass(slots=True)
class CameraCfg:
    hfov_deg: float = 65.0  # replace with a measured value: `skykiller calibrate`
    az_deg: float = 0.0
    el_deg: float = 0.0


@dataclass(slots=True)
class ModelCfg:
    #: Preferred detector. Absent on a fresh clone -- run `skykiller fetch-model`.
    weights: str = "models/drone-yolo11x.pt"
    #: Used when `weights` is missing, so a fresh clone still demonstrates.
    fallback_weights: str = "yolo11n.pt"
    #: COCO ids kept by the fallback: 4 airplane, 14 bird, 33 kite.
    fallback_classes: list[int] = field(default_factory=lambda: [4, 14, 33])
    fallback_label: str = "uav-candidate"
    device: str = "auto"  # auto | mps | cuda | cpu
    #: Detector input size. "auto" matches the capture width, which matters more
    #: than it sounds: downscaling the frame throws away the few pixels a distant
    #: drone occupies. At imgsz 640 a 1280-wide frame loses a 150px target
    #: entirely -- measured, not theorised. Effective range scales with
    #: imgsz/capture_width, so halving this halves how far the lane can see.
    imgsz: int | str = "auto"
    conf: float = 0.25
    iou: float = 0.5


@dataclass(slots=True)
class MqttCfg:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    topic: str = "skykiller/detections"


@dataclass(slots=True)
class SinkCfg:
    stdout: bool = True
    jsonl_path: str | None = None
    mqtt: MqttCfg = field(default_factory=MqttCfg)


@dataclass(slots=True)
class LaneCfg:
    #: True width of the expected target, metres. 0.35 for a DJI Mini 4 Pro.
    #: Left unset by default: a range derived from an assumed size is a guess,
    #: and fusion should be told None rather than a confident wrong number.
    target_width_m: float | None = None
    min_conf_emit: float = 0.25


@dataclass(slots=True)
class Config:
    source: str = "0"
    tracker: str = "botsort.yaml"
    show: bool = True
    window: str = "SKYKILLER L2 - visual tracking"
    camera: CameraCfg = field(default_factory=CameraCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    lane: LaneCfg = field(default_factory=LaneCfg)
    sink: SinkCfg = field(default_factory=SinkCfg)


def _merge(target: Any, data: dict[str, Any]) -> Any:
    """Overlay a dict onto a dataclass instance, recursing into nested ones.

    Unknown keys raise rather than being ignored -- a silently dropped setting is
    a config that lies about what the system is doing.
    """
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key {key!r} (expected one of {sorted(known)})")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, value)
    return target


def load(path: str | Path | None = None) -> Config:
    """Load config, falling back to defaults when no file is present."""
    cfg = Config()
    p = Path(path) if path else DEFAULT_CONFIG
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}
        _merge(cfg, raw)
    return cfg


def resolve_device(requested: str) -> str:
    """Pick a torch device. `auto` prefers Apple MPS, then CUDA, then CPU.

    This machine is an Apple M5, so MPS is the fast path -- checking only for
    CUDA would silently drop the whole pipeline onto the CPU.
    """
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
