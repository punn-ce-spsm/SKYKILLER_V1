"""Message contracts for the SKYKILLER bus.

`Detection` is fixed by section 2 of the spec board and is emitted by *every*
sensor lane -- L1a RF, L1b Remote ID, L2 visual, L3 acoustic, and the L4 radar
simulator. Fusion subscribes to this and nothing else, so the field names here
are load-bearing: changing one is a breaking change across every lane.

    Detection { t_utc, src, az, el, r, conf, raw_id }

`raw_id` carries whatever identifier the lane natively produces -- a Remote ID
serial for L1b, a visual track id for L2 -- or None when the lane has no notion
of identity. Anything lane-specific that fusion does not need goes in `extra`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

#: Sensor lane identifiers. Fusion uses these to weight and gate detections.
SRC_RF = "L1a"
SRC_REMOTE_ID = "L1b"
SRC_VISUAL = "L2"
SRC_ACOUSTIC = "L3"
SRC_RADAR_SIM = "L4"


@dataclass(slots=True)
class Detection:
    """One observation from one lane at one instant.

    Angles are degrees in the mast's local frame: `az` clockwise from north,
    `el` positive above the horizon. A lane that cannot measure range leaves
    `r` as None -- it is not zero, and fusion must not read it as zero.
    """

    src: str
    az: float
    el: float
    conf: float
    t_utc: float = field(default_factory=time.time)
    r: float | None = None
    raw_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError(f"conf must be in [0,1], got {self.conf}")
        if self.r is not None and self.r < 0:
            raise ValueError(f"r must be non-negative or None, got {self.r}")
        # Normalise azimuth into [0,360) so downstream gating never has to.
        self.az %= 360.0

    def to_json(self) -> str:
        """Serialise to one line of JSON -- the on-the-wire form."""
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Detection":
        return cls(**json.loads(line))
