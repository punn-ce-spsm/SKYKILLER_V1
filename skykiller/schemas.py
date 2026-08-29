"""Message contracts for the SKYKILLER bus.

`Detection` is fixed by section 2 of the spec board and is emitted by *every*
sensor lane -- L1a RF, L1b Remote ID, L2 visual, L3 acoustic, and the L4 radar
simulator. Fusion subscribes to this and nothing else, so the field names here
are load-bearing: changing one is a breaking change across every lane.

    Detection { t_utc, src, az, el, r, conf, raw_id }

`raw_id` carries whatever identifier the lane natively produces -- a Remote ID
serial for L1b, a visual track id for L2 -- or None when the lane has no notion
of identity. Anything lane-specific that fusion does not need goes in `extra`.

Build 2 adds two more contracts downstream of fusion:

    Friendly { t_utc, id, enu, source, conf }
    Track    { id, enu, cov, vel, iff, sites, first_seen, last_seen, score }

`Friendly` is a cooperative position report -- the feed only our own aircraft
produce. `Track` is what fusion publishes to the console and to TAK: a position
with an honest error ellipse and an IFF verdict.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

#: Sensor lane identifiers. Fusion uses these to weight and gate detections.
SRC_RF = "L1a"
SRC_REMOTE_ID = "L1b"
SRC_VISUAL = "L2"
SRC_ACOUSTIC = "L3"
SRC_RADAR_SIM = "L4"


#: IFF verdicts. Ordered by what they permit, least to most.
IFF_FRIENDLY = "FRIENDLY"
IFF_UNKNOWN = "UNKNOWN"
IFF_HOSTILE = "HOSTILE"


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


@dataclass(slots=True)
class Friendly:
    """One cooperative position report from an aircraft on our side.

    This is the *only* thing that makes a track friendly. Nothing about a
    drone's appearance, frequency or behaviour is used, because none of those
    separate the sides when both fly the same airframes on the same bands.

    `sigma_m` is how well the friendly knows its own position, one sigma. A
    drone reporting GNSS is worth a few metres; a hand-typed orbit centre is
    worth tens. It widens the correlation gate, so an over-optimistic value
    here makes friendlies harder to recognise, not easier.
    """

    id: str
    enu: tuple[float, float, float]
    source: str                       # "remote-id", "telemetry", "c2", "manual"
    conf: float = 1.0
    sigma_m: float = 10.0
    t_utc: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError(f"conf must be in [0,1], got {self.conf}")
        if self.sigma_m <= 0:
            raise ValueError(f"sigma_m must be positive, got {self.sigma_m}")
        self.enu = tuple(float(v) for v in self.enu)
        if len(self.enu) != 3:
            raise ValueError(f"enu must be 3 numbers, got {self.enu}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Friendly":
        d = json.loads(line)
        d["enu"] = tuple(d["enu"])
        return cls(**d)


@dataclass(slots=True)
class Track:
    """A fused air contact: where it is, how well we know, and whose it is.

    `cov` is a 3x3 position covariance in metres squared, carried alongside the
    position everywhere rather than collapsed to a single radius. The console
    and the effector both need the shape of the error, not just its size: a
    bearings-only fix is an elongated ellipsoid pointing down-range, and a beam
    aimed at its centre covers it or does not depending on which way it lies.
    """

    id: str
    enu: tuple[float, float, float]
    cov: list[list[float]]
    iff: str = IFF_UNKNOWN
    vel: tuple[float, float, float] | None = None
    sites: list[str] = field(default_factory=list)
    friendly_id: str | None = None    # which feed entry matched, when one did
    score: float = 0.0                # detection confidence, carried through
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.iff not in (IFF_FRIENDLY, IFF_UNKNOWN, IFF_HOSTILE):
            raise ValueError(f"iff must be one of the three verdicts, got {self.iff!r}")
        self.enu = tuple(float(v) for v in self.enu)
        if len(self.enu) != 3:
            raise ValueError(f"enu must be 3 numbers, got {self.enu}")

    @property
    def sigma_m(self) -> float:
        """One-sigma position error on the worst axis. The number to quote."""
        return float(np.sqrt(max(np.linalg.eigvalsh(np.array(self.cov)).max(), 0.0)))

    @property
    def age_s(self) -> float:
        return self.last_seen - self.first_seen

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Track":
        d = json.loads(line)
        d["enu"] = tuple(d["enu"])
        if d.get("vel") is not None:
            d["vel"] = tuple(d["vel"])
        return cls(**d)
