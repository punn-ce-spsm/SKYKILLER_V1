"""Surveyed observation posts, and the bridge from the bus to the solver.

A `Detection` on the bus carries a bearing and the name of the lane that
produced it -- but no position, deliberately: build 1's contract is unchanged
and a sensor should not have to know where it is standing. The position lives
here, in config, keyed by `Detection.src`. That is what lets two posts fuse
without either lane's code changing.

The tether is why this file is short. A tethered observer hangs above a surveyed
mast, so its position is *measured once and then known*, not estimated per
frame. A free-flying observer would need its own navigation solution and its own
error budget on top of the bearing error; the tether removes that entire
problem, which is most of the reason the platform is tethered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Camera
from .schemas import Detection
from .triangulate import MIN_SEPARATION_DEG, Ray

#: Bearing accuracy of one post, one sigma, degrees. Not a guess to leave alone:
#: it sets both the association gate and every covariance the console reports.
#: The pinhole model with a hand-entered field of view is worth about 2-3 deg
#: absolute (see geometry.py); 0.5 deg assumes the FOV has been *measured* per
#: ACTION.md item 0 check 3. Until it has, this number is optimistic.
DEFAULT_SIGMA_DEG = 0.5


@dataclass(slots=True)
class Site:
    """One observation post: where it stands, and what it looks through."""

    name: str
    enu: np.ndarray            # surveyed position, metres, ENU from the datum
    camera: Camera
    sigma_deg: float = DEFAULT_SIGMA_DEG

    def __post_init__(self) -> None:
        self.enu = np.asarray(self.enu, dtype=float)
        if self.enu.shape != (3,):
            raise ValueError(f"site {self.name}: enu must be 3 numbers, got {self.enu.shape}")
        if self.sigma_deg <= 0:
            raise ValueError(f"site {self.name}: sigma_deg must be positive")

    @property
    def height_m(self) -> float:
        """Height above the datum plane -- the whole product, in one number."""
        return float(self.enu[2])

    def ray(self, det: Detection) -> Ray:
        """Turn one of this site's detections into a ray in the world frame.

        The bearing arrives already in world axes: `Camera.bearing` applied the
        mounting pose upstream. All this adds is the origin.
        """
        return Ray(origin=self.enu.copy(), az_deg=det.az, el_deg=det.el,
                   src=self.name, sigma_deg=self.sigma_deg)


@dataclass(slots=True)
class SiteNetwork:
    """The posts, addressed by the `src` field their detections carry."""

    sites: dict[str, Site] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.sites) >= 2:
            self._check_baseline()

    def _check_baseline(self) -> None:
        """Refuse a network whose posts are effectively in the same place.

        Two posts a metre apart are one post with extra cabling: the parallax
        that makes range observable scales with the baseline, so a short one
        gives a fix whose down-range error is enormous but whose covariance --
        honestly computed -- says so. This catches the surveying mistake early,
        where it reads as a config error rather than as mysteriously bad range.
        """
        names = list(self.sites)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = float(np.linalg.norm(self.sites[a].enu - self.sites[b].enu))
                if d < 1.0:
                    raise ValueError(
                        f"sites {a!r} and {b!r} are {d:.2f} m apart. Two posts at "
                        f"the same place cannot triangulate -- check the survey.")

    @property
    def baseline_m(self) -> float:
        """Widest separation between any two posts. Zero if there is only one."""
        best = 0.0
        for i, a in enumerate(self.sites.values()):
            for b in list(self.sites.values())[i + 1:]:
                best = max(best, float(np.linalg.norm(a.enu - b.enu)))
        return best

    def rays(self, dets: list[Detection]) -> list[Ray]:
        """Detections to rays, dropping any whose `src` is not a known post.

        Dropping is deliberate. An unknown source means the config and the bus
        disagree about who is out there, and inventing a position for it would
        put a confident fix on the console derived from a site that does not
        exist. `unknown_sources` reports what was dropped so the caller can say
        so out loud rather than silently fusing less than it thinks.
        """
        return [self.sites[d.src].ray(d) for d in dets if d.src in self.sites]

    def unknown_sources(self, dets: list[Detection]) -> set[str]:
        return {d.src for d in dets if d.src not in self.sites}

    def max_range_m(self, min_separation_deg: float = MIN_SEPARATION_DEG) -> float:
        """Range at which the two posts stop looking at the target from
        meaningfully different directions, and the solver starts refusing.

        The crossing angle is about `atan(B / R)`, so it shrinks as the target
        recedes and the useful range is set by the baseline alone -- not by the
        camera, the detector, or the frame rate. Measured against the real
        solver at sigma 0.5 deg, fixes stay at 100% out to about half this
        figure and then fall away: a 100 m baseline fixes every frame at 1 km,
        91% at 2 km and 73% at 2.5 km.

        The practical reading is that a 100 m baseline is a 1.5 km system and a
        200 m baseline covers the whole 2.5 km early-warning band. Spreading
        the masts is free; it is the one deployment choice that buys range.
        """
        if self.baseline_m <= 0:
            return 0.0
        return self.baseline_m / math.tan(math.radians(min_separation_deg))

    def expected_accuracy_m(self, range_m: float) -> float:
        """Rough down-range error at a given range, for sizing a deployment.

        Two posts a baseline B apart, each good to sigma, fix a target at range
        R with a down-range error of about `sigma_rad * R^2 / B` -- the range
        term is squared, which is why doubling the standoff quadruples the
        error and why the baseline is the lever that matters. Use it to answer
        "how far apart do the masts need to be", not to report accuracy: the
        real number comes from `Fix.cov`, which knows the actual geometry.
        """
        if self.baseline_m <= 0:
            return float("inf")   # one post gives a bearing, never a position
        sigma = max(s.sigma_deg for s in self.sites.values())
        return math.radians(sigma) * range_m ** 2 / self.baseline_m
