"""What the terrain hides, which is the entire reason this product is airborne.

The customer already owns ground cameras. Their problem is not sensor quality,
it is that a treeline or a ridge cuts the line of sight, and no lens fixes a
line of sight. A camera at 3 m behind a 20 m treeline 200 m away cannot see
anything below 88 m altitude at 1 km, or 173 m at 2 km -- while the drones that
matter fly at 50 m. That single fact is the argument for putting the observer
on a tether at 80-200 m, and this module is where it is computed rather than
asserted.

The model is a vertical screen: an obstacle is a crest line between two points
on the ground with an altitude, and it blocks a line of sight that crosses it
below the crest. Crude next to a real DEM, and deliberately so -- a treeline and
a ridge are both screens, the geometry that matters is the crossing height, and
a demonstration needs a number the customer can check on a map rather than a
terrain database they have to trust.

What it does not model: refraction, earth curvature (under 3 km these are
centimetres), vegetation you can partly see through, and anything below the
crest line. All of those make the real picture *better* than this predicts,
which is the right direction for a claim about what the customer cannot see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Obstacle:
    """A screen between two ground points, opaque below `crest_m`.

    `a` and `b` are the ends of the crest line in ENU; only their horizontal
    components are used. `crest_m` is an absolute altitude in the same frame as
    everything else, not a height above local ground -- a ridge and a treeline
    on a slope are then the same object, and there is no second datum to get
    wrong.
    """

    name: str
    a: np.ndarray
    b: np.ndarray
    crest_m: float

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float)[:2]
        self.b = np.asarray(self.b, dtype=float)[:2]
        if np.allclose(self.a, self.b):
            raise ValueError(f"obstacle {self.name!r}: a and b are the same point")

    def crossing(self, observer, target) -> float | None:
        """Fraction along observer->target where the sight line crosses, or None.

        Purely a 2-D question -- where the horizontal path crosses the crest
        line. The altitude test is separate, in `blocks`, because the two
        failures are different: missing the screen entirely and passing over it.
        """
        o = np.asarray(observer, dtype=float)[:2]
        t = np.asarray(target, dtype=float)[:2]
        r, s = t - o, self.b - self.a
        denom = r[0] * s[1] - r[1] * s[0]
        if abs(denom) < 1e-12:
            return None                       # parallel, or a degenerate path
        diff = self.a - o
        u = (diff[0] * s[1] - diff[1] * s[0]) / denom      # along observer->target
        v = (diff[0] * r[1] - diff[1] * r[0]) / denom      # along the crest line
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None                       # the crossing is off one end
        return float(u)

    def blocks(self, observer, target) -> bool:
        """Is the target hidden from the observer by this screen?"""
        u = self.crossing(observer, target)
        if u is None:
            return False
        o = np.asarray(observer, dtype=float)
        t = np.asarray(target, dtype=float)
        return o[2] + u * (t[2] - o[2]) < self.crest_m

    def lowest_visible_alt(self, observer, target_xy) -> float | None:
        """Lowest altitude at `target_xy` this screen still permits to be seen.

        The number the customer recognises: 'behind that treeline you cannot see
        anything below X'. None when the screen is not in the way at all.
        """
        probe = np.array([target_xy[0], target_xy[1], self.crest_m + 1e6])
        u = self.crossing(observer, probe)
        if u is None or u <= 0.0:
            return None
        o = np.asarray(observer, dtype=float)
        # Solve o_z + u * (h - o_z) = crest for h: the altitude whose sight line
        # grazes the crest exactly.
        return float(o[2] + (self.crest_m - o[2]) / u)


def visible(observer, target, obstacles: list[Obstacle]) -> bool:
    """Clear line of sight past every screen."""
    return not any(ob.blocks(observer, target) for ob in obstacles)


def lowest_visible_alt(observer, target_xy, obstacles: list[Obstacle]) -> float:
    """The worst screen wins: the highest floor any of them imposes."""
    floors = [f for f in (ob.lowest_visible_alt(observer, target_xy)
                          for ob in obstacles) if f is not None]
    return max(floors) if floors else -math.inf


def treeline(name: str, height_m: float, distance_m: float, bearing_deg: float = 0.0,
             width_m: float = 20_000.0, observer_xy=(0.0, 0.0)) -> Obstacle:
    """A screen `distance_m` from a point, square-on to `bearing_deg`.

    Convenience for the common case and for checking the model against a
    hand-computed number, where 'a treeline 200 m out' is how anyone states it.
    """
    b = math.radians(bearing_deg)
    forward = np.array([math.sin(b), math.cos(b)])
    across = np.array([math.cos(b), -math.sin(b)])
    centre = np.asarray(observer_xy, dtype=float) + forward * distance_m
    return Obstacle(name=name,
                    a=centre - across * width_m / 2,
                    b=centre + across * width_m / 2,
                    crest_m=height_m)
