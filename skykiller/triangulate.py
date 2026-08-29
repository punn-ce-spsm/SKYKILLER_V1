"""Intersect bearings from several sites into one position.

A single camera gives a ray, never a point -- range is unobservable from one
fixed sensor no matter how good the tracker. Two surveyed sites fix that, and
that is the whole reason the product is two tethered posts rather than one.

Frame is ENU throughout: X east, Y north, Z up, metres, origin at a surveyed
datum. Azimuth is degrees clockwise from north, elevation degrees above the
horizon -- the same convention `geometry.Camera.bearing` emits, so a `Detection`
drops straight in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Reject a fix whose rays are too close to parallel to intersect meaningfully.
#: Two rays 1 degree apart at 1 km put the crossing point kilometres away with
#: enormous error; better to report nothing than a confident fiction.
MIN_SEPARATION_DEG = 2.0

#: Reject when skew rays pass further apart than this many times the miss that
#: honest bearing noise would produce. The gate has to be dimensionless because
#: the noise miss grows linearly with range: measured at sigma 0.5 deg, the 99th
#: percentile miss is about 1.9 m at 120 m and 32 m at 2 km. A fixed metre gate
#: is therefore either useless up close or trigger-happy far out.
DEFAULT_MISS_GATE_SIGMAS = 4.0

#: Floor under the gate, so a target a few metres away is not rejected by a gate
#: that has shrunk to millimetres.
MIN_MISS_GATE_M = 1.0


def unit(az_deg: float, el_deg: float) -> np.ndarray:
    """Bearing to a unit vector in ENU."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([
        math.sin(az) * math.cos(el),   # east
        math.cos(az) * math.cos(el),   # north
        math.sin(el),                  # up
    ])


def to_bearing(v: np.ndarray) -> tuple[float, float]:
    """Inverse of `unit`: an ENU vector back to (azimuth, elevation) degrees."""
    east, north, up = v
    az = math.degrees(math.atan2(east, north)) % 360.0
    el = math.degrees(math.asin(up / np.linalg.norm(v)))
    return az, el


@dataclass(slots=True)
class Ray:
    """One site's line of sight: where it stands, and which way it is looking."""

    origin: np.ndarray       # ENU position of the site, metres
    az_deg: float
    el_deg: float
    src: str = ""            # which site produced it
    sigma_deg: float = 0.5   # bearing accuracy, one sigma

    @property
    def direction(self) -> np.ndarray:
        return unit(self.az_deg, self.el_deg)


@dataclass(slots=True)
class Fix:
    """A fused position and how much to trust it."""

    enu: np.ndarray          # metres
    cov: np.ndarray          # 3x3 covariance, metres squared
    miss_m: float            # how close the rays came to actually meeting
    sites: list[str]

    @property
    def sigma_m(self) -> float:
        """One-sigma position error, worst axis. The number to quote."""
        return float(math.sqrt(max(np.linalg.eigvalsh(self.cov).max(), 0.0)))

    @property
    def ellipse_axes_m(self) -> tuple[float, float, float]:
        """One-sigma semi-axes, ascending. The error is an ellipsoid, not a ball."""
        vals = np.linalg.eigvalsh(self.cov)
        return tuple(float(math.sqrt(max(v, 0.0))) for v in vals)  # type: ignore[return-value]


def max_separation_deg(rays: list[Ray]) -> float:
    """Widest angle between any pair of rays. Near zero means no usable geometry."""
    best = 0.0
    for i, a in enumerate(rays):
        for b in rays[i + 1:]:
            cosang = float(np.clip(np.dot(a.direction, b.direction), -1.0, 1.0))
            best = max(best, math.degrees(math.acos(cosang)))
    return best


def triangulate(
    rays: list[Ray],
    miss_gate_sigmas: float = DEFAULT_MISS_GATE_SIGMAS,
    min_separation_deg: float = MIN_SEPARATION_DEG,
) -> Fix | None:
    """Least-squares closest point to a set of rays, with a covariance.

    Each ray constrains the target to lie on a line. The point minimising the
    summed squared perpendicular distance solves `A p = b`, where
    `A = sum(I - d d^T)` projects out the along-ray direction -- exactly the
    direction a bearing cannot measure.

    Weighting is `1 / (sigma^2 * r^2)`: a bearing's *positional* uncertainty
    grows with range, so a distant site should pull the answer less than a near
    one. Returns None rather than a confident fiction when the geometry is
    degenerate or the rays do not nearly meet.
    """
    if len(rays) < 2:
        return None
    if max_separation_deg(rays) < min_separation_deg:
        return None

    # Unweighted pass first: we need a range estimate before we can weight by it.
    p = _solve(rays, weights=None)
    if p is None:
        return None
    ranges = [max(float(np.linalg.norm(p - r.origin)), 1.0) for r in rays]

    weights = [1.0 / ((math.radians(r.sigma_deg) * rng) ** 2)
               for r, rng in zip(rays, ranges)]
    p = _solve(rays, weights=weights)
    if p is None:
        return None

    # Covariance of a weighted least-squares solution is the inverse of the
    # information matrix. Rank-deficient geometry was already rejected above.
    #
    # This treats each ray's perpendicular plane as isotropic, which slightly
    # over-states the error: an azimuth wobble of d_az displaces the target by
    # r*cos(el)*d_az, not r*d_az. Measured, the reported sigma is high by about
    # 1/cos(el) -- 0.5% at 0 deg elevation, 8% at 24 deg, 2x at 67 deg. Left as
    # is on purpose. It errs conservative, which is the safe direction for a
    # number that gates an effector, and the targets this exists to track sit
    # under 10 deg elevation (80 m up at 500 m out is 9 deg), where it is
    # under 2%. Correcting it would mean an anisotropic per-ray term for no
    # operational gain.
    info = np.zeros((3, 3))
    for ray, w in zip(rays, weights):
        d = ray.direction
        info += w * (np.eye(3) - np.outer(d, d))
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None

    # Scale the association gate to the miss that this geometry's own bearing
    # noise would produce anyway. See MIN_SEPARATION_DEG's neighbour above.
    expected = math.sqrt(sum((math.radians(r.sigma_deg) * rng) ** 2
                             for r, rng in zip(rays, ranges)) / len(rays))
    gate = max(miss_gate_sigmas * expected, MIN_MISS_GATE_M)

    miss = _miss_distance(rays, p)
    if miss > gate:
        return None

    return Fix(enu=p, cov=cov, miss_m=miss, sites=[r.src for r in rays])


def _solve(rays: list[Ray], weights: list[float] | None) -> np.ndarray | None:
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for i, ray in enumerate(rays):
        d = ray.direction
        proj = np.eye(3) - np.outer(d, d)
        w = 1.0 if weights is None else weights[i]
        a += w * proj
        b += w * (proj @ ray.origin)
    if np.linalg.cond(a) > 1e12:
        return None
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None


def _miss_distance(rays: list[Ray], p: np.ndarray) -> float:
    """Largest perpendicular distance from the solution to any ray.

    Zero when the rays truly intersect, and a large value does mean the sites
    are not looking at the same object. The converse is what does not hold, and
    it is worth being blunt about: **a small miss is not proof of a correct
    association.**

    Two rays that cross at all produce a miss of zero, wherever they cross. Only
    the component of a disagreement that takes one ray *out of the plane of the
    other* shows up here. Measured on the two-post reference geometry, putting
    site B 20 deg off in elevation gives a 22 m miss for a 23 m position error --
    caught. The same 20 deg off in *azimuth* gives a 4 m miss for a 75 m position
    error -- passed. Azimuth slides the crossing point along; it does not open
    the rays apart.

    So this gate catches out-of-plane mis-association and altitude disagreement,
    and it is the wrong tool for the in-plane case. A third site restores the
    check (the same azimuth error then misses by 23 m). With two sites, the
    defence is temporal, upstream in the track store: a mis-associated pair
    traces a trajectory that does not hold together frame to frame.
    """
    worst = 0.0
    for ray in rays:
        v = p - ray.origin
        along = float(np.dot(v, ray.direction))
        worst = max(worst, float(np.linalg.norm(v - along * ray.direction)))
    return worst
