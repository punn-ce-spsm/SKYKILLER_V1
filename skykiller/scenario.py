"""Synthetic scenarios that drive the *real* pipeline.

Nothing here fakes a result. It renders what two surveyed posts would see of
aircraft on known trajectories, adds bearing noise, and hands the resulting
`Detection`s to the same `associate` / `AirPicture` / `RoeGate` code the field
system runs. So accuracy against truth is measured rather than asserted, and
every number in MEMORY.md came out of this rather than out of an estimate.

It exists because the interesting failures are not in any one module. Both of
the serious defects found in build 2 -- unpaired cross-site detections, and
ghosts surviving into HOSTILE -- were invisible in unit tests and obvious the
first time a second aircraft was put in the sky.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .air_picture import AirPicture, FriendlyFeed
from .associate import associate
from .effector import Effector, RoeGate
from .masking import Obstacle, visible
from .schemas import Detection, Friendly, Track
from .sites import SiteNetwork
from .triangulate import to_bearing


@dataclass(slots=True)
class Aircraft:
    """One aircraft on a known path. `friendly` decides only whether it appears
    on the cooperative feed -- it changes nothing about how it is *sensed*,
    which is the point: the sensors cannot tell the sides apart."""

    id: str
    path: Callable[[float], np.ndarray]
    friendly: bool = False
    sigma_m: float = 5.0            # how well a friendly knows its own position

    def at(self, t: float) -> np.ndarray:
        return np.asarray(self.path(t), dtype=float)


@dataclass(slots=True)
class Scenario:
    net: SiteNetwork
    aircraft: list[Aircraft]
    hz: float = 5.0
    duration_s: float = 40.0
    sigma_deg: float = 0.5
    seed: int = 0
    #: Terrain between the posts and the target. A post with no line of sight
    #: produces no detection -- which is the whole customer problem, so it has
    #: to be modelled here rather than assumed away.
    obstacles: list[Obstacle] = field(default_factory=list)
    feed_alive: Callable[[float], bool] = lambda t: True
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def times(self) -> list[float]:
        return [n / self.hz for n in range(int(self.hz * self.duration_s))]

    def detections(self, t: float) -> list[Detection]:
        """What every post sees this instant, with bearing noise applied.

        A post that has no line of sight to an aircraft reports nothing about
        it, which is how the altitude argument gets *demonstrated* rather than
        claimed: run the same trajectory past a low post and an elevated one
        and the low one simply has no detections to contribute.

        Range and detector limits are not modelled. Every aircraft in clear
        view is seen, so the sensing assumption is optimistic and the
        association load is pessimistic -- more contacts in frame is the harder
        case for the pairing.
        """
        out = []
        for name, site in self.net.sites.items():
            for craft in self.aircraft:
                if not visible(site.enu, craft.at(t), self.obstacles):
                    continue
                az, el = to_bearing(craft.at(t) - site.enu)
                out.append(Detection(
                    src=name,
                    az=az + self._rng.normal(0, self.sigma_deg),
                    el=el + self._rng.normal(0, self.sigma_deg),
                    conf=0.9, t_utc=t, raw_id=craft.id))
        return out

    def friendlies(self, t: float) -> list[Friendly]:
        """The cooperative feed. Only friendly aircraft report."""
        return [Friendly(id=c.id, enu=tuple(c.at(t)), source="telemetry",
                         sigma_m=c.sigma_m, t_utc=t)
                for c in self.aircraft if c.friendly]

    def truth(self, t: float) -> dict[str, np.ndarray]:
        return {c.id: c.at(t) for c in self.aircraft}

    def nearest_truth(self, enu, t: float) -> tuple[str | None, float]:
        """Which aircraft a reported position is closest to, and how far off."""
        best, best_d = None, float("inf")
        for name, pos in self.truth(t).items():
            d = float(np.linalg.norm(np.asarray(enu, dtype=float) - pos))
            if d < best_d:
                best, best_d = name, d
        return best, best_d


@dataclass(slots=True)
class Frame:
    """One cycle's worth of everything, for a harness to measure against."""

    t: float
    tracks: list[Track]
    prompts: list[Track]
    ambiguous: bool
    n_fixes: int


def run(scenario: Scenario, gate: RoeGate | None = None,
        air_picture: AirPicture | None = None):
    """Drive the real pipeline over a scenario, yielding a Frame per cycle."""
    ap = air_picture or AirPicture(feed=FriendlyFeed())
    for t in scenario.times():
        if scenario.feed_alive(t):
            ap.feed.update(scenario.friendlies(t), t_utc=t)
        got = associate(scenario.net, scenario.detections(t))
        for fix in got.fixes:
            ap.ingest(fix, now=t, score=0.9, confident=not got.ambiguous)
        ap.reclassify(now=t)
        ap.prune(now=t)
        prompts = gate.update(ap.tracks, now=t) if gate else []
        yield Frame(t=t, tracks=list(ap.tracks), prompts=prompts,
                    ambiguous=got.ambiguous, n_fixes=len(got.fixes))


def straight_line(start, velocity) -> Callable[[float], np.ndarray]:
    start, velocity = np.asarray(start, float), np.asarray(velocity, float)
    return lambda t: start + velocity * t


def orbit(centre, radius: float, altitude: float, period_s: float = 75.0):
    centre = np.asarray(centre, float)
    def path(t: float) -> np.ndarray:
        a = 2 * np.pi * t / period_s
        return np.array([centre[0] + radius * np.cos(a),
                         centre[1] + radius * np.sin(a), altitude])
    return path
