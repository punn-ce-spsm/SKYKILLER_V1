"""Measure a run against truth: the numbers a procurement document needs.

Three questions, and the answers have to be measured because every one of them
is a claim the customer will test:

  - **How much warning?** When was the aircraft first held, at what range, and
    how long from there to the effector's envelope edge.
  - **How long inside?** Dwell inside the envelope is the window an operator
    actually has, and it is shorter than people expect.
  - **How accurate?** Error against known truth, as a distribution rather than
    a best case, because the 95th percentile is what a beam has to cover.

It also counts what went wrong: tracks that correspond to no aircraft, and any
moment one of our own was called hostile. A report that only carried the good
numbers would be worth nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .air_picture import AirPicture, FriendlyFeed
from .effector import Effector, RoeGate
from .schemas import IFF_HOSTILE, Track
from .scenario import Scenario, run

#: How close a track must be to an aircraft to be counted as *that* aircraft
#: rather than a ghost. Generous on purpose: a track 300 m from anything is not
#: a mislocated aircraft, it is an invented one.
GHOST_RADIUS_M = 300.0


@dataclass(slots=True)
class Report:
    target: str
    first_seen_t: float | None = None
    first_seen_range_m: float | None = None
    declared_t: float | None = None
    declared_range_m: float | None = None
    envelope_entry_t: float | None = None
    envelope_exit_t: float | None = None
    errors_m: list[float] = field(default_factory=list)
    #: Error at the moment the target crossed the envelope edge -- the only
    #: accuracy figure an effector cares about. The p50 over a whole run is a
    #: worse comparison than it looks: a system that holds the target from 1.5 km
    #: spends most of its frames at long range where a bearings-only fix is
    #: honestly poor, so it reports a *larger* median error than a system that
    #: only ever sees the target at 500 m -- while being strictly better.
    error_at_envelope_m: float | None = None
    last_t: float | None = None
    still_inside: bool = False
    prompts: int = 0
    peak_tracks: int = 0
    #: Frames in which some track sat further than GHOST_RADIUS_M from every
    #: aircraft. Console clutter, and worth watching, but not by itself a
    #: failure -- an unconfirmed track that lives half a second and stays
    #: UNKNOWN is track-before-declare working, not failing.
    ghost_frames: int = 0
    #: Frames in which such a track was declared HOSTILE. This one *is* a
    #: failure: it is an effector pointed at something that is not there.
    ghost_hostile_frames: int = 0
    friendly_ever_hostile: bool = False

    @property
    def warning_s(self) -> float | None:
        """Seconds between first holding the target and it reaching the envelope.

        The product's headline number. It is not the same as detection range
        divided by speed: the target has to be *held*, and a track that is
        dropped and reacquired has not been held.
        """
        if self.first_seen_t is None or self.envelope_entry_t is None:
            return None
        return self.envelope_entry_t - self.first_seen_t

    @property
    def dwell_s(self) -> float | None:
        """Seconds the target spends inside the envelope. The decision window.

        A target that is still inside when the run ends has its dwell measured
        to the end of the run and `still_inside` set -- that is a lower bound,
        and reporting it as unknown would throw away the useful half of the
        answer while reporting it as final would overstate nothing but imply
        the engagement ended.
        """
        if self.envelope_entry_t is None:
            return None
        end = self.envelope_exit_t if self.envelope_exit_t is not None else self.last_t
        return None if end is None else end - self.envelope_entry_t

    @property
    def error_p50(self) -> float | None:
        return float(np.median(self.errors_m)) if self.errors_m else None

    @property
    def error_p95(self) -> float | None:
        return float(np.percentile(self.errors_m, 95)) if self.errors_m else None

    def lines(self) -> list[str]:
        def fmt(v, unit="", nd=1):
            return "--" if v is None else f"{v:.{nd}f}{unit}"
        return [
            f"target                {self.target}",
            f"first held            {fmt(self.first_seen_t, ' s')}"
            f" at {fmt(self.first_seen_range_m, ' m', 0)}",
            f"declared hostile      {fmt(self.declared_t, ' s')}"
            f" at {fmt(self.declared_range_m, ' m', 0)}",
            f"warning before envelope {fmt(self.warning_s, ' s')}",
            f"dwell inside envelope {fmt(self.dwell_s, ' s')}"
            + ("  (still inside at run end)" if self.still_inside else ""),
            f"error at envelope edge {fmt(self.error_at_envelope_m, ' m')}",
            f"position error, whole run  p50 {fmt(self.error_p50, ' m')}"
            f"  p95 {fmt(self.error_p95, ' m')}",
            f"ARM prompts           {self.prompts}",
            f"peak tracks           {self.peak_tracks}",
            f"frames with a ghost   {self.ghost_frames}"
            f"  (declared hostile: {self.ghost_hostile_frames})",
            f"own aircraft called hostile  {self.friendly_ever_hostile}",
        ]


def measure(scenario: Scenario, target: str, effector: Effector,
            air_picture: AirPicture | None = None) -> Report:
    """Run a scenario and report what the system actually achieved."""
    gate = RoeGate(effector=effector)
    ap = air_picture or AirPicture(feed=FriendlyFeed())
    report = Report(target=target)
    friendly_ids = {c.id for c in scenario.aircraft if c.friendly}

    for frame in run(scenario, gate=gate, air_picture=ap):
        truth = scenario.truth(frame.t)
        report.prompts += len(frame.prompts)
        report.peak_tracks = max(report.peak_tracks, len(frame.tracks))

        held = _nearest_track(frame.tracks, truth[target])
        if held is not None:
            error = float(np.linalg.norm(np.array(held.enu) - truth[target]))
            report.errors_m.append(error)
            range_m = effector.aim(truth[target])[2]
            if report.first_seen_t is None:
                report.first_seen_t = frame.t
                report.first_seen_range_m = range_m
            if held.iff == IFF_HOSTILE and report.declared_t is None:
                report.declared_t = frame.t
                report.declared_range_m = range_m

        # Envelope timings come from *truth*, not from the track. They are a
        # property of the engagement geometry; measuring them off our own
        # estimate would let a bad fix flatter the warning figure.
        report.last_t = frame.t
        inside = effector.in_envelope(truth[target])
        if inside and report.envelope_entry_t is None:
            report.envelope_entry_t = frame.t
            if held is not None:
                report.error_at_envelope_m = float(
                    np.linalg.norm(np.array(held.enu) - truth[target]))
        if not inside and report.envelope_entry_t is not None \
                and report.envelope_exit_t is None and frame.t > report.envelope_entry_t:
            report.envelope_exit_t = frame.t

        ghosts = [t for t in frame.tracks
                  if _nearest_truth_distance(t, truth) > GHOST_RADIUS_M]
        if ghosts:
            report.ghost_frames += 1
        if any(t.iff == IFF_HOSTILE for t in ghosts):
            report.ghost_hostile_frames += 1
        for track in frame.tracks:
            if track.iff != IFF_HOSTILE:
                continue
            who = _nearest_truth_name(track, truth)
            if who in friendly_ids and _nearest_truth_distance(track, truth) < GHOST_RADIUS_M:
                report.friendly_ever_hostile = True

    report.still_inside = (report.envelope_entry_t is not None
                           and report.envelope_exit_t is None)
    return report


def _nearest_track(tracks: list[Track], position) -> Track | None:
    best, best_d = None, GHOST_RADIUS_M
    for track in tracks:
        d = float(np.linalg.norm(np.array(track.enu) - position))
        if d < best_d:
            best, best_d = track, d
    return best


def _nearest_truth_distance(track: Track, truth: dict) -> float:
    return min(float(np.linalg.norm(np.array(track.enu) - p)) for p in truth.values())


def _nearest_truth_name(track: Track, truth: dict) -> str:
    return min(truth, key=lambda k: float(np.linalg.norm(np.array(track.enu) - truth[k])))
