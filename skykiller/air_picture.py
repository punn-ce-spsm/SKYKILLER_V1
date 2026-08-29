"""The fused air picture, and the question the whole system exists to answer:
whose drone is that?

The stated rule going in was "if the frequency is not ours, it is theirs." That
does not hold. Both sides fly DJI-class aircraft on the same 2.4 and 5.8 GHz
bands, so frequency separates nothing, and neither does size, shape or flight
profile. Anything an adversary can also buy is not an identity check.

What does hold is the asymmetry of *cooperation*: our aircraft report their own
positions and theirs do not. So identity is decided by correlation against a
feed only our side produces -- Remote ID, telemetry, or the C2 air picture. This
is Remote ID's weakness as a threat sensor (a hostile simply turns it off) read
the other way round, as its strength as a friend sensor, and it needs no RF
receiver at all.

Three verdicts, and the gaps between them are the safety property:

    FRIENDLY   correlated, right now, with a live feed entry.
    UNKNOWN    we cannot currently tell. The default, and where doubt goes.
    HOSTILE    a healthy feed says nothing is there, and has said so for a while.

Two rules make that safe, and both are about what *cannot* happen:

1.  **HOSTILE is only assertable when the feed is known healthy.** If the feed
    dies, every track stops correlating -- which, read naively, declares the
    entire sky hostile at the exact moment we have lost the ability to tell.
    A dead feed means UNKNOWN for everything, including tracks that were
    FRIENDLY a second ago.

2.  **No track goes FRIENDLY -> HOSTILE directly.** It passes through UNKNOWN
    and has to stay uncorrelated for `hostile_hold_s` first. One dropped packet
    should not re-label our own aircraft as a target.

The known weakness, stated plainly: a hostile flying inside the correlation gate
of a friendly is called FRIENDLY. That is inherent to position-correlation IFF
and no amount of code removes it. The gate is sized from the covariance rather
than fixed, so a good fix gives a tight gate and a small window; a bad fix
honestly gives a wide one. Closing it further needs a second discriminant --
Remote ID serials, which lane L1b provides -- not a smaller number here.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .schemas import IFF_FRIENDLY, IFF_HOSTILE, IFF_UNKNOWN, Friendly, Track
from .triangulate import Fix

#: How long a feed may go silent before it is presumed dead rather than empty.
#: This is why the feed must send an explicit "nothing airborne" heartbeat: an
#: empty report and a broken link look identical otherwise, and they mean
#: opposite things -- one licenses HOSTILE, the other forbids it.
DEFAULT_FEED_MAX_AGE_S = 5.0

#: Chi-squared 99% point on 3 degrees of freedom. A track inside this Mahalanobis
#: distance of a friendly is the same aircraft 99 times in 100.
GATE_CHI2_3DOF = 11.345

#: How long a track must stay uncorrelated, under a healthy feed, before it is
#: called hostile. Covers a late Remote ID beacon and a dropped report.
DEFAULT_HOSTILE_HOLD_S = 3.0

#: Drop a track that has not been updated in this long.
DEFAULT_TRACK_MAX_AGE_S = 5.0

#: Confident frames a track needs before it may be called HOSTILE.
#:
#: This is track-before-declare, and it exists because the alternative was
#: measured and was bad. Running two aircraft past two posts, the cross-site
#: pairing is ambiguous on most frames (the targets fly near mast height, which
#: is the degenerate geometry -- see associate.py). Every wrongly paired frame
#: drops a ghost at a fresh position; those spawned tracks, the tracks served
#: the hostile hold, and a two-aircraft sky produced five to eight tracks with
#: three ghosts, most of them labelled HOSTILE. That is a jammer pointed at
#: things that are not there.
#:
#: A ghost cannot survive this gate because it is not repeatable: it jumps to a
#: different place each frame, so it never accumulates confident hits in one
#: spot. A real aircraft does.
#:
#: Three, and not more. Raising it looks like free safety and is not: measured
#: over eight runs of the two-aircraft scenario, going from 3 to 12 halved the
#: surviving ghosts (8/8 runs to 3/8) but pushed the moment a real hostile is
#: declared from 3 s out to 20-40 s. At 15 m/s that is 300 m of standoff traded
#: for a partial fix. The residual ghosts are a *geometry* problem -- see
#: `SiteNetwork.elevation_separation_deg` -- and paying for them in warning
#: time buys the wrong thing.
MIN_CONFIRM_HITS = 3

#: Fastest a target is assumed to move, m/s, used to size the association gate
#: for a track whose velocity is not yet known. Covers an FPV racer with room
#: to spare; a Mini 4 Pro tops out near 16.
MAX_SPEED_MS = 50.0

#: Manoeuvre allowance once velocity is known *well*, m/s^2 -- roughly 2 g,
#: which a quadcopter can pull. This is what the prediction is wrong by once
#: the heading is trustworthy, and it is far tighter than MAX_SPEED_MS.
MAX_ACCEL_MS2 = 20.0

#: Two tracks closer than this in Mahalanobis distance are the same object and
#: are merged. Same chi-squared 99% point the association gate uses, and for the
#: same reason: if a *measurement* this close would be judged the same object,
#: two track estimates this close are too.
#:
#: Without merging, one aircraft acquires a pair of tracks that straddle it a
#: dozen metres apart, each nearest to the fix on alternate frames. Both stay
#: fed, so neither ages out, and the pair survives indefinitely. Measured on the
#: two-aircraft scenario: every aircraft ended the run with exactly two tracks,
#: born about 25 s apart -- and a duplicate of the *hostile* raised its own
#: second ARM prompt, while a duplicate of the *friendly* spent its first
#: seconds uncorrelated and was briefly declared HOSTILE. Both are our own
#: bookkeeping presented to an operator as a second aircraft.
MERGE_CHI2 = GATE_CHI2_3DOF

#: Process noise for the constant-velocity filter, m/s^2. How hard the target
#: is assumed to be able to manoeuvre between updates. Larger makes the filter
#: chase the measurements (and stop smoothing); smaller makes it lag a real
#: turn. 5 m/s^2 is brisk for a quadcopter under normal flight.
PROCESS_ACCEL_MS2 = 5.0


def mahalanobis_sq(a: np.ndarray, b: np.ndarray, cov: np.ndarray) -> float:
    """Squared separation in units of the combined uncertainty.

    Plain metres cannot answer "is this the same aircraft": 30 m is a decisive
    mismatch for a 2 m fix and well inside the noise for a 200 m one. This
    divides by the error, so one threshold works at every range.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    try:
        return float(d @ np.linalg.solve(cov, d))
    except np.linalg.LinAlgError:
        return float("inf")   # singular covariance: refuse to associate


@dataclass(slots=True)
class FriendlyFeed:
    """Current cooperative reports, and whether the feed is alive at all.

    `update` must be called even when nothing is airborne. Silence is not the
    same message as an empty list, and this class exists mostly to keep those
    two apart.
    """

    max_age_s: float = DEFAULT_FEED_MAX_AGE_S
    _entries: dict[str, Friendly] = field(default_factory=dict)
    _last_update: float | None = None

    def update(self, friendlies: list[Friendly], t_utc: float | None = None) -> None:
        """Replace the picture. An empty list is a valid, meaningful report."""
        self._entries = {f.id: f for f in friendlies}
        self._last_update = time.time() if t_utc is None else t_utc

    def healthy(self, now: float) -> bool:
        return (self._last_update is not None
                and now - self._last_update <= self.max_age_s)

    @property
    def entries(self) -> list[Friendly]:
        return list(self._entries.values())

    def match(self, enu: np.ndarray, cov: np.ndarray,
              now: float | None = None) -> tuple[Friendly | None, float]:
        """Closest friendly within the gate, by Mahalanobis distance.

        Three things widen the gate, and all three are the honest width:

        - the track's own position error,
        - the friendly's error in its own position (`sigma_m`), and
        - **how old the friendly's report is.**

        The third was missing and it mislabelled our own aircraft. A report is
        a statement about where something *was*; a drone orbiting at 21 m/s is
        105 m from a five-second-old report, which is far outside a gate built
        from metres of position error. Measured on the two-aircraft scenario,
        a feed that merely went quiet -- still inside its freshness window, not
        yet declared dead -- pushed the friendly out of correlation, through the
        hostile hold, and had our own aircraft declared HOSTILE at 23 s.

        So an aged report is inflated by how far the aircraft could have flown
        since. That errs toward FRIENDLY and UNKNOWN, which is the safe
        direction: the cost of a gate too wide is failing to notice a hostile
        shadowing a friendly, and the cost of one too narrow is shooting at our
        own aircraft.
        """
        now = time.time() if now is None else now
        best: Friendly | None = None
        best_d2 = float("inf")
        for f in self._entries.values():
            age = max(now - f.t_utc, 0.0)
            drift = MAX_SPEED_MS * age
            combined = (np.asarray(cov, dtype=float)
                        + np.eye(3) * (f.sigma_m ** 2 + drift ** 2))
            d2 = mahalanobis_sq(np.array(f.enu), enu, combined)
            if d2 < best_d2:
                best, best_d2 = f, d2
        if best is None or best_d2 > GATE_CHI2_3DOF:
            return None, best_d2
        return best, best_d2


@dataclass(slots=True)
class _TrackState:
    """A track, its filter, and the bookkeeping the wire contract does not carry.

    `x` and `P` are a six-state constant-velocity filter: position and velocity
    in ENU, with a 6x6 covariance. The filter is not decoration and replaced a
    scheme that stored the raw last fix as the track position. That scheme had a
    defect visible only in an end-to-end run: the distance from a track to the
    next fix was then the difference of *two* noisy measurements, so a 99%
    association gate spawned a spurious duplicate roughly 1% of the time. Over
    an 80 s run at 5 Hz that is several duplicates per aircraft -- and a
    duplicate of our own aircraft starts uncorrelated, serves the hostile hold,
    and is declared a target.

    A filtered state is the fix: it averages the noise down, so the innovation
    is measurement noise alone rather than the difference of two draws of it.
    """

    track: Track
    x: np.ndarray                             # [e, n, u, ve, vn, vu]
    P: np.ndarray                             # 6x6 covariance
    uncorrelated_since: float | None = None   # when the hostile hold started
    confident_hits: int = 0                   # frames whose pairing was unambiguous

    @property
    def confirmed(self) -> bool:
        return self.confident_hits >= MIN_CONFIRM_HITS


@dataclass(slots=True)
class AirPicture:
    """Track store and IFF classifier.

    Association is nearest-in-Mahalanobis to the track's *predicted* position,
    same gate shape as the friendly correlation. Two reasons it has to predict
    rather than compare against the last fix:

    - A target closing at 10 m/s moves 10 m between frames at 1 Hz. Against a
      2 m fix that is a 5-sigma mismatch, so a position-only gate would split
      one aircraft into a new track every frame -- and would do it *worse* the
      better the fix got, which is exactly backwards.
    - A coarse fix at 2 km wanders 80 m frame to frame from honest bearing
      noise alone, and must not spawn a new track either.

    Both are handled by the same thing: gate on where the track should be, with
    the covariance inflated by how far it could have strayed since.
    """

    feed: FriendlyFeed = field(default_factory=FriendlyFeed)
    hostile_hold_s: float = DEFAULT_HOSTILE_HOLD_S
    track_max_age_s: float = DEFAULT_TRACK_MAX_AGE_S
    _states: dict[str, _TrackState] = field(default_factory=dict)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    @property
    def tracks(self) -> list[Track]:
        return [s.track for s in self._states.values()]

    def get(self, track_id: str) -> Track | None:
        state = self._states.get(track_id)
        return state.track if state else None

    def ingest(self, fix: Fix, now: float | None = None, score: float = 0.0,
               confident: bool = True) -> Track | None:
        """Fold one fused position into the picture and return its track.

        `confident` is `not Association.ambiguous` -- whether the cross-site
        pairing that produced this fix was decided by geometry or by noise. An
        ambiguous fix may *update* an existing track, because a track carries a
        prediction and that prediction is itself strong evidence about which
        pairing was right. It may not *create* one, and this returns None when
        it would have: a fix that is both unexplained by anything already
        tracked and unresolvable by geometry is not a target, it is a maybe.
        """
        now = time.time() if now is None else now
        state = self._associate(fix, now)
        if state is None and not confident:
            return None
        if state is None:
            state = self._birth(fix, now, score)
        else:
            self._filter_update(state, fix, now, score, confident)

        self._classify(state, now)
        return state.track

    def _birth(self, fix: Fix, now: float, score: float) -> _TrackState:
        """Start a track from one fix. Velocity is unknown, and says so.

        The velocity block of P is MAX_SPEED_MS squared -- not zero. A new track
        that claimed to know it was stationary would gate as though it did, and
        reject the very next frame of a moving target.
        """
        x = np.concatenate([fix.enu, np.zeros(3)])
        P = np.zeros((6, 6))
        P[:3, :3] = fix.cov
        P[3:, 3:] = np.eye(3) * MAX_SPEED_MS ** 2
        track = Track(
            id=f"K-{next(self._ids):03d}", enu=tuple(fix.enu),
            cov=fix.cov.tolist(), sites=list(fix.sites), score=score,
            first_seen=now, last_seen=now,
        )
        state = _TrackState(track=track, x=x, P=P, confident_hits=1)
        self._states[track.id] = state
        return state

    @staticmethod
    def _predict(state: _TrackState, now: float) -> tuple[np.ndarray, np.ndarray]:
        """Constant-velocity propagation with a manoeuvre allowance."""
        dt = max(now - state.track.last_seen, 0.0)
        f = np.eye(6)
        f[:3, 3:] = np.eye(3) * dt
        q = PROCESS_ACCEL_MS2 ** 2
        noise = np.zeros((6, 6))
        noise[:3, :3] = np.eye(3) * (q * dt ** 3 / 3.0)
        noise[:3, 3:] = noise[3:, :3] = np.eye(3) * (q * dt ** 2 / 2.0)
        noise[3:, 3:] = np.eye(3) * (q * dt)
        return f @ state.x, f @ state.P @ f.T + noise

    def _filter_update(self, state: _TrackState, fix: Fix, now: float,
                       score: float, confident: bool) -> None:
        """Standard Kalman update on a direct position measurement."""
        x_pred, p_pred = self._predict(state, now)
        h = np.zeros((3, 6))
        h[:, :3] = np.eye(3)
        s_mat = h @ p_pred @ h.T + fix.cov
        try:
            gain = p_pred @ h.T @ np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            return                                   # unusable measurement
        state.x = x_pred + gain @ (fix.enu - h @ x_pred)
        state.P = (np.eye(6) - gain @ h) @ p_pred

        t = state.track
        t.enu = tuple(state.x[:3])
        t.cov = state.P[:3, :3].tolist()
        t.vel = tuple(state.x[3:])
        t.sites = list(fix.sites)
        t.score = score
        t.last_seen = now
        if confident:
            state.confident_hits += 1

    def prune(self, now: float | None = None) -> list[str]:
        """Drop stale tracks and merge duplicates. Returns the ids removed."""
        now = time.time() if now is None else now
        stale = [tid for tid, s in self._states.items()
                 if now - s.track.last_seen > self.track_max_age_s]
        for tid in stale:
            del self._states[tid]
        return stale + self._merge(now)

    def _merge(self, now: float) -> list[str]:
        """Fold together tracks that are statistically the same object.

        The survivor is the better-established of the pair, and it inherits the
        other's confident hits and the earlier `first_seen` -- so a merge never
        loses the moment the aircraft was actually first seen, which is the
        number the early-warning claim rests on.

        The states are not fused. Two tracks of one aircraft are built from
        overlapping measurements and are anything but independent, so combining
        their covariances would manufacture a confidence neither has earned.
        Keeping the stronger estimate is the conservative choice.
        """
        removed = []
        states = sorted(self._states.values(),
                        key=lambda s: (s.confident_hits, -s.track.first_seen),
                        reverse=True)
        for i, keep in enumerate(states):
            if keep.track.id in removed:
                continue
            for drop in states[i + 1:]:
                if drop.track.id in removed:
                    continue
                d2 = mahalanobis_sq(np.array(keep.track.enu), np.array(drop.track.enu),
                                    np.array(keep.track.cov) + np.array(drop.track.cov))
                if d2 > MERGE_CHI2:
                    continue
                keep.confident_hits += drop.confident_hits
                keep.track.first_seen = min(keep.track.first_seen,
                                            drop.track.first_seen)
                keep.track.last_seen = max(keep.track.last_seen, drop.track.last_seen)
                del self._states[drop.track.id]
                removed.append(drop.track.id)
        if removed:
            for state in self._states.values():
                self._classify(state, now)
        return removed

    def reclassify(self, now: float | None = None) -> None:
        """Re-run IFF on every track without new sensor data.

        Needed because a verdict can go stale on its own: the feed dying makes
        every FRIENDLY wrong, and no detection has to arrive for that to happen.
        A console that only re-classified on new fixes would keep displaying
        FRIENDLY over a dead feed, which is the one thing this must not do.
        """
        now = time.time() if now is None else now
        for state in self._states.values():
            self._classify(state, now)

    # --- internals ----------------------------------------------------------

    def _associate(self, fix: Fix, now: float) -> _TrackState | None:
        """Nearest track by innovation distance, gated at chi-squared 99%.

        The comparison is against the filter's *predicted* position with the
        filter's own predicted covariance, so the quantity being gated is the
        measurement noise -- not, as it was before the filter existed, the
        difference between two independently noisy measurements.
        """
        best, best_d2 = None, float("inf")
        for state in self._states.values():
            if now - state.track.last_seen > self.track_max_age_s:
                continue
            x_pred, p_pred = self._predict(state, now)
            d2 = mahalanobis_sq(x_pred[:3], fix.enu, p_pred[:3, :3] + fix.cov)
            if d2 < best_d2:
                best, best_d2 = state, d2
        return best if best_d2 <= GATE_CHI2_3DOF else None

    def _classify(self, state: _TrackState, now: float) -> None:
        t = state.track

        # Rule 1: without a healthy feed we cannot tell anything about anyone.
        # Not "assume the last answer still holds" -- the last answer is exactly
        # what has just become unsupported.
        if not self.feed.healthy(now):
            t.iff, t.friendly_id = IFF_UNKNOWN, None
            state.uncorrelated_since = None
            return

        match, _ = self.feed.match(np.array(t.enu), np.array(t.cov), now)
        if match is not None:
            t.iff, t.friendly_id = IFF_FRIENDLY, match.id
            state.uncorrelated_since = None
            return

        # Uncorrelated under a healthy feed. Rule 2: serve the hold in UNKNOWN
        # before asserting HOSTILE, whatever the previous verdict was. Rule 3:
        # and be a confirmed track, so a ghost that appears once cannot be
        # declared a target. FRIENDLY needs no confirmation -- mislabelling a
        # ghost as friendly costs nothing, since nothing is fired at friendlies.
        t.friendly_id = None
        if state.uncorrelated_since is None:
            state.uncorrelated_since = now
        held = now - state.uncorrelated_since
        t.iff = (IFF_HOSTILE if held >= self.hostile_hold_s and state.confirmed
                 else IFF_UNKNOWN)
