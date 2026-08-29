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

#: Smoothing on the velocity estimate. A two-point difference divides position
#: noise by dt, so at 5 Hz a 100 m fix produces a 700 m/s phantom velocity --
#: and an unsmoothed track then flies its prediction straight out of its own
#: gate. Averaging trades lag for a velocity that is worth predicting with.
VEL_SMOOTHING = 0.3


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

    def match(self, enu: np.ndarray, cov: np.ndarray) -> tuple[Friendly | None, float]:
        """Closest friendly within the gate, by Mahalanobis distance.

        The friendly's own position error is added to the track's, so a friendly
        that is vague about where it is gets a correspondingly generous gate.
        """
        best: Friendly | None = None
        best_d2 = float("inf")
        for f in self._entries.values():
            combined = np.asarray(cov, dtype=float) + np.eye(3) * f.sigma_m ** 2
            d2 = mahalanobis_sq(np.array(f.enu), enu, combined)
            if d2 < best_d2:
                best, best_d2 = f, d2
        if best is None or best_d2 > GATE_CHI2_3DOF:
            return None, best_d2
        return best, best_d2


@dataclass(slots=True)
class _TrackState:
    """A track plus the bookkeeping IFF and association need but the wire does not."""

    track: Track
    uncorrelated_since: float | None = None   # when the hostile hold started
    vel_sigma: float = float("inf")           # how much the velocity is worth
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
            track = Track(
                id=f"K-{next(self._ids):03d}", enu=tuple(fix.enu),
                cov=fix.cov.tolist(), sites=list(fix.sites), score=score,
                first_seen=now, last_seen=now,
            )
            state = _TrackState(track=track, confident_hits=1)
            self._states[track.id] = state
        else:
            t = state.track
            dt = now - t.last_seen
            if dt > 0:
                self._update_velocity(state, fix, dt)
            t.enu = tuple(fix.enu)
            t.cov = fix.cov.tolist()
            t.sites = list(fix.sites)
            t.score = score
            t.last_seen = now
            if confident:
                state.confident_hits += 1

        self._classify(state, now)
        return state.track

    def prune(self, now: float | None = None) -> list[str]:
        """Drop tracks nothing has confirmed lately. Returns the ids dropped."""
        now = time.time() if now is None else now
        stale = [tid for tid, s in self._states.items()
                 if now - s.track.last_seen > self.track_max_age_s]
        for tid in stale:
            del self._states[tid]
        return stale

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
        best, best_d2 = None, float("inf")
        for state in self._states.values():
            if now - state.track.last_seen > self.track_max_age_s:
                continue
            predicted, spread = self._predict(state, now)
            combined = (np.asarray(state.track.cov, dtype=float) + fix.cov
                        + np.eye(3) * spread ** 2)
            d2 = mahalanobis_sq(predicted, fix.enu, combined)
            if d2 < best_d2:
                best, best_d2 = state, d2
        return best if best_d2 <= GATE_CHI2_3DOF else None

    @staticmethod
    def _update_velocity(state: _TrackState, fix: Fix, dt: float) -> None:
        """Blend a new two-point velocity in, and keep what it is worth.

        `vel_sigma` is the part that was missing and that broke long-range
        tracking: the difference of two positions each good to sigma is a
        velocity good to only `sqrt(2)*sigma/dt`. At 2 km, where sigma is
        honestly 100 m, a 5 Hz difference is worth +/-700 m/s -- worthless, and
        predicting with it is worse than not predicting at all. Carrying the
        number means the gate can tell a trustworthy heading from a phantom.
        """
        t = state.track
        prev_var = float(np.trace(np.asarray(t.cov, dtype=float))) / 3.0
        new_var = float(np.trace(fix.cov)) / 3.0
        raw = (np.array(fix.enu) - np.array(t.enu)) / dt
        raw_sigma = math.sqrt(prev_var + new_var) / dt

        if t.vel is None:
            t.vel, state.vel_sigma = tuple(raw), raw_sigma
            return
        a = VEL_SMOOTHING
        t.vel = tuple(a * raw + (1.0 - a) * np.array(t.vel))
        # Variance of a weighted sum of two independent estimates.
        state.vel_sigma = math.sqrt((a * raw_sigma) ** 2
                                    + ((1.0 - a) * state.vel_sigma) ** 2)

    @staticmethod
    def _predict(state: _TrackState, now: float) -> tuple[np.ndarray, float]:
        """Where the track should be by now, and how far off that could be.

        The allowance is never worse than "anywhere within MAX_SPEED_MS of
        where we left it" -- that bound holds with no velocity at all, so a
        prediction is only ever allowed to *improve* on it. A track with a
        trustworthy heading gates on the manoeuvre it could have flown; a track
        whose velocity is noise falls back to the speed bound rather than
        chasing its own phantom.
        """
        track = state.track
        dt = max(now - track.last_seen, 0.0)
        here = np.array(track.enu)
        no_velocity = MAX_SPEED_MS * dt
        if track.vel is None:
            return here, no_velocity
        earned = state.vel_sigma * dt + 0.5 * MAX_ACCEL_MS2 * dt ** 2
        if earned >= no_velocity:
            return here, no_velocity      # the heading is not worth predicting with
        return here + np.array(track.vel) * dt, earned

    def _classify(self, state: _TrackState, now: float) -> None:
        t = state.track

        # Rule 1: without a healthy feed we cannot tell anything about anyone.
        # Not "assume the last answer still holds" -- the last answer is exactly
        # what has just become unsupported.
        if not self.feed.healthy(now):
            t.iff, t.friendly_id = IFF_UNKNOWN, None
            state.uncorrelated_since = None
            return

        match, _ = self.feed.match(np.array(t.enu), np.array(t.cov))
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
