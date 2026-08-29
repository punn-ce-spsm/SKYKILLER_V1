"""IFF state machine tests.

The exit criterion for phase B is at the top; the rest of the file exists to
pin the transitions that are *not* allowed, because those are the safety
property. Two of them matter more than everything else here:

  - nothing becomes FRIENDLY without a live correlation, and
  - nothing becomes HOSTILE while the feed is unhealthy.
"""

import numpy as np
import pytest

from skykiller.air_picture import (
    DEFAULT_FEED_MAX_AGE_S, DEFAULT_HOSTILE_HOLD_S, GATE_CHI2_3DOF,
    AirPicture, FriendlyFeed, mahalanobis_sq,
)
from skykiller.schemas import IFF_FRIENDLY, IFF_HOSTILE, IFF_UNKNOWN, Friendly
from skykiller.triangulate import Fix

TIGHT = np.eye(3) * 4.0        # a 2 m fix
COARSE = np.eye(3) * 10_000.0  # a 100 m fix, honest at long range


def _fix(enu, cov=TIGHT) -> Fix:
    return Fix(enu=np.array(enu, dtype=float), cov=np.asarray(cov, dtype=float),
               miss_m=0.0, sites=["A", "B"])


def _feed(entries, t=100.0) -> FriendlyFeed:
    f = FriendlyFeed()
    f.update(entries, t_utc=t)
    return f


# --- the phase B exit criterion --------------------------------------------

def test_one_matching_and_one_not_classify_friendly_and_hostile():
    blue = Friendly(id="blue-1", enu=(500.0, 500.0, 100.0), source="telemetry",
                    sigma_m=5.0, t_utc=100.0)
    ap = AirPicture(feed=_feed([blue]))

    ours = ap.ingest(_fix([502.0, 499.0, 100.0]), now=100.0)
    theirs = ap.ingest(_fix([-800.0, 1200.0, 60.0]), now=100.0)
    assert ours.iff == IFF_FRIENDLY and ours.friendly_id == "blue-1"

    # The hostile has to serve the hold first -- it is UNKNOWN until then.
    assert theirs.iff == IFF_UNKNOWN
    ap.feed.update([blue], t_utc=100.0 + DEFAULT_HOSTILE_HOLD_S)
    ap.reclassify(now=100.0 + DEFAULT_HOSTILE_HOLD_S)
    assert ap.get(theirs.id).iff == IFF_HOSTILE
    assert ap.get(ours.id).iff == IFF_FRIENDLY


def test_removing_the_feed_flips_friendly_to_unknown_not_to_friendly_or_hostile():
    """The other half of the exit criterion, and the one that could kill someone.

    When the feed dies every track stops correlating. Read naively that means
    'nothing is friendly', i.e. the whole sky is hostile -- asserted at the
    exact moment the system has lost its ability to tell. It must go the other
    way, and it must not silently keep saying FRIENDLY either.
    """
    blue = Friendly(id="blue-1", enu=(500.0, 500.0, 100.0), source="telemetry",
                    sigma_m=5.0, t_utc=100.0)
    ap = AirPicture(feed=_feed([blue]))
    track = ap.ingest(_fix([501.0, 500.0, 100.0]), now=100.0)
    assert track.iff == IFF_FRIENDLY

    # Feed goes silent. Walk well past its staleness window.
    dead = 100.0 + DEFAULT_FEED_MAX_AGE_S + 60.0
    ap.reclassify(now=dead)
    assert track.iff == IFF_UNKNOWN
    assert track.friendly_id is None

    # And it stays UNKNOWN however long the outage runs -- the hold must not
    # quietly promote it to HOSTILE in the background.
    ap.reclassify(now=dead + 3600.0)
    assert track.iff == IFF_UNKNOWN


def test_a_dead_feed_cannot_create_a_hostile_either():
    ap = AirPicture(feed=FriendlyFeed())         # never updated: never healthy
    track = ap.ingest(_fix([0.0, 1000.0, 80.0]), now=100.0)
    for t in (100.0, 200.0, 1000.0):
        ap.reclassify(now=t)
        assert track.iff == IFF_UNKNOWN


def test_an_empty_report_is_not_the_same_message_as_silence():
    """The distinction the heartbeat exists to preserve."""
    ap = AirPicture(feed=_feed([], t=100.0))     # explicit: nothing airborne
    track = ap.ingest(_fix([0.0, 1000.0, 80.0]), now=100.0)
    ap.feed.update([], t_utc=100.0 + DEFAULT_HOSTILE_HOLD_S)
    ap.reclassify(now=100.0 + DEFAULT_HOSTILE_HOLD_S)
    assert track.iff == IFF_HOSTILE              # healthy feed, nothing there

    ap2 = AirPicture(feed=FriendlyFeed())        # silence
    t2 = ap2.ingest(_fix([0.0, 1000.0, 80.0]), now=100.0)
    ap2.reclassify(now=100.0 + DEFAULT_HOSTILE_HOLD_S)
    assert t2.iff == IFF_UNKNOWN


# --- forbidden transitions --------------------------------------------------

def test_friendly_never_jumps_straight_to_hostile():
    """A dropped report must not re-label our own aircraft as a target."""
    blue = Friendly(id="blue-1", enu=(500.0, 500.0, 100.0), source="remote-id",
                    sigma_m=5.0, t_utc=100.0)
    ap = AirPicture(feed=_feed([blue]))
    track = ap.ingest(_fix([500.0, 500.0, 100.0]), now=100.0)
    assert track.iff == IFF_FRIENDLY

    # The feed stays alive but that aircraft's report goes missing.
    seen = []
    for t in np.arange(100.5, 100.5 + DEFAULT_HOSTILE_HOLD_S + 1.0, 0.5):
        ap.feed.update([], t_utc=float(t))
        ap.reclassify(now=float(t))
        seen.append(track.iff)

    assert seen[0] == IFF_UNKNOWN                 # via UNKNOWN, always
    assert IFF_HOSTILE in seen                    # and it does get there
    assert seen.index(IFF_UNKNOWN) < seen.index(IFF_HOSTILE)


def test_a_returning_report_takes_a_track_back_out_of_the_hold():
    blue = Friendly(id="blue-1", enu=(500.0, 500.0, 100.0), source="remote-id",
                    sigma_m=5.0, t_utc=100.0)
    ap = AirPicture(feed=_feed([blue]))
    track = ap.ingest(_fix([500.0, 500.0, 100.0]), now=100.0)

    ap.feed.update([], t_utc=101.0)               # one missed report
    ap.reclassify(now=101.0)
    assert track.iff == IFF_UNKNOWN

    ap.feed.update([blue], t_utc=101.5)           # it comes back
    ap.reclassify(now=101.5)
    assert track.iff == IFF_FRIENDLY

    # The hold must have been reset, not merely paused: waiting out the old
    # hold window with a live correlation must not produce HOSTILE.
    ap.feed.update([blue], t_utc=101.5 + DEFAULT_HOSTILE_HOLD_S + 1.0)
    ap.reclassify(now=101.5 + DEFAULT_HOSTILE_HOLD_S + 1.0)
    assert track.iff == IFF_FRIENDLY


def test_the_hold_is_served_once_not_restarted_by_each_new_fix():
    ap = AirPicture(feed=_feed([]))
    track = ap.ingest(_fix([0.0, 1000.0, 80.0]), now=100.0)
    for t in (100.5, 101.0, 101.5, 102.0, 102.5, 103.0):
        ap.feed.update([], t_utc=t)
        track = ap.ingest(_fix([0.0, 1000.0, 80.0]), now=t)
    assert track.iff == IFF_HOSTILE


# --- the correlation gate ---------------------------------------------------

def test_the_gate_widens_with_the_track_covariance():
    """Why the gate is Mahalanobis and not a radius in metres.

    The same 60 m offset is a decisive mismatch for a 2 m fix and comfortably
    inside the noise for a 100 m one. A fixed radius has to choose one of those
    to get wrong.
    """
    blue = Friendly(id="blue-1", enu=(0.0, 1000.0, 80.0), source="telemetry",
                    sigma_m=5.0, t_utc=100.0)
    offset = [60.0, 1000.0, 80.0]

    tight = AirPicture(feed=_feed([blue]))
    assert tight.ingest(_fix(offset, TIGHT), now=100.0).iff == IFF_UNKNOWN

    coarse = AirPicture(feed=_feed([blue]))
    assert coarse.ingest(_fix(offset, COARSE), now=100.0).iff == IFF_FRIENDLY


def test_a_vague_friendly_gets_a_correspondingly_generous_gate():
    vague = Friendly(id="blue-1", enu=(0.0, 1000.0, 80.0), source="manual",
                     sigma_m=100.0, t_utc=100.0)
    precise = Friendly(id="blue-1", enu=(0.0, 1000.0, 80.0), source="remote-id",
                       sigma_m=2.0, t_utc=100.0)
    offset = [60.0, 1000.0, 80.0]
    assert AirPicture(feed=_feed([vague])).ingest(
        _fix(offset), now=100.0).iff == IFF_FRIENDLY
    assert AirPicture(feed=_feed([precise])).ingest(
        _fix(offset), now=100.0).iff == IFF_UNKNOWN


def test_the_nearest_of_several_friendlies_is_the_one_reported():
    near = Friendly(id="blue-near", enu=(0.0, 1000.0, 80.0), source="telemetry",
                    sigma_m=5.0, t_utc=100.0)
    far = Friendly(id="blue-far", enu=(8.0, 1000.0, 80.0), source="telemetry",
                   sigma_m=5.0, t_utc=100.0)
    ap = AirPicture(feed=_feed([far, near]))
    assert ap.ingest(_fix([0.5, 1000.0, 80.0]), now=100.0).friendly_id == "blue-near"


def test_mahalanobis_is_scale_free():
    a, b = np.zeros(3), np.array([10.0, 0.0, 0.0])
    assert mahalanobis_sq(a, b, np.eye(3) * 25.0) == pytest.approx(4.0)   # 2 sigma
    assert mahalanobis_sq(a, b, np.eye(3) * 100.0) == pytest.approx(1.0)  # 1 sigma
    assert mahalanobis_sq(a, b, np.zeros((3, 3))) == float("inf")         # singular


def test_the_known_weakness_is_real_and_documented():
    """A hostile inside a friendly's gate is called FRIENDLY. Pinned honestly.

    This is inherent to position-correlation IFF, not a defect in this code.
    The point of the test is that nobody discovers it in the field.
    """
    blue = Friendly(id="blue-1", enu=(0.0, 1000.0, 80.0), source="telemetry",
                    sigma_m=5.0, t_utc=100.0)
    ap = AirPicture(feed=_feed([blue]))
    shadowing = ap.ingest(_fix([3.0, 1002.0, 81.0]), now=100.0)
    assert shadowing.iff == IFF_FRIENDLY


# --- the track store --------------------------------------------------------

def test_repeated_fixes_of_one_target_stay_one_track():
    ap = AirPicture(feed=_feed([]))
    ids = {ap.ingest(_fix([n, 1000.0, 80.0]), now=100.0 + n * 0.1).id
           for n in range(10)}
    assert len(ids) == 1


def test_two_separated_targets_are_two_tracks():
    ap = AirPicture(feed=_feed([]))
    a = ap.ingest(_fix([0.0, 1000.0, 80.0]), now=100.0)
    b = ap.ingest(_fix([400.0, 1000.0, 80.0]), now=100.0)
    assert a.id != b.id
    assert len(ap.tracks) == 2


def test_velocity_is_derived_from_consecutive_fixes():
    ap = AirPicture(feed=_feed([]))
    ap.ingest(_fix([0.0, 1000.0, 80.0]), now=100.0)
    t = ap.ingest(_fix([0.0, 990.0, 80.0]), now=101.0)   # 10 m closer in 1 s
    assert t.vel == pytest.approx((0.0, -10.0, 0.0))


def test_stale_tracks_are_pruned_and_do_not_capture_new_fixes():
    ap = AirPicture(feed=_feed([]))
    old = ap.ingest(_fix([0.0, 1000.0, 80.0]), now=100.0)
    later = 100.0 + ap.track_max_age_s + 1.0
    fresh = ap.ingest(_fix([0.0, 1000.0, 80.0]), now=later)
    assert fresh.id != old.id                    # too old to associate to
    assert ap.prune(now=later) == [old.id]
    assert [t.id for t in ap.tracks] == [fresh.id]


# --- association under motion, the case a position-only gate got wrong ------

def _fly(ap, start, vel, *, hz: float, seconds: float, t0: float = 100.0,
         cov=TIGHT) -> list[str]:
    """Fly a straight line and return the track id seen at each frame."""
    ids = []
    for n in range(int(hz * seconds)):
        t = t0 + n / hz
        pos = np.array(start, dtype=float) + np.array(vel, dtype=float) * (t - t0)
        ap.feed.update([], t_utc=t)
        ids.append(ap.ingest(_fix(pos, cov), now=t).id)
    return ids


def test_a_closing_target_holds_one_track_id():
    """Regression: the gate must predict, not compare against the last fix.

    15 m/s at 5 Hz is 3 m between frames -- a 1.5-sigma step against a 2 m fix,
    which a position-only gate rejects. That split one aircraft into a new
    track every frame, and did it *more* readily the better the fix was.
    """
    ap = AirPicture(feed=_feed([]))
    ids = _fly(ap, [0.0, 1500.0, 80.0], [0.0, -15.0, 0.0], hz=5, seconds=20)
    assert len(set(ids)) == 1, f"{len(set(ids))} tracks for one aircraft"
    assert len(ap.tracks) == 1


def test_it_holds_even_at_1hz_where_the_target_jumps_15m_per_frame():
    ap = AirPicture(feed=_feed([]))
    ids = _fly(ap, [0.0, 1500.0, 80.0], [0.0, -15.0, 0.0], hz=1, seconds=30)
    assert len(set(ids)) == 1


def test_a_coarse_long_range_fix_does_not_spawn_tracks_from_its_own_noise():
    ap = AirPicture(feed=_feed([]))
    rng = np.random.default_rng(41)
    ids = []
    for n in range(40):
        t = 100.0 + n * 0.2
        jitter = rng.normal(0, 100.0, 3)          # the real error at 2 km
        ids.append(ap.ingest(_fix(np.array([0.0, 2000.0, 80.0]) + jitter, COARSE),
                             now=t).id)
    assert len(set(ids)) == 1


def test_two_aircraft_on_parallel_courses_stay_two_tracks():
    """The gate must be wide enough to follow motion and no wider."""
    ap = AirPicture(feed=_feed([]))
    seen = []
    for n in range(50):
        t = 100.0 + n * 0.2
        north = 1500.0 - 15.0 * (t - 100.0)
        a = ap.ingest(_fix([-100.0, north, 80.0]), now=t).id
        b = ap.ingest(_fix([100.0, north, 80.0]), now=t).id
        seen.append((a, b))
    assert len(ap.tracks) == 2
    assert seen[-1][0] != seen[-1][1]
    assert seen[-1] == seen[1]                    # and never swapped over


def test_a_good_fix_earns_a_tighter_gate_than_a_bad_one():
    """A heading is only worth predicting with if the fixes behind it were good.

    Same aircraft, same speed, same rate -- only the fix quality differs. The
    2 m fix earns a prediction; the 100 m fix does not, and correctly falls
    back to the plain speed bound rather than flying off on a phantom.
    """
    from skykiller.air_picture import MAX_SPEED_MS

    def allowance(cov) -> float:
        ap = AirPicture(feed=_feed([]))
        for n in range(6):
            t = 100.0 + n * 0.2
            ap.ingest(_fix([0.0, 1000.0 - 15.0 * n * 0.2, 80.0], cov), now=t)
        state = next(iter(ap._states.values()))
        return AirPicture._predict(state, 100.0 + 6 * 0.2)[1]

    assert allowance(TIGHT) < allowance(COARSE)
    assert allowance(COARSE) == pytest.approx(MAX_SPEED_MS * 0.2)   # bound, not phantom
    assert allowance(TIGHT) < MAX_SPEED_MS * 0.2


def test_a_phantom_velocity_never_widens_the_gate_beyond_the_speed_bound():
    """The invariant that keeps a bad velocity from being worse than none."""
    from skykiller.air_picture import MAX_SPEED_MS
    ap = AirPicture(feed=_feed([]))
    rng = np.random.default_rng(53)
    for n in range(20):
        t = 100.0 + n * 0.2
        ap.ingest(_fix(np.array([0.0, 2000.0, 80.0]) + rng.normal(0, 100.0, 3),
                       COARSE), now=t)
    state = next(iter(ap._states.values()))
    for dt in (0.2, 1.0, 3.0):
        _, spread = AirPicture._predict(state, state.track.last_seen + dt)
        assert spread <= MAX_SPEED_MS * dt + 1e-9
