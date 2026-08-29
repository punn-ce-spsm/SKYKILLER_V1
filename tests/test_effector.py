"""Effector handoff: the re-projection, the beam budget, and the ROE gate.

The gate tests are the ones that matter. Most of them assert that something
does *not* happen, because the safety property of a decision gate is entirely
about what it refuses.
"""

import math

import numpy as np
import pytest

from skykiller.effector import (
    ARM_VALID_S, ARMED, AUTHORISED, PROMPTED, Effector, RoeGate,
)
from skykiller.schemas import IFF_FRIENDLY, IFF_HOSTILE, IFF_UNKNOWN, Track
from skykiller.triangulate import to_bearing

TIGHT = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
COARSE = [[10_000.0, 0.0, 0.0], [0.0, 10_000.0, 0.0], [0.0, 0.0, 10_000.0]]


def _jammer(beamwidth=60.0, envelope=500.0, enu=(0.0, 0.0, 5.0)) -> Effector:
    return Effector(id="J1", enu=np.array(enu), tier="T1",
                    beamwidth_deg=beamwidth, envelope_m=envelope)


def _track(enu, iff=IFF_HOSTILE, cov=TIGHT, tid="K-001") -> Track:
    return Track(id=tid, enu=tuple(enu), cov=cov, iff=iff,
                 first_seen=100.0, last_seen=100.0)


# --- the re-projection ------------------------------------------------------

def test_the_bearing_is_taken_from_the_effector_not_the_observer():
    """The reason fusion needs a position and not just a bearing.

    Observer on a mast at the origin, effector 400 m east on the ground, target
    300 m north. The two see it 53 degrees apart -- pass the observer's bearing
    through and the effector points at nothing.
    """
    effector = _jammer(enu=(400.0, 0.0, 5.0))
    target = np.array([0.0, 300.0, 60.0])
    observer = np.array([0.0, 0.0, 200.0])

    from_observer, _ = to_bearing(target - observer)
    az, el, range_m = effector.aim(target)

    assert az == pytest.approx(math.degrees(math.atan2(-400.0, 300.0)) % 360.0, abs=1e-6)
    assert range_m == pytest.approx(float(np.linalg.norm(target - effector.enu)))
    assert abs(((az - from_observer + 180.0) % 360.0) - 180.0) > 50.0


def test_the_aim_actually_points_at_the_target():
    """Phase C exit criterion: the bearing is right to within the beam half-width."""
    effector = _jammer(beamwidth=10.0, enu=(250.0, -100.0, 8.0))
    for target in ([0.0, 400.0, 60.0], [-300.0, 200.0, 120.0], [100.0, -50.0, 30.0]):
        az, el, range_m = effector.aim(target)
        d = np.array([math.sin(math.radians(az)) * math.cos(math.radians(el)),
                      math.cos(math.radians(az)) * math.cos(math.radians(el)),
                      math.sin(math.radians(el))])
        # The unit vector times the range must land on the target exactly.
        assert effector.enu + d * range_m == pytest.approx(np.array(target), abs=1e-9)


def test_elevation_is_signed_correctly_for_a_ground_effector():
    effector = _jammer(enu=(0.0, 0.0, 5.0))
    assert effector.aim([0.0, 300.0, 200.0])[1] > 0     # looking up at a drone
    assert effector.aim([0.0, 300.0, 2.0])[1] < 0       # below the emitter


# --- the beam budget --------------------------------------------------------

def test_beam_width_grows_linearly_with_range():
    j = _jammer(beamwidth=60.0)
    assert j.beam_width_m(500.0) == pytest.approx(2 * 500 * math.tan(math.radians(30)))
    assert j.beam_width_m(1000.0) == pytest.approx(2 * j.beam_width_m(500.0))


def test_a_wide_beam_tolerates_an_error_a_narrow_one_does_not():
    """The accuracy requirement is the beam width, not a number we chose."""
    target, cov = [0.0, 500.0, 80.0], COARSE      # 100 m sigma
    assert _jammer(beamwidth=60.0).beam_covers(target, cov)
    assert not _jammer(beamwidth=10.0).beam_covers(target, cov)


def test_error_along_the_boresight_does_not_count_against_coverage():
    """Uncertainty in range moves the target within the same cone.

    Counting it would reject solutions for an error that cannot cause a miss --
    which matters here specifically, because a bearings-only fix's error
    ellipsoid is elongated *down-range*, exactly the direction that is free.
    """
    j = _jammer(beamwidth=10.0)
    target = [0.0, 500.0, 5.0]                    # due north of the emitter
    down_range = [[1.0, 0.0, 0.0], [0.0, 40_000.0, 0.0], [0.0, 0.0, 1.0]]
    across = [[40_000.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert j.beam_covers(target, down_range)      # 200 m of range error: fine
    assert not j.beam_covers(target, across)      # 200 m across the beam: not


def test_the_same_fix_stops_being_covered_as_the_target_closes():
    """Counter-intuitive and worth pinning: the cone narrows as it shortens."""
    j = _jammer(beamwidth=10.0)
    cov = [[400.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 400.0]]   # 20 m sigma
    assert j.beam_covers([0.0, 500.0, 5.0], cov)
    assert not j.beam_covers([0.0, 100.0, 5.0], cov)


def test_the_plan_accuracy_table_holds():
    """The numbers the whole design was sized against, checked."""
    for beamwidth, expected in ((60.0, 577.0), (30.0, 268.0), (10.0, 87.0)):
        assert _jammer(beamwidth=beamwidth).beam_width_m(500.0) == pytest.approx(
            expected, rel=0.01)


# --- the envelope and the prompt --------------------------------------------

def test_a_track_crossing_the_boundary_raises_exactly_one_prompt():
    """Phase C exit criterion. Jitter on the boundary must not re-prompt."""
    gate = RoeGate(effector=_jammer(envelope=500.0))
    prompts = 0
    for n in range(120):
        # Closes from 700 m to 400 m, then hovers on the 500 m line.
        north = 700.0 - 3.0 * n if n < 100 else 500.0 + (-1) ** n * 8.0
        prompts += len(gate.update([_track([0.0, north, 5.0])], now=100.0 + n))
    assert prompts == 1


def test_no_prompt_outside_the_envelope():
    gate = RoeGate(effector=_jammer(envelope=500.0))
    assert gate.update([_track([0.0, 900.0, 5.0])], now=100.0) == []
    assert gate.state("K-001") is None


def test_leaving_and_returning_raises_a_second_prompt():
    """A new crossing is a new decision, so it must ask again."""
    gate = RoeGate(effector=_jammer(envelope=500.0))
    assert len(gate.update([_track([0.0, 400.0, 5.0])], now=100.0)) == 1
    assert gate.update([_track([0.0, 900.0, 5.0])], now=101.0) == []   # withdrawn
    assert gate.state("K-001") is None
    assert len(gate.update([_track([0.0, 400.0, 5.0])], now=102.0)) == 1


# --- what the gate refuses --------------------------------------------------

def test_only_a_hostile_track_is_ever_prompted():
    gate = RoeGate(effector=_jammer())
    for verdict in (IFF_FRIENDLY, IFF_UNKNOWN):
        assert gate.update([_track([0.0, 400.0, 5.0], iff=verdict)], now=100.0) == []
        assert gate.state("K-001") is None


def test_a_standing_arm_is_withdrawn_when_the_track_stops_being_hostile():
    """The case the whole gate exists for.

    An arm is not a token to be spent later. If the friendly feed catches up and
    the track turns out to be ours, the decision is revoked without anyone
    having to remember to revoke it.
    """
    gate = RoeGate(effector=_jammer())
    gate.update([_track([0.0, 400.0, 5.0])], now=100.0)
    assert gate.arm("K-001", operator="op", now=100.0)
    assert gate.state("K-001") == ARMED

    ours = _track([0.0, 400.0, 5.0], iff=IFF_FRIENDLY)
    gate.update([ours], now=101.0)
    assert gate.state("K-001") is None
    assert gate.authorise(ours, operator="op", now=101.0) is None


def test_a_standing_arm_is_withdrawn_when_the_track_is_lost():
    gate = RoeGate(effector=_jammer())
    gate.update([_track([0.0, 400.0, 5.0])], now=100.0)
    gate.arm("K-001", operator="op", now=100.0)
    gate.update([], now=101.0)                     # track pruned from the picture
    assert gate.state("K-001") is None


def test_an_arm_lapses_and_the_operator_is_asked_again():
    """A stale arm is withdrawn, but the threat has not gone away.

    The target is still hostile and still inside the envelope, so the right
    outcome is not silence -- it is a fresh prompt. What must not survive is
    the *authority*: the operator has to decide again against the situation as
    it now is.
    """
    gate = RoeGate(effector=_jammer())
    track = _track([0.0, 400.0, 5.0])
    gate.update([track], now=100.0)
    gate.arm("K-001", operator="op", now=100.0)
    assert gate.state("K-001") == ARMED

    late = 100.0 + ARM_VALID_S + 1.0
    re_prompted = gate.update([track], now=late)

    assert gate.state("K-001") == PROMPTED             # not ARMED any more
    assert [t.id for t in re_prompted] == ["K-001"]    # and the operator is told
    assert gate.authorise(track, operator="op", now=late) is None


def test_a_lapsed_arm_does_not_reprompt_faster_than_the_validity_window():
    """The prompt must not turn into a loop the operator learns to ignore."""
    gate = RoeGate(effector=_jammer())
    track = _track([0.0, 400.0, 5.0])
    prompts = 0
    for n in range(200):                                # 200 s at 1 Hz
        now = 100.0 + n
        prompts += len(gate.update([track], now=now))
        gate.arm("K-001", operator="op", now=now)       # operator keeps arming
    assert prompts <= 200 / ARM_VALID_S + 1


def test_authorise_without_arming_is_refused():
    gate = RoeGate(effector=_jammer())
    track = _track([0.0, 400.0, 5.0])
    gate.update([track], now=100.0)
    assert gate.state("K-001") == PROMPTED
    assert gate.authorise(track, operator="op", now=100.0) is None


def test_arming_without_a_prompt_is_refused():
    gate = RoeGate(effector=_jammer())
    assert not gate.arm("K-001", operator="op", now=100.0)
    assert not gate.arm("nonexistent", operator="op", now=100.0)


def test_the_gate_never_arms_itself():
    """No sequence of updates alone produces an authorised request."""
    gate = RoeGate(effector=_jammer())
    track = _track([0.0, 400.0, 5.0])
    for n in range(200):
        gate.update([track], now=100.0 + n)
        assert gate.state(track.id) == PROMPTED
        assert gate.authorise(track, operator="op", now=100.0 + n) is None


# --- the emitted request ----------------------------------------------------

def _authorised(gate, track, now=100.0):
    gate.update([track], now=now)
    gate.arm(track.id, operator="op", now=now)
    return gate.authorise(track, operator="op", now=now + 1.0)


def test_an_authorised_request_carries_the_decision_and_the_uncertainty():
    gate = RoeGate(effector=_jammer(beamwidth=60.0, envelope=500.0))
    track = _track([0.0, 400.0, 60.0])
    req = _authorised(gate, track)

    assert req is not None
    assert req.simulated is True
    assert req.track_id == "K-001" and req.jammer_id == "J1" and req.tier == "T1"
    assert req.in_envelope and req.beam_covers
    assert req.pos_cov == TIGHT
    assert req.operator == "op"
    assert req.armed_at == 100.0 and req.authorised_at == 101.0
    assert gate.state("K-001") == AUTHORISED

    az, el, range_m = gate.effector.aim(track.enu)
    assert (req.aim_az, req.aim_el, req.range_m) == pytest.approx((az, el, range_m))
    assert req.to_json()                                   # survives the wire


def test_the_request_reports_an_uncovered_target_rather_than_hiding_it():
    """A poor fix does not block authorisation -- it is reported to the operator.

    Refusing here would be the wrong call: whether a marginal solution is worth
    firing is a judgement about the situation, and the gate's job is to make
    sure the judgement is informed, not to make it.
    """
    gate = RoeGate(effector=_jammer(beamwidth=10.0, envelope=500.0))
    req = _authorised(gate, _track([0.0, 400.0, 60.0], cov=COARSE))
    assert req is not None
    assert req.in_envelope
    assert not req.beam_covers


def test_the_aiming_solution_uses_the_position_at_authorisation_not_at_prompt():
    """A request must record the decision that was actually made."""
    gate = RoeGate(effector=_jammer(envelope=600.0))
    gate.update([_track([0.0, 500.0, 60.0])], now=100.0)
    gate.arm("K-001", operator="op", now=100.0)
    moved = _track([0.0, 300.0, 60.0])                     # closed 200 m
    req = gate.authorise(moved, operator="op", now=105.0)
    assert req.range_m == pytest.approx(gate.effector.aim(moved.enu)[2])
    assert req.range_m < 400.0


# --- the edges, probed rather than assumed -----------------------------------

def test_a_surviving_hostile_can_be_engaged_again():
    """An authorisation completes an engagement; it does not consume the track.

    Leaving the authorised state standing locked a survivor out for the whole
    arm-validity window -- 30 s, or 450 m of closure -- as an accident of the
    staleness check rather than a decision anyone made.
    """
    gate = RoeGate(effector=_jammer())
    track = _track([0.0, 400.0, 5.0])
    gate.update([track], now=100.0)
    gate.arm(track.id, operator="op", now=100.0)
    assert gate.authorise(track, operator="op", now=101.0) is not None

    re_prompted = gate.update([track], now=101.2)
    assert [t.id for t in re_prompted] == ["K-001"]
    assert gate.state("K-001") == PROMPTED

    # And the second engagement still needs both human acts.
    assert gate.authorise(track, operator="op", now=101.4) is None
    assert gate.arm(track.id, operator="op", now=101.4)
    assert gate.authorise(track, operator="op", now=101.6) is not None


def test_one_authorisation_yields_one_request():
    gate = RoeGate(effector=_jammer())
    track = _track([0.0, 400.0, 5.0])
    gate.update([track], now=100.0)
    gate.arm(track.id, operator="op", now=100.0)
    assert gate.authorise(track, operator="op", now=101.0) is not None
    assert gate.authorise(track, operator="op", now=101.0) is None   # not twice
    assert not gate.arm(track.id, operator="op", now=101.0)          # nor re-armed


def test_identity_is_checked_at_authorisation_even_without_an_update():
    """Defence in depth: the refusal must not depend on the caller's loop."""
    gate = RoeGate(effector=_jammer())
    gate.update([_track([0.0, 400.0, 5.0])], now=100.0)
    gate.arm("K-001", operator="op", now=100.0)
    ours = _track([0.0, 400.0, 5.0], iff=IFF_FRIENDLY)
    assert gate.authorise(ours, operator="op", now=101.0) is None


def test_range_is_reported_rather_than_refused():
    """The deliberate asymmetry with identity, pinned so it stays deliberate."""
    gate = RoeGate(effector=_jammer(envelope=500.0))
    gate.update([_track([0.0, 400.0, 5.0])], now=100.0)
    gate.arm("K-001", operator="op", now=100.0)
    fled = _track([0.0, 5000.0, 60.0])
    req = gate.authorise(fled, operator="op", now=101.0)
    assert req is not None
    assert req.in_envelope is False
    assert req.range_m > 4000.0
