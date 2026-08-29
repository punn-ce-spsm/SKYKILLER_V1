"""End-to-end runs through the real pipeline.

These are the tests that found what the unit tests could not. Both serious
build-2 defects -- unpaired cross-site detections, and ghosts surviving into
HOSTILE -- were invisible module by module and obvious the moment a second
aircraft went into the sky.
"""

import numpy as np
import pytest

from skykiller.air_picture import AirPicture, FriendlyFeed
from skykiller.effector import ARMED, Effector, RoeGate
from skykiller.geometry import Camera
from skykiller.schemas import IFF_FRIENDLY, IFF_HOSTILE, IFF_UNKNOWN
from skykiller.scenario import Aircraft, Scenario, orbit, run, straight_line
from skykiller.sites import Site, SiteNetwork

CAM = Camera(width=1280, height=720, hfov_deg=65.0)


def _net(baseline: float = 200.0, mast: float = 200.0) -> SiteNetwork:
    """The sited configuration: masts above the traffic, 200 m apart.

    Both numbers are the measured recommendations, not defaults -- see
    SiteNetwork.max_range_m and elevation_separation_deg.
    """
    return SiteNetwork({"A": Site("A", np.array([0.0, 0.0, mast]), CAM),
                        "B": Site("B", np.array([baseline, 0.0, mast]), CAM)})


def _scenario(seed: int = 0, baseline: float = 200.0, mast: float = 200.0,
              duration_s: float = 80.0, feed_alive=lambda t: True) -> Scenario:
    """A hostile inbound at 15 m/s while one of ours orbits inside the envelope.

    80 seconds because that is how long the ingress actually takes: from 1.5 km
    to the 500 m envelope edge is 1 km at 15 m/s, or 67 s. A shorter run ends
    with the target still outside the envelope and tests nothing about the gate.
    """
    return Scenario(
        net=_net(baseline, mast),
        aircraft=[
            Aircraft(id="red-1", path=straight_line([0.0, 1500.0, 60.0],
                                                    [0.0, -15.0, 0.0])),
            Aircraft(id="blue-1", friendly=True,
                     path=orbit([0.0, 400.0], radius=250.0, altitude=100.0)),
        ],
        duration_s=duration_s, seed=seed, feed_alive=feed_alive)


def _gate(envelope: float = 500.0, beamwidth: float = 60.0) -> RoeGate:
    return RoeGate(effector=Effector(id="J1", enu=np.array([100.0, 0.0, 5.0]),
                                     tier="T1", beamwidth_deg=beamwidth,
                                     envelope_m=envelope))


# --- the phase C exit criterion ---------------------------------------------

def test_one_ingress_raises_exactly_one_arm_prompt():
    for seed in range(5):
        gate = _gate()
        prompts = [p for f in run(_scenario(seed), gate=gate) for p in f.prompts]
        assert len(prompts) == 1, f"seed {seed}: {len(prompts)} prompts"


def test_the_prompt_is_for_the_hostile_and_never_for_our_own_aircraft():
    """The friendly orbits *inside* the envelope for the whole run."""
    scenario = _scenario(seed=1)
    gate = _gate()
    prompted = []
    for frame in run(scenario, gate=gate):
        for track in frame.prompts:
            prompted.append(scenario.nearest_truth(track.enu, frame.t)[0])
    assert prompted == ["red-1"]


def test_the_friendly_is_never_declared_hostile_at_any_point():
    scenario = _scenario(seed=2)
    for frame in run(scenario):
        for track in frame.tracks:
            who, distance = scenario.nearest_truth(track.enu, frame.t)
            if who == "blue-1" and distance < 100.0:
                assert track.iff != IFF_HOSTILE, f"t={frame.t}"


def test_the_aiming_bearing_points_at_the_real_target():
    """The other half of the exit criterion, measured against truth.

    The request must point at where the aircraft actually is, from the
    effector's position, to within the beam half-width.
    """
    scenario = _scenario(seed=3)
    gate = _gate(beamwidth=60.0)
    request = None
    for frame in run(scenario, gate=gate):
        for track in frame.prompts:
            gate.arm(track.id, operator="test", now=frame.t)
        if request is None:
            for track in frame.tracks:
                if gate.state(track.id) == ARMED:
                    request = (gate.authorise(track, operator="test", now=frame.t),
                               frame.t)
                    break
    assert request is not None, "the run never reached an authorised request"
    req, t = request
    assert req.simulated is True

    truth = scenario.truth(t)["red-1"]
    true_az, true_el, _ = gate.effector.aim(truth)
    half_width = gate.effector.beamwidth_deg / 2.0
    assert abs(((req.aim_az - true_az + 180.0) % 360.0) - 180.0) < half_width
    assert abs(req.aim_el - true_el) < half_width
    assert req.in_envelope
    assert len(req.pos_cov) == 3          # the uncertainty travels with it


def test_no_ghost_tracks_survive_in_the_sited_configuration():
    """Two aircraft, twelve seeds, masts above the traffic. No invented targets."""
    for seed in range(12):
        scenario = _scenario(seed)
        final = None
        for frame in run(scenario):
            final = frame
        for track in final.tracks:
            who, distance = scenario.nearest_truth(track.enu, final.t)
            if distance > 300.0:
                assert track.iff != IFF_HOSTILE, (
                    f"seed {seed}: ghost {distance:.0f} m from anything, "
                    f"declared HOSTILE")


def test_poor_siting_costs_accuracy_and_warning_time():
    """The siting recommendations have to be load-bearing, or they are advice.

    They used to be load-bearing in an alarming way: before the track filter
    and merge existed, masts level with the traffic produced ghost tracks
    declared HOSTILE in most runs. That failure is gone -- the system now
    degrades rather than inventing targets, which is the right shape -- so the
    recommendations are pinned by what they still buy.

    Measured over eight seeds of the two-aircraft ingress:

        100 m baseline, 120 m masts   30.0 m median error, hostile at 15.4 s
        200 m baseline, 120 m masts   14.7 m,              hostile at  8.7 s
        200 m baseline, 200 m masts   14.3 m,              hostile at  3.0 s

    Twelve seconds of declaration delay is 185 m of standoff at 15 m/s, which
    is most of a jammer envelope.
    """
    def measure(baseline: float, mast: float) -> tuple[float, float]:
        errors, declared = [], []
        for seed in range(8):
            scenario = _scenario(seed, baseline=baseline, mast=mast)
            first = None
            for frame in run(scenario):
                for track in frame.tracks:
                    d = float(np.linalg.norm(
                        np.array(track.enu) - scenario.truth(frame.t)["red-1"]))
                    if d < 200.0:
                        errors.append(d)
                        if track.iff == IFF_HOSTILE and first is None:
                            first = frame.t
            if first is not None:
                declared.append(first)
        return float(np.median(errors)), float(np.median(declared))

    narrow_err, narrow_t = measure(100.0, 120.0)
    wide_err, wide_t = measure(200.0, 120.0)
    tall_err, tall_t = measure(200.0, 200.0)

    # Baseline buys accuracy.
    assert narrow_err > 1.5 * wide_err
    # Mast height buys warning time, without changing accuracy much.
    assert tall_t < wide_t < narrow_t
    assert tall_t <= 5.0
    # And in none of them does the system invent a target any more.
    assert tall_err == pytest.approx(wide_err, rel=0.4)


# --- the feed failing mid-run, end to end -----------------------------------

def test_killing_the_feed_mid_run_makes_everything_unknown_not_hostile():
    """Phase B's rule, exercised through the whole chain rather than in a stub."""
    scenario = _scenario(seed=4, feed_alive=lambda t: t < 20.0)
    gate = _gate()
    after = [f for f in run(scenario, gate=gate) if f.t > 26.0]
    assert after, "the run must continue past the outage"
    for frame in after:
        assert all(t.iff == IFF_UNKNOWN for t in frame.tracks), f"t={frame.t}"
        assert frame.prompts == []


def test_the_friendly_is_recognised_again_when_the_feed_returns():
    scenario = _scenario(seed=5, feed_alive=lambda t: not 20.0 <= t < 30.0)
    seen = {}
    for frame in run(scenario):
        for track in frame.tracks:
            who, distance = scenario.nearest_truth(track.enu, frame.t)
            if who == "blue-1" and distance < 100.0:
                seen.setdefault(round(frame.t), track.iff)
    assert seen[10] == IFF_FRIENDLY
    assert seen[35] == IFF_FRIENDLY     # after the outage
    assert all(v != IFF_HOSTILE for v in seen.values())
