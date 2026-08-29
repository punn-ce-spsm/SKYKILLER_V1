"""The measured demonstration: what the customer is actually buying.

The headline test is the altitude comparison. Same target, same trajectory,
same software, same obstacle -- only the observer's height differs, and the
warning time moves by an order of magnitude. That is the product case, and it
is measured here rather than asserted anywhere.
"""

import numpy as np
import pytest

from skykiller.effector import Effector
from skykiller.geometry import Camera
from skykiller.harness import GHOST_RADIUS_M, measure
from skykiller.masking import treeline
from skykiller.scenario import Aircraft, Scenario, orbit, straight_line
from skykiller.sites import Site, SiteNetwork

CAM = Camera(width=1280, height=720, hfov_deg=65.0)
TREELINE = [treeline("treeline", height_m=20.0, distance_m=200.0)]


def _effector() -> Effector:
    return Effector(id="J1", enu=np.array([100.0, 0.0, 5.0]), tier="T1",
                    beamwidth_deg=60.0, envelope_m=500.0)


def _scenario(mast: float, seed: int = 0, obstacles=None,
              duration_s: float = 95.0) -> Scenario:
    """A hostile inbound at 50 m AGL -- below any realistic treeline."""
    return Scenario(
        net=SiteNetwork({"A": Site("A", np.array([0.0, 0.0, mast]), CAM),
                         "B": Site("B", np.array([200.0, 0.0, mast]), CAM)}),
        aircraft=[
            Aircraft(id="red-1", path=straight_line([0.0, 1500.0, 50.0],
                                                    [0.0, -15.0, 0.0])),
            Aircraft(id="blue-1", friendly=True,
                     path=orbit([0.0, 400.0], radius=250.0, altitude=100.0)),
        ],
        duration_s=duration_s, seed=seed,
        obstacles=TREELINE if obstacles is None else obstacles)


# --- the phase D exit criterion ---------------------------------------------

def test_the_harness_reports_error_against_truth_for_a_full_run():
    report = measure(_scenario(mast=200.0), "red-1", _effector())
    assert report.first_seen_t is not None
    assert len(report.errors_m) > 100
    assert report.error_p50 is not None and report.error_p95 is not None
    assert report.error_p50 < report.error_p95
    assert report.error_at_envelope_m is not None
    assert report.dwell_s is not None
    assert all(line for line in report.lines())


def test_altitude_buys_an_order_of_magnitude_of_warning():
    """The demonstration, side by side. Nothing differs but observer height.

    Behind a 20 m treeline 200 m out, a 3 m camera cannot see below 88 m at
    1 km -- and the target flies at 50 m. It is held only once it is almost on
    top of the position.
    """
    ground = measure(_scenario(mast=3.0), "red-1", _effector())
    airborne = measure(_scenario(mast=200.0), "red-1", _effector())

    assert ground.warning_s < 10.0
    assert airborne.warning_s > 60.0
    assert airborne.warning_s > 10 * ground.warning_s

    # And the elevated pair holds it from far further out.
    assert ground.first_seen_range_m < 700.0
    assert airborne.first_seen_range_m > 1400.0


def test_the_elevated_pair_is_also_more_accurate_where_it_matters():
    """Guards against a real way of reading the numbers wrongly.

    The elevated pair's median error over the whole run is *worse*, because it
    holds the target for 95 s of mostly long range where a bearings-only fix is
    honestly poor, while the ground camera only ever sees it inside 600 m. At
    the envelope edge -- the one moment an effector cares about -- the elevated
    pair is better.
    """
    ground = measure(_scenario(mast=3.0), "red-1", _effector())
    airborne = measure(_scenario(mast=200.0), "red-1", _effector())

    assert airborne.error_p50 > ground.error_p50          # the misleading figure
    assert airborne.error_at_envelope_m < ground.error_at_envelope_m


def test_without_terrain_the_ground_camera_does_fine():
    """The obstacle is doing the work, not a thumb on the scale.

    Remove the treeline and the low pair performs like the high one. The
    product case rests on terrain, and this makes that falsifiable.
    """
    clear = measure(_scenario(mast=3.0, obstacles=[]), "red-1", _effector())
    assert clear.warning_s > 60.0
    assert clear.first_seen_range_m > 1400.0


# --- the numbers a procurement document needs -------------------------------

def test_dwell_is_reported_as_a_lower_bound_when_the_target_is_still_inside():
    report = measure(_scenario(mast=200.0), "red-1", _effector())
    assert report.still_inside is True
    assert report.dwell_s == pytest.approx(report.last_t - report.envelope_entry_t)


def test_dwell_closes_when_the_target_leaves_the_envelope():
    """A pass-through rather than an approach: entry and exit both measured."""
    scenario = _scenario(mast=200.0, obstacles=[])
    scenario.aircraft[0].path = straight_line([0.0, 1500.0, 50.0], [0.0, -30.0, 0.0])
    report = measure(scenario, "red-1", _effector())
    assert report.envelope_exit_t is not None
    assert report.still_inside is False
    assert 0.0 < report.dwell_s < report.last_t


def test_envelope_timings_come_from_truth_not_from_our_own_estimate():
    """Otherwise a bad fix could flatter the warning figure.

    Wrecking the sensing (5 degrees of bearing noise) must not move the
    envelope entry time, which is a property of the geometry.
    """
    good = measure(_scenario(mast=200.0), "red-1", _effector())
    noisy_scenario = _scenario(mast=200.0)
    noisy_scenario.sigma_deg = 5.0
    noisy = measure(noisy_scenario, "red-1", _effector())
    assert noisy.envelope_entry_t == pytest.approx(good.envelope_entry_t)
    assert noisy.error_p50 > good.error_p50          # but the accuracy collapses


def test_the_report_counts_failures_and_not_only_successes():
    report = measure(_scenario(mast=200.0), "red-1", _effector())
    assert report.ghost_frames == 0
    assert report.ghost_hostile_frames == 0
    assert report.friendly_ever_hostile is False
    assert report.prompts == 1
    assert report.peak_tracks <= 4


def test_a_run_across_seeds_is_stable():
    """One good seed is an anecdote."""
    warnings, declared, errors = [], [], []
    for seed in range(6):
        r = measure(_scenario(mast=200.0, seed=seed), "red-1", _effector())
        assert not r.friendly_ever_hostile, f"seed {seed}"
        # A ghost that appears for a moment and stays UNKNOWN is the gate
        # working. One that reaches HOSTILE is an effector aimed at nothing.
        assert r.ghost_hostile_frames == 0, f"seed {seed}"
        assert r.ghost_frames < 0.02 * len(r.errors_m), f"seed {seed}"
        warnings.append(r.warning_s)
        declared.append(r.declared_t)
        errors.append(r.error_at_envelope_m)
    assert min(warnings) > 60.0
    assert max(declared) < 10.0
    assert max(errors) < GHOST_RADIUS_M / 10.0


def test_the_demo_command_runs_and_prints_both_configurations(capsys):
    """The demonstration has to be reproducible by the user, not just by us."""
    from skykiller.cli import main
    assert main(["demo", "--speed-ms", "30.0", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "ground camera pair at 3 m" in out
    assert "tethered observers at 200 m" in out
    assert "no transmission" in out
    assert out.count("warning before envelope") == 2


def test_the_demo_command_accepts_clear_terrain():
    from skykiller.cli import main
    assert main(["demo", "--treeline-m", "0", "--speed-ms", "30.0"]) == 0
