"""Site network tests: the bus-to-solver bridge, and deployment sizing."""

import math

import numpy as np
import pytest

from skykiller.geometry import Camera
from skykiller.schemas import Detection
from skykiller.sites import DEFAULT_SIGMA_DEG, Site, SiteNetwork
from skykiller.triangulate import to_bearing, triangulate

CAM = Camera(width=1280, height=720, hfov_deg=65.0)


def _net(a=(0.0, 0.0, 80.0), b=(100.0, 0.0, 80.0)) -> SiteNetwork:
    return SiteNetwork({"A": Site("A", np.array(a), CAM),
                        "B": Site("B", np.array(b), CAM)})


def test_a_site_rejects_a_position_that_is_not_a_position():
    with pytest.raises(ValueError, match="3 numbers"):
        Site("A", np.array([0.0, 0.0]), CAM)
    with pytest.raises(ValueError, match="sigma_deg"):
        Site("A", np.array([0.0, 0.0, 0.0]), CAM, sigma_deg=0.0)


def test_height_is_read_off_the_survey():
    assert Site("A", np.array([10.0, 20.0, 95.0]), CAM).height_m == 95.0


def test_coincident_sites_are_refused_at_construction():
    with pytest.raises(ValueError, match="cannot triangulate"):
        SiteNetwork({"A": Site("A", np.array([0.0, 0.0, 80.0]), CAM),
                     "B": Site("B", np.array([0.5, 0.0, 80.0]), CAM)})


def test_one_site_is_allowed_but_gives_no_baseline():
    solo = SiteNetwork({"A": Site("A", np.array([0.0, 0.0, 80.0]), CAM)})
    assert solo.baseline_m == 0.0
    assert solo.expected_accuracy_m(1000.0) == float("inf")


def test_a_detection_becomes_a_ray_from_the_right_place():
    net = _net()
    det = Detection(src="B", az=12.0, el=-3.0, conf=0.9)
    ray = net.sites["B"].ray(det)
    assert ray.origin == pytest.approx([100.0, 0.0, 80.0])
    assert (ray.az_deg, ray.el_deg) == (12.0, -3.0)
    assert ray.src == "B"
    assert ray.sigma_deg == DEFAULT_SIGMA_DEG


def test_detections_from_unknown_sites_are_dropped_and_reported():
    net = _net()
    dets = [Detection(src="A", az=1.0, el=1.0, conf=0.9),
            Detection(src="L3", az=2.0, el=2.0, conf=0.9),   # the acoustic lane
            Detection(src="B", az=3.0, el=3.0, conf=0.9)]
    assert [r.src for r in net.rays(dets)] == ["A", "B"]
    assert net.unknown_sources(dets) == {"L3"}


def test_the_bus_round_trip_recovers_a_known_target():
    """The whole point of the file: two Detections in, one position out."""
    net = _net()
    truth = np.array([300.0, 900.0, 120.0])
    dets = []
    for name, site in net.sites.items():
        az, el = to_bearing(truth - site.enu)
        dets.append(Detection(src=name, az=az, el=el, conf=0.9))
    fix = triangulate(net.rays(dets))
    assert fix is not None
    assert np.linalg.norm(fix.enu - truth) < 1e-6
    assert sorted(fix.sites) == ["A", "B"]


def test_the_ray_origin_is_a_copy_not_the_site_itself():
    # A solver that shifted a ray origin must not silently move the mast.
    net = _net()
    ray = net.sites["A"].ray(Detection(src="A", az=0.0, el=0.0, conf=0.9))
    ray.origin += 1000.0
    assert net.sites["A"].enu == pytest.approx([0.0, 0.0, 80.0])


# --- deployment sizing ------------------------------------------------------

def test_the_accuracy_rule_of_thumb_matches_the_solver():
    """`expected_accuracy_m` is a planning shortcut; check it is not a fiction.

    Compares the closed form against the down-range spread the real solver
    produces under the same bearing noise. Agreement inside 20% is enough for
    a number whose job is to answer "how far apart should the masts be".
    """
    # 200 m baseline: at 100 m the separation gate starts refusing fixes near
    # 2 km, which is correct behaviour and tested separately below.
    net = _net(b=(200.0, 0.0, 80.0))
    rng = np.random.default_rng(23)
    for range_m in (500.0, 1000.0, 2000.0):
        truth = np.array([0.0, range_m, 80.0])
        bearings = [(name, *to_bearing(truth - s.enu)) for name, s in net.sites.items()]
        errors = []
        for _ in range(1500):
            dets = [Detection(src=n, az=az + rng.normal(0, DEFAULT_SIGMA_DEG),
                              el=el + rng.normal(0, DEFAULT_SIGMA_DEG), conf=0.9)
                    for n, az, el in bearings]
            fix = triangulate(net.rays(dets), miss_gate_sigmas=1e6)
            errors.append(abs(fix.enu[1] - truth[1]))   # down-range component
        measured = float(np.median(errors))
        predicted = net.expected_accuracy_m(range_m)
        assert 0.8 < measured / predicted < 1.2, (range_m, measured, predicted)


def test_error_grows_with_the_square_of_range_and_falls_with_baseline():
    net = _net()
    assert net.expected_accuracy_m(2000.0) == pytest.approx(
        4 * net.expected_accuracy_m(1000.0))
    wide = _net(b=(200.0, 0.0, 80.0))
    assert wide.expected_accuracy_m(1000.0) == pytest.approx(
        net.expected_accuracy_m(1000.0) / 2)


def test_accuracy_at_jammer_range_meets_the_beam_budget():
    """The requirement that actually has to be met, at the range it applies.

    Accuracy and detection range pull in opposite directions and both are
    satisfied, because they are needed at different ranges. At 2 km the fix is
    hundreds of metres out -- irrelevant, that range is early warning, where the
    question is only whether something is inbound. The number that has to be
    good is the one at the jammer's envelope edge, and there it is.

    Narrowest beam in the plan is a 10 deg dish: 87 m wide at 500 m, so
    +/-44 m of error is affordable.
    """
    net = _net()
    assert net.expected_accuracy_m(500.0) < 44.0
    # And the 2 km figure is genuinely poor -- documenting that this is known
    # and tolerated, not overlooked.
    assert net.expected_accuracy_m(2000.0) > 100.0


def test_baseline_sets_the_usable_range_not_the_camera():
    """The deployment lever, pinned with the numbers that justify it."""
    assert _net().max_range_m() == pytest.approx(100.0 / math.tan(math.radians(2.0)))
    # Doubling the baseline doubles the reach, linearly.
    assert _net(b=(200.0, 0.0, 80.0)).max_range_m() == pytest.approx(
        2 * _net().max_range_m())
    assert SiteNetwork({"A": Site("A", np.array([0.0, 0.0, 80.0]), CAM)}).max_range_m() == 0.0


def test_a_100m_baseline_actually_starts_refusing_before_a_200m_one_does():
    """Not a formula check -- run the real solver at 2 km and count."""
    rng = np.random.default_rng(31)

    def fix_rate(baseline: float) -> float:
        net = _net(b=(baseline, 0.0, 80.0))
        truth = np.array([0.0, 2000.0, 80.0])
        bearings = [(n, *to_bearing(truth - s.enu)) for n, s in net.sites.items()]
        got = 0
        for _ in range(400):
            dets = [Detection(src=n, az=az + rng.normal(0, 0.5),
                              el=el + rng.normal(0, 0.5), conf=0.9)
                    for n, az, el in bearings]
            got += triangulate(net.rays(dets), miss_gate_sigmas=1e6) is not None
        return got / 400

    assert fix_rate(100.0) < 0.98      # degrading at 2 km
    assert fix_rate(200.0) > 0.99      # comfortable at 2 km


def test_elevation_separation_is_what_makes_a_pairing_resolvable():
    """The siting number, checked against the scenario it came from.

    Two targets that subtend the same elevation from a post put all four rays
    in one plane, where the true and crossed cross-site pairings intersect
    equally well and nothing can separate them.
    """
    def net_at(mast: float) -> SiteNetwork:
        return SiteNetwork({"A": Site("A", np.array([0.0, 0.0, mast]), CAM),
                            "B": Site("B", np.array([200.0, 0.0, mast]), CAM)})

    hostile, friendly = [0.0, 1000.0, 60.0], [400.0, 300.0, 100.0]

    # Masts level with the traffic: separation below the 0.5 deg bearing noise,
    # so the pairing is decided by chance.
    assert net_at(120.0).elevation_separation_deg(hostile, friendly) < 0.5
    # Masts above it: several times the noise, and resolvable.
    assert net_at(200.0).elevation_separation_deg(hostile, friendly) > 3.0


def test_raising_the_masts_separates_targets_at_different_ranges():
    """The mechanism, stated so the recommendation is not folklore.

    Height helps because the *near* target gains depression angle faster than
    the far one. Two targets at the same range gain nothing from height, which
    is the case the recommendation does not cover.
    """
    def sep(mast: float, a, b) -> float:
        n = SiteNetwork({"A": Site("A", np.array([0.0, 0.0, mast]), CAM),
                         "B": Site("B", np.array([200.0, 0.0, mast]), CAM)})
        return n.elevation_separation_deg(a, b)

    near, far = [0.0, 300.0, 80.0], [0.0, 1500.0, 80.0]
    assert sep(200.0, near, far) > sep(80.0, near, far)

    # Same range, same altitude, different bearing: height changes nothing.
    left, right = [-400.0, 1000.0, 80.0], [400.0, 1000.0, 80.0]
    assert sep(200.0, left, right) == pytest.approx(sep(80.0, left, right), abs=1e-9)


def test_a_target_level_with_the_masts_has_zero_separation_from_another():
    net = SiteNetwork({"A": Site("A", np.array([0.0, 0.0, 80.0]), CAM),
                       "B": Site("B", np.array([100.0, 0.0, 80.0]), CAM)})
    assert net.elevation_separation_deg(
        [0.0, 1000.0, 80.0], [20.0, 1600.0, 80.0]) == pytest.approx(0.0, abs=1e-9)
