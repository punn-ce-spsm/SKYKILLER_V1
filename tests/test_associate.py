"""Cross-site association: which of A's contacts is which of B's.

Without this step `triangulate` fuses every ray in a frame into one position,
so a sky with two aircraft in it produces one target that is neither. The tests
that matter here are the ghost tests -- a crossed pairing intersects too, and
can report a *tighter* covariance than the real target it is made from.
"""

import numpy as np
import pytest

from skykiller.associate import MAX_CONTACTS_PER_SITE, associate
from skykiller.geometry import Camera
from skykiller.schemas import Detection
from skykiller.sites import Site, SiteNetwork
from skykiller.triangulate import to_bearing

CAM = Camera(width=1280, height=720, hfov_deg=65.0)


def _net(b_east: float = 100.0) -> SiteNetwork:
    return SiteNetwork({"A": Site("A", np.array([0.0, 0.0, 80.0]), CAM),
                        "B": Site("B", np.array([b_east, 0.0, 80.0]), CAM)})


def _frame(net: SiteNetwork, targets, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    dets = []
    for name, site in net.sites.items():
        for t in targets:
            az, el = to_bearing(np.asarray(t, dtype=float) - site.enu)
            dets.append(Detection(src=name, az=az + rng.normal(0, noise),
                                  el=el + rng.normal(0, noise), conf=0.9))
    return dets


def _nearest(fixes, truth) -> float:
    return min(float(np.linalg.norm(f.enu - np.asarray(truth))) for f in fixes)


# --- the case that motivated the file ---------------------------------------

def test_two_aircraft_produce_two_targets_not_one_average():
    """Without this step triangulate() fuses all four rays into one position."""
    net = _net(b_east=200.0)
    t1, t2 = [-400.0, 1000.0, 40.0], [400.0, 1400.0, 160.0]
    got = associate(net, _frame(net, [t1, t2]))
    assert got.n_targets == 2
    assert _nearest(got.fixes, t1) < 1e-6
    assert _nearest(got.fixes, t2) < 1e-6
    assert got.matching == [(0, 0), (1, 1)]
    assert not got.ambiguous


def test_the_crossed_pairing_is_not_chosen():
    """The ghost is a real intersection, so it has to be beaten, not filtered."""
    net = _net()
    t1, t2 = [0.0, 1000.0, 80.0], [20.0, 1600.0, 90.0]
    got = associate(net, _frame(net, [t1, t2]))
    assert got.n_targets == 2
    for truth in (t1, t2):
        assert _nearest(got.fixes, truth) < 1e-6
    # And the known ghost positions are absent.
    for ghost in ([1.0, 1969.0, 86.0], [11.0, 886.0, 83.0]):
        assert _nearest(got.fixes, ghost) > 50.0


def test_one_aircraft_still_works():
    net = _net()
    truth = [0.0, 1200.0, 80.0]
    got = associate(net, _frame(net, [truth]))
    assert got.n_targets == 1
    assert _nearest(got.fixes, truth) < 1e-6
    assert got.unpaired == []


def test_three_aircraft_all_recovered():
    net = _net()
    targets = [[-500.0, 900.0, 60.0], [0.0, 1300.0, 110.0], [600.0, 1100.0, 90.0]]
    got = associate(net, _frame(net, targets))
    assert got.n_targets == 3
    for t in targets:
        assert _nearest(got.fixes, t) < 1e-6


# --- what it refuses to pretend about ---------------------------------------

def test_targets_level_with_the_masts_are_exactly_degenerate_and_are_flagged():
    """The deployment hazard, and the reason it is not a corner case.

    With the masts at 80 m and both targets at 80 m, every ray lies in one
    horizontal plane. Every pairing then intersects *exactly*, the crossed
    assignment scores the same zero as the true one, and the winner is decided
    by floating-point noise. A tethered observer sits at 80-120 m and hunts
    drones flying at much the same height, so this is the expected geometry.

    The code cannot resolve it -- nothing can, from two posts -- but it must
    say so rather than present a coin toss as an answer.
    """
    net = _net()
    dets = _frame(net, [[0.0, 1000.0, 80.0], [20.0, 1600.0, 80.0]])
    assert all(d.el == pytest.approx(0.0) for d in dets), "rays must be coplanar"
    got = associate(net, dets)
    assert got.margin_m < 1e-6           # the two assignments are tied
    assert got.ambiguous                 # and it says so


def test_closely_spaced_same_altitude_targets_are_flagged_ambiguous():
    """The same degeneracy with noise on: still flagged, not silently resolved."""
    net = _net()
    got = associate(net, _frame(net, [[0.0, 1000.0, 80.0], [20.0, 1600.0, 80.0]],
                                noise=0.5, seed=3))
    assert got.ambiguous


def test_a_well_separated_frame_is_not_flagged_ambiguous():
    """The flag has to mean something -- it must not fire on every frame.

    The first threshold tried here reused the unpaired price and called 90-99%
    of frames ambiguous, which is a flag that tells an operator nothing.
    """
    net = _net(b_east=200.0)
    got = associate(net, _frame(net, [[-600.0, 900.0, 40.0], [600.0, 1400.0, 200.0]],
                                noise=0.5, seed=5))
    assert not got.ambiguous
    assert got.n_targets == 2


def test_a_contact_only_one_post_can_see_is_reported_not_invented():
    """An occluded or false-positive contact must not be forced into a pairing."""
    net = _net()
    dets = _frame(net, [[0.0, 1000.0, 80.0]])
    dets.append(Detection(src="A", az=270.0, el=5.0, conf=0.9))   # behind, A only
    got = associate(net, dets)
    assert got.n_targets == 1
    assert _nearest(got.fixes, [0.0, 1000.0, 80.0]) < 1e-6
    assert [d.az for d in got.unpaired] == [270.0]


def test_detections_from_one_post_alone_give_no_position():
    net = _net()
    dets = [d for d in _frame(net, [[0.0, 1000.0, 80.0]]) if d.src == "A"]
    got = associate(net, dets)
    assert got.fixes == []
    assert len(got.unpaired) == 1


def test_an_unknown_site_never_becomes_a_target():
    net = _net()
    dets = _frame(net, [[0.0, 1000.0, 80.0]])
    dets.append(Detection(src="L3", az=45.0, el=10.0, conf=0.9))
    got = associate(net, dets)
    assert got.n_targets == 1
    assert [d.src for d in got.unpaired] == ["L3"]


def test_a_third_post_is_refused_rather_than_silently_mishandled():
    net = SiteNetwork({"A": Site("A", np.array([0.0, 0.0, 80.0]), CAM),
                       "B": Site("B", np.array([100.0, 0.0, 80.0]), CAM),
                       "C": Site("C", np.array([0.0, 200.0, 80.0]), CAM)})
    with pytest.raises(NotImplementedError, match="two posts"):
        associate(net, _frame(net, [[0.0, 1000.0, 80.0]]))


def test_a_swarm_is_refused_rather_than_hanging_on_factorial_growth():
    net = _net()
    targets = [[float(n) * 200 - 900, 1000.0 + 40 * n, 80.0]
               for n in range(MAX_CONTACTS_PER_SITE + 1)]
    with pytest.raises(NotImplementedError, match="capped"):
        associate(net, _frame(net, targets))


def test_an_empty_frame_is_empty_not_an_error():
    got = associate(_net(), [])
    assert got.fixes == [] and got.unpaired == [] and not got.ambiguous


# --- under noise ------------------------------------------------------------

def test_well_separated_targets_stay_correctly_paired_under_realistic_noise():
    """Checked on the pairing itself, not on where the answer landed.

    Position error and pairing error are different failures and the first
    attempt here conflated them: at 1.4 km a 200 m baseline is honestly worth
    ~85 m of down-range error, so a position-distance check fails on correctly
    paired frames and hides what it was meant to measure.
    """
    net = _net(b_east=200.0)
    t1, t2 = [-500.0, 1000.0, 40.0], [500.0, 1400.0, 180.0]
    correct = sum(associate(net, _frame(net, [t1, t2], noise=0.5, seed=s)).matching
                  == [(0, 0), (1, 1)] for s in range(60))
    assert correct >= 57, f"{correct}/60 frames paired correctly"


def test_a_confident_verdict_is_worth_more_than_an_ambiguous_one():
    """The flag must actually predict correctness, or it is decoration."""
    # A deliberately marginal geometry -- a 100 m baseline with the targets
    # only 30 m apart in height -- so that both verdicts actually occur.
    net = _net()
    t1, t2 = [0.0, 1000.0, 60.0], [200.0, 1500.0, 90.0]
    conf_right = conf_n = amb_right = amb_n = 0
    for seed in range(300):
        got = associate(net, _frame(net, [t1, t2], noise=0.5, seed=seed))
        right = got.matching == [(0, 0), (1, 1)]
        if got.ambiguous:
            amb_n += 1
            amb_right += right
        else:
            conf_n += 1
            conf_right += right
    assert conf_n > 0 and amb_n > 0, "geometry must produce both verdicts"
    assert conf_right / conf_n > amb_right / amb_n
    assert conf_right / conf_n > 0.9
