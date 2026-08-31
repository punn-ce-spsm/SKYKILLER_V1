"""Cross-site association: which of A's contacts is which of B's.

Without this step `triangulate` fuses every ray in a frame into one position,
so a sky with two aircraft in it produces one target that is neither. The tests
that matter here are the ghost tests -- a crossed pairing intersects too, and
can report a *tighter* covariance than the real target it is made from.
"""

import math

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


def _bearing(net: SiteNetwork, site: str, target) -> dict:
    az, el = to_bearing(np.asarray(target, dtype=float) - net.sites[site].enu)
    return {"az": az, "el": el, "conf": 0.9}


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


# --- the solver swap: JV must agree with the enumeration it replaced ---------

def _matchings(n: int, m: int):
    """Every partial one-to-one pairing of n things with m things.

    This *was* `associate._matchings`, and it is the reason the JV rewrite can
    be trusted: it is slow but obviously correct, so it stays here as the oracle
    the fast path is checked against. Deleting it from the module and from the
    tests at the same time would have left the swap unverified.
    """
    import itertools
    out = []
    for k in range(min(n, m) + 1):
        for a_sel in itertools.combinations(range(n), k):
            for b_sel in itertools.permutations(range(m), k):
                out.append(list(zip(a_sel, b_sel)))
    return out


def _brute(cost, price):
    """(best score, best matching, second-best score) by enumeration."""
    n, m = cost.shape
    scored = sorted(
        (sum(cost[p] for p in match) + price * (n + m - 2 * len(match)), match)
        for match in _matchings(n, m))
    return scored[0][0], scored[0][1], scored[1][0]


@pytest.mark.parametrize("n,m", [(1, 1), (1, 3), (3, 1), (2, 2), (3, 3),
                                 (4, 3), (4, 4), (5, 4)])
def test_jv_matches_the_enumeration_it_replaced(n, m):
    """Same total, and the same pairing wherever the optimum is unique.

    Costs include infeasible cells, because that is what forced the padded
    matrix to use a finite sentinel -- `linear_sum_assignment` raises on a
    board it cannot assign at all, and one stray infinity would take a whole
    good frame with it.
    """
    from skykiller.associate import _pad, _second_best, _solve
    rng = np.random.default_rng(11)
    for trial in range(40):
        cost = rng.uniform(0.0, 10.0, size=(n, m))
        cost[rng.random(size=(n, m)) < 0.15] = np.inf   # unpairable combinations
        price = float(rng.uniform(1.0, 6.0))

        square, used = _pad(cost, price)
        assert used == price
        got_match, got_best = _solve(square, cost)
        got_second = _second_best(square, cost, got_match, got_best, price)

        want_best, want_match, want_second = _brute(cost, price)
        assert got_best == pytest.approx(want_best), f"trial {trial}"
        assert got_second == pytest.approx(want_second), f"trial {trial}"
        # The pairing itself, but only where the answer is not a coin toss:
        # equal-cost assignments are genuinely interchangeable and the two
        # solvers break that tie by different rules.
        if want_second > want_best + 1e-9:
            assert got_match == want_match, f"trial {trial}"


def test_a_frame_that_used_to_be_unsolvable_now_solves_inside_the_frame_budget():
    """Ten contacts a post is 234,662,231 partial matchings. Eight -- the old
    cap -- took 1.27 s against a 200 ms budget at 5 Hz, so brute force failed
    inside the stated requirement rather than at some far-off swarm limit."""
    import time
    net = _net(b_east=200.0)
    targets = [[-900.0 + 200.0 * k, 900.0 + 70.0 * k, 40.0 + 15.0 * k]
               for k in range(10)]
    dets = _frame(net, targets, noise=0.2, seed=1)
    t0 = time.perf_counter()
    got = associate(net, dets)
    elapsed = time.perf_counter() - t0
    assert got.n_targets == 10
    assert got.matching == [(k, k) for k in range(10)]
    assert elapsed < 0.05, f"{elapsed * 1e3:.0f} ms for a 10x10 frame"


def test_a_reference_geometry_too_poor_to_price_does_not_crash_the_frame():
    """`_gate_scale` prices an unpaired contact off the *first* contact at each
    post. Two nearly-parallel rays will not triangulate at all, so that price
    comes back infinite -- and an infinite price makes retiring any contact
    infinitely expensive, so the only finite assignments left are the ones that
    pair every contact with another. When the posts see different numbers of
    contacts -- one of them occluded, which is the whole reason the unpaired
    price exists -- no such assignment exists at all and
    `linear_sum_assignment` raises `ValueError: cost matrix is infeasible`.

    That is a crash in the middle of a live frame, not a degraded answer, and a
    100 m baseline runs out of parallax past ~2.9 km, so the frame is reachable.
    `_pad` substitutes the sentinel for the price instead.
    """
    net = _net()                       # 100 m baseline
    far, near, other = [0.0, 6000.0, 80.0], [-400.0, 900.0, 140.0], [500.0, 1100.0, 60.0]
    dets = _frame(net, [far, near])
    dets.append(Detection(src="A", **_bearing(net, "A", other)))   # B is occluded
    got = associate(net, dets)         # the assertion is that this returns
    assert _nearest(got.fixes, near) < 1e-6


def test_fixes_priced_off_an_unmeasurable_scale_are_never_confident():
    """What the sentinel substitution costs, and why it is still safe.

    Priced at the sentinel, pairing always beats retiring, so the solver takes
    any pairing that passed `triangulate`'s own miss gate -- including a wrong
    one. Measured on the frame below, it fuses A's third contact with B's first
    into a ghost at (96, 212, 78).

    The ghost cannot become a track, and that is not luck: `ambiguity_floor`
    comes from the same unmeasurable `_gate_scale`, so it is infinite too, while
    the margin is finite whenever anything was paired at all (every cell of the
    padded board is finite by construction). So `ambiguous` is True for every
    fix in such a frame, and `AirPicture.ingest(confident=False)` may update an
    established track but may not create one.
    """
    net = _net()
    far, near, other = [0.0, 6000.0, 80.0], [-400.0, 900.0, 140.0], [500.0, 1100.0, 60.0]
    dets = _frame(net, [far, near])
    dets.append(Detection(src="A", **_bearing(net, "A", other)))
    got = associate(net, dets)
    assert got.n_targets == 2, "the mis-pairing is expected; being fooled is not"
    assert _nearest(got.fixes, [96.4, 211.9, 78.3]) < 1.0       # the ghost
    assert math.isfinite(got.margin_m)
    assert got.ambiguous, "an unmeasurable price must never read as confident"
