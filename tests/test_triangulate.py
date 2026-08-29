"""Triangulation tests against hand-computed geometry.

The reference case is chosen so every number can be derived on paper:
two sites 100 m apart on the east axis, a target 100 m north and 50 m up,
placed symmetrically between them.

    A = (0, 0, 0)      target = (50, 100, 50)      B = (100, 0, 0)

From A the target lies at (50, 100, 50), slant range sqrt(15000) = 122.474 m,
so az = atan2(50, 100) = 26.565 deg and el = asin(50/122.474) = 24.095 deg.
From B it lies at (-50, 100, 50) -- the mirror image, az = -26.565 deg, which
normalises to 333.435 deg, and the same elevation.
"""

import math

import numpy as np
import pytest

from skykiller.triangulate import (
    DEFAULT_MISS_GATE_SIGMAS,
    Fix,
    Ray,
    max_separation_deg,
    to_bearing,
    triangulate,
    unit,
)

A = np.array([0.0, 0.0, 0.0])
B = np.array([100.0, 0.0, 0.0])
TARGET = np.array([50.0, 100.0, 50.0])

AZ_A, EL_A = 26.565051177, 24.094842552
AZ_B, EL_B = 333.434948823, 24.094842552


def _rays(sigma_deg: float = 0.5) -> list[Ray]:
    return [
        Ray(origin=A.copy(), az_deg=AZ_A, el_deg=EL_A, src="A", sigma_deg=sigma_deg),
        Ray(origin=B.copy(), az_deg=AZ_B, el_deg=EL_B, src="B", sigma_deg=sigma_deg),
    ]


# --- the bearing conventions the rest of the file rests on -------------------

def test_unit_vector_matches_enu_convention():
    # Due north, level: the north component is the whole vector.
    assert unit(0.0, 0.0) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    # Due east, level.
    assert unit(90.0, 0.0) == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    # Straight up: azimuth is irrelevant.
    assert unit(217.0, 90.0) == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


def test_to_bearing_inverts_unit():
    for az, el in [(0.0, 0.0), (90.0, 30.0), (333.435, 24.095), (181.0, -12.0)]:
        got_az, got_el = to_bearing(unit(az, el))
        assert got_az == pytest.approx(az, abs=1e-9)
        assert got_el == pytest.approx(el, abs=1e-9)


def test_hand_computed_bearings_do_point_at_the_target():
    # If this fails, every expected number below is built on sand.
    for origin, az, el in [(A, AZ_A, EL_A), (B, AZ_B, EL_B)]:
        v = TARGET - origin
        want_az, want_el = to_bearing(v)
        assert want_az == pytest.approx(az, abs=1e-6)
        assert want_el == pytest.approx(el, abs=1e-6)


# --- the exit criterion: exact bearings recover the target ------------------

def test_exact_bearings_recover_the_target():
    fix = triangulate(_rays())
    assert fix is not None
    assert np.linalg.norm(fix.enu - TARGET) < 1e-6      # far inside the 1 m gate
    assert fix.miss_m == pytest.approx(0.0, abs=1e-9)
    assert fix.sites == ["A", "B"]


def test_three_sites_also_recover_the_target():
    # A third post north-east of both; over-determined, still exact.
    c = np.array([0.0, 200.0, 10.0])
    az_c, el_c = to_bearing(TARGET - c)
    rays = _rays() + [Ray(origin=c, az_deg=az_c, el_deg=el_c, src="C")]
    fix = triangulate(rays)
    assert fix is not None
    assert np.linalg.norm(fix.enu - TARGET) < 1e-6
    assert fix.sites == ["A", "B", "C"]


def test_a_moved_target_is_recovered_too():
    # Guards against a solver that happens to return the baseline midpoint.
    for truth in [
        np.array([-300.0, 800.0, 90.0]),
        np.array([420.0, 1500.0, 120.0]),
        np.array([50.0, 100.0, -20.0]),   # below the datum plane
    ]:
        rays = []
        for origin, name in [(A, "A"), (B, "B")]:
            az, el = to_bearing(truth - origin)
            rays.append(Ray(origin=origin.copy(), az_deg=az, el_deg=el, src=name))
        fix = triangulate(rays, miss_gate_sigmas=1e6)
        assert fix is not None, truth
        assert np.linalg.norm(fix.enu - truth) < 1e-6, truth


# --- refusals: report nothing rather than a confident fiction ---------------

def test_one_ray_is_not_a_fix():
    assert triangulate(_rays()[:1]) is None
    assert triangulate([]) is None


def test_parallel_rays_are_refused():
    # Both sites looking due north: the rays never converge.
    rays = [
        Ray(origin=A.copy(), az_deg=0.0, el_deg=10.0, src="A"),
        Ray(origin=B.copy(), az_deg=0.0, el_deg=10.0, src="B"),
    ]
    assert max_separation_deg(rays) == pytest.approx(0.0, abs=1e-5)
    assert triangulate(rays) is None


def test_near_parallel_rays_are_refused_by_the_separation_gate():
    # A distant target subtends almost nothing across a 100 m baseline.
    far = np.array([0.0, 200_000.0, 0.0])
    rays = []
    for origin, name in [(A, "A"), (B, "B")]:
        az, el = to_bearing(far - origin)
        rays.append(Ray(origin=origin.copy(), az_deg=az, el_deg=el, src=name))
    assert max_separation_deg(rays) < 2.0
    assert triangulate(rays) is None


def test_elevation_disagreement_is_refused():
    """The mis-association the miss gate can actually see."""
    rays = _rays()
    rays[1].el_deg = EL_B + 20.0
    assert triangulate(rays, miss_gate_sigmas=DEFAULT_MISS_GATE_SIGMAS) is None
    # With the gate opened the same geometry does produce a fix -- so the
    # refusal above is the gate, not a solver failure.
    assert triangulate(rays, miss_gate_sigmas=1e6) is not None


def test_azimuth_disagreement_is_NOT_caught_by_two_rays():
    """The limitation, pinned so it cannot change silently.

    This is geometry, not a bug: two rays that cross produce a zero miss
    wherever they cross, so an in-plane disagreement moves the answer without
    opening the rays apart. Measured here: a 20 deg azimuth error puts the fix
    75 m from truth and still passes the gate, while the same error in
    elevation (test above) is rejected.
    """
    rays = _rays()
    rays[1].az_deg = (AZ_B + 20.0) % 360.0
    fix = triangulate(rays, miss_gate_sigmas=DEFAULT_MISS_GATE_SIGMAS)
    assert fix is not None                                   # it gets through
    assert np.linalg.norm(fix.enu - TARGET) > 50.0           # and it is wrong
    assert fix.sigma_m < 10.0                                # and looks confident


def test_a_third_site_catches_what_two_cannot():
    """Why the answer to the above is a third bearing, not a cleverer gate."""
    c = np.array([0.0, 200.0, 10.0])
    az_c, el_c = to_bearing(TARGET - c)
    rays = _rays() + [Ray(origin=c, az_deg=az_c, el_deg=el_c, src="C")]
    rays[1].az_deg = (AZ_B + 20.0) % 360.0
    assert triangulate(rays, miss_gate_sigmas=DEFAULT_MISS_GATE_SIGMAS) is None


def test_miss_distance_grows_with_disagreement():
    misses = []
    for offset in (0.0, 1.0, 5.0):
        rays = _rays()
        rays[1].el_deg = EL_B + offset
        fix = triangulate(rays, miss_gate_sigmas=1e6)
        assert fix is not None
        misses.append(fix.miss_m)
    assert misses[0] < misses[1] < misses[2]


def test_the_gate_scales_with_range_rather_than_being_a_fixed_distance():
    """A fixed metre gate would be far too loose near and too tight far away.

    Same angular disagreement, two ranges an order of magnitude apart: both are
    rejected, which a single metre threshold could not manage.
    """
    for truth in [np.array([0.0, 200.0, 30.0]), np.array([0.0, 2000.0, 300.0])]:
        rays = []
        for origin, name in [(A, "A"), (B, "B")]:
            az, el = to_bearing(truth - origin)
            rays.append(Ray(origin=origin.copy(), az_deg=az, el_deg=el, src=name))
        assert triangulate(rays) is not None, truth        # clean fix passes
        rays[1].el_deg += 5.0                              # 5 deg out of plane
        assert triangulate(rays) is None, truth            # and is rejected


# --- covariance: does the reported error match the measured spread? ---------

def _monte_carlo(sigma_deg: float, n: int, seed: int) -> np.ndarray:
    """Errors from n noisy fixes of the same reference geometry."""
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(n):
        rays = [
            Ray(origin=A.copy(), az_deg=AZ_A + rng.normal(0, sigma_deg),
                el_deg=EL_A + rng.normal(0, sigma_deg), src="A", sigma_deg=sigma_deg),
            Ray(origin=B.copy(), az_deg=AZ_B + rng.normal(0, sigma_deg),
                el_deg=EL_B + rng.normal(0, sigma_deg), src="B", sigma_deg=sigma_deg),
        ]
        fix = triangulate(rays, miss_gate_sigmas=1e6)
        assert fix is not None
        errors.append(fix.enu - TARGET)
    return np.array(errors)


def test_reported_covariance_matches_the_measured_spread():
    """The exit criterion for phase A.

    The predicted covariance is the inverse information matrix; the measured
    one is the sample covariance of 2000 noisy fixes. They should agree to
    within Monte-Carlo scatter -- a factor of 1.5 either way is generous but
    still catches the failure that matters, a covariance wrong by an order of
    magnitude or in the wrong direction.
    """
    sigma_deg = 0.5
    errors = _monte_carlo(sigma_deg, n=2000, seed=7)

    predicted = triangulate(_rays(sigma_deg), miss_gate_sigmas=1e6).cov
    measured = np.cov(errors.T)

    for axis in range(3):
        ratio = math.sqrt(measured[axis, axis] / predicted[axis, axis])
        assert 1 / 1.5 < ratio < 1.5, (axis, ratio)

    # The error is an ellipsoid, not a ball: along the baseline (east) it is
    # tighter than down-range (north), because that is where the parallax is.
    assert measured[1, 1] > measured[0, 0]


def test_the_fix_is_unbiased():
    errors = _monte_carlo(0.5, n=2000, seed=11)
    mean = errors.mean(axis=0)
    sem = errors.std(axis=0) / math.sqrt(len(errors))
    for axis in range(3):
        assert abs(mean[axis]) < 4 * sem[axis], (axis, mean[axis], sem[axis])


def test_sigma_m_and_ellipse_axes_agree_with_the_covariance():
    fix = triangulate(_rays(), miss_gate_sigmas=1e6)
    axes = fix.ellipse_axes_m
    assert axes[0] <= axes[1] <= axes[2]                 # ascending
    assert fix.sigma_m == pytest.approx(axes[2])         # worst axis
    assert fix.sigma_m ** 2 == pytest.approx(np.linalg.eigvalsh(fix.cov).max())


def test_error_scales_linearly_with_bearing_noise():
    a = triangulate(_rays(0.5), miss_gate_sigmas=1e6).sigma_m
    b = triangulate(_rays(1.0), miss_gate_sigmas=1e6).sigma_m
    assert b == pytest.approx(2 * a, rel=1e-9)


def test_a_more_accurate_site_pulls_the_answer_toward_itself():
    # B is deliberately wrong by 2 degrees. When B is the sloppy sensor the
    # solution should sit nearer A's ray than when the two are trusted equally.
    def miss_from_a(sigma_b: float) -> float:
        rays = _rays()
        rays[1].az_deg = (AZ_B + 2.0) % 360.0
        rays[1].sigma_deg = sigma_b
        fix = triangulate(rays, miss_gate_sigmas=1e6)
        ray_a = rays[0]
        v = fix.enu - ray_a.origin
        along = float(np.dot(v, ray_a.direction))
        return float(np.linalg.norm(v - along * ray_a.direction))

    assert miss_from_a(sigma_b=5.0) < miss_from_a(sigma_b=0.5)


def test_covariance_is_conservative_by_one_over_cos_elevation():
    """Pins the known, deliberate approximation in the information matrix.

    Reported sigma is high by ~1/cos(el) because the azimuth term is treated as
    isotropic. Checked at two elevations: near the horizon, where the product
    actually operates, it is within a couple of percent; steeply overhead it is
    visibly conservative. Conservative is the safe direction -- this number
    gates an effector -- but it must never flip to optimistic.
    """
    for z, tol in [(0.0, 0.04), (260.0, 0.60)]:
        truth = np.array([50.0, 100.0, z])
        az_a, el_a = to_bearing(truth - A)
        az_b, el_b = to_bearing(truth - B)
        rng = np.random.default_rng(19)
        errors = []
        for _ in range(2000):
            rays = [
                Ray(A.copy(), az_a + rng.normal(0, 0.5), el_a + rng.normal(0, 0.5),
                    "A", sigma_deg=0.5),
                Ray(B.copy(), az_b + rng.normal(0, 0.5), el_b + rng.normal(0, 0.5),
                    "B", sigma_deg=0.5),
            ]
            errors.append(triangulate(rays, miss_gate_sigmas=1e6).enu - truth)
        measured = np.cov(np.array(errors).T)
        predicted = triangulate(
            [Ray(A.copy(), az_a, el_a, "A"), Ray(B.copy(), az_b, el_b, "B")],
            miss_gate_sigmas=1e6).cov
        ratio = np.mean([math.sqrt(measured[i, i] / predicted[i, i]) for i in range(3)])
        assert ratio <= 1.0 + tol, (z, ratio)   # never optimistic
        assert ratio > 0.3, (z, ratio)          # nor absurdly pessimistic
