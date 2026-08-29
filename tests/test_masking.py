"""Terrain masking: the argument for putting the observer in the air.

The exit criterion for phase D is the first test -- the model has to reproduce
the table the whole product case was built on, and it has to do it from the
general screen geometry rather than from the closed form that generated the
table, or it proves nothing.
"""

import math

import numpy as np
import pytest

from skykiller.masking import Obstacle, lowest_visible_alt, treeline, visible

GROUND_CAMERA = np.array([0.0, 0.0, 3.0])


# --- the phase D exit criterion ---------------------------------------------

@pytest.mark.parametrize("height,distance,range_m,expected", [
    (20.0, 200.0, 1000.0, 88.0),
    (20.0, 200.0, 2000.0, 173.0),
    (20.0, 500.0, 1000.0, 37.0),
    (20.0, 500.0, 2000.0, 71.0),
    (60.0, 1000.0, 2000.0, 117.0),
])
def test_the_model_reproduces_the_product_case(height, distance, range_m, expected):
    """Each number here is why the observer is airborne rather than on a mast.

    Derived independently: the sight line from a camera at h_c grazing a crest
    h_o at d_o reaches altitude h_c + (h_o - h_c) * R / d_o at range R. The
    model computes it by intersecting a screen segment in three dimensions and
    must agree.
    """
    obstacles = [treeline("t", height, distance)]
    got = lowest_visible_alt(GROUND_CAMERA, (0.0, range_m), obstacles)
    assert got == pytest.approx(expected, abs=0.5)
    # And the closed form, so the expectation is not just a recorded output.
    assert got == pytest.approx(3.0 + (height - 3.0) * range_m / distance, abs=0.5)


def test_a_drone_at_50m_is_invisible_to_a_ground_camera_past_600m():
    """The specific claim in the plan, checked."""
    obstacles = [treeline("treeline", 20.0, 200.0)]
    assert visible(GROUND_CAMERA, np.array([0.0, 500.0, 50.0]), obstacles)
    assert not visible(GROUND_CAMERA, np.array([0.0, 700.0, 50.0]), obstacles)
    assert not visible(GROUND_CAMERA, np.array([0.0, 1000.0, 50.0]), obstacles)


def test_height_is_the_answer_and_a_longer_lens_is_not():
    """An elevated observer sees the same target the ground camera cannot.

    No sensor change: the obstacle and the target are identical, only the
    observer's altitude differs.
    """
    obstacles = [treeline("treeline", 20.0, 200.0)]
    target = np.array([0.0, 1500.0, 50.0])
    assert not visible(np.array([0.0, 0.0, 3.0]), target, obstacles)
    assert visible(np.array([0.0, 0.0, 80.0]), target, obstacles)
    assert visible(np.array([0.0, 0.0, 200.0]), target, obstacles)


# --- the screen geometry ----------------------------------------------------

def test_a_target_above_the_crest_line_is_visible():
    obstacles = [treeline("t", 20.0, 200.0)]
    assert visible(GROUND_CAMERA, np.array([0.0, 1000.0, 200.0]), obstacles)
    assert not visible(GROUND_CAMERA, np.array([0.0, 1000.0, 50.0]), obstacles)


def test_an_obstacle_the_sight_line_misses_does_not_block():
    """A screen of finite width blocks only what crosses it."""
    narrow = treeline("narrow", 20.0, 200.0, width_m=100.0)
    assert not visible(GROUND_CAMERA, np.array([0.0, 1000.0, 50.0]), [narrow])
    # 400 m east at 1 km passes well outside a 100 m wide screen.
    assert visible(GROUND_CAMERA, np.array([400.0, 1000.0, 50.0]), [narrow])


def test_an_obstacle_behind_the_target_does_not_block_it():
    obstacles = [treeline("far", 60.0, 900.0)]
    assert visible(GROUND_CAMERA, np.array([0.0, 500.0, 20.0]), obstacles)
    assert not visible(GROUND_CAMERA, np.array([0.0, 1500.0, 20.0]), obstacles)


def test_a_target_behind_the_observer_is_not_blocked_by_a_screen_in_front():
    obstacles = [treeline("t", 20.0, 200.0)]
    assert visible(GROUND_CAMERA, np.array([0.0, -1000.0, 50.0]), obstacles)


def test_the_worst_of_several_screens_sets_the_floor():
    obstacles = [treeline("near", 20.0, 200.0), treeline("far", 60.0, 1000.0)]
    # At 2 km: the near treeline imposes 173 m, the ridge 117 m. The higher wins.
    assert lowest_visible_alt(GROUND_CAMERA, (0.0, 2000.0), obstacles) == pytest.approx(
        173.0, abs=0.5)


def test_no_obstacles_means_no_floor():
    assert lowest_visible_alt(GROUND_CAMERA, (0.0, 1000.0), []) == -math.inf
    assert visible(GROUND_CAMERA, np.array([0.0, 1000.0, 1.0]), [])


def test_a_bearing_offset_screen_blocks_the_bearing_it_is_on():
    east = treeline("east", 30.0, 300.0, bearing_deg=90.0, width_m=400.0)
    assert not visible(GROUND_CAMERA, np.array([1000.0, 0.0, 50.0]), [east])
    assert visible(GROUND_CAMERA, np.array([0.0, 1000.0, 50.0]), [east])


def test_a_degenerate_obstacle_is_refused():
    with pytest.raises(ValueError, match="same point"):
        Obstacle(name="bad", a=np.array([1.0, 1.0]), b=np.array([1.0, 1.0]),
                 crest_m=20.0)
