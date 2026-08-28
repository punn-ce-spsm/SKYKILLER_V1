"""Geometry tests against hand-computed values.

Every expected number here is derived on paper from the pinhole model, not
captured from a previous run of the code. A test that records what the code
already does cannot tell you the code is wrong.
"""

import math

import pytest

from skykiller.geometry import Camera, hfov_from_reference

# A 1280x720 frame at 65 degrees horizontal -- the laptop webcam case.
WEBCAM = Camera(width=1280, height=720, hfov_deg=65.0)


def test_focal_length_matches_hand_calculation():
    # f = (W/2) / tan(HFOV/2) = 640 / tan(32.5 deg)
    assert WEBCAM.focal_px == pytest.approx(640.0 / math.tan(math.radians(32.5)))
    assert WEBCAM.focal_px == pytest.approx(1004.60, abs=0.01)


def test_frame_centre_is_the_bore():
    az, el = WEBCAM.bearing(640.0, 360.0)
    assert az == pytest.approx(0.0)
    assert el == pytest.approx(0.0)


def test_horizontal_edges_are_half_the_declared_fov():
    az_right, el_right = WEBCAM.bearing(1280.0, 360.0)
    assert az_right == pytest.approx(32.5)
    assert el_right == pytest.approx(0.0)

    # Left edge wraps to just under 360, not to -32.5.
    az_left, _ = WEBCAM.bearing(0.0, 360.0)
    assert az_left == pytest.approx(327.5)


def test_vertical_edge_is_half_the_derived_vfov():
    # VFOV is not given; it follows from the same focal length and the height.
    expected = WEBCAM.vfov_deg / 2.0
    _, el_top = WEBCAM.bearing(640.0, 0.0)
    assert el_top == pytest.approx(expected)
    assert expected == pytest.approx(math.degrees(math.atan(360.0 / WEBCAM.focal_px)))

    _, el_bottom = WEBCAM.bearing(640.0, 720.0)
    assert el_bottom == pytest.approx(-expected)


def test_elevation_shrinks_toward_the_corners():
    """The bug this catches: computing el as atan(-dy/f), ignoring dx.

    A corner pixel is further from the optical axis than a top-edge pixel, so
    its ray is more oblique and its true elevation is *lower*. The naive form
    returns the same elevation for both and is wrong by degrees at the edges.
    """
    _, el_top_centre = WEBCAM.bearing(640.0, 0.0)
    _, el_top_corner = WEBCAM.bearing(1280.0, 0.0)
    assert el_top_corner < el_top_centre

    # Exact value for the corner: atan(-dy / sqrt(dx^2 + f^2)).
    f = WEBCAM.focal_px
    expected = math.degrees(math.atan(360.0 / math.sqrt(640.0**2 + f**2)))
    assert el_top_corner == pytest.approx(expected)


def test_camera_pose_rotates_the_whole_frame():
    """A camera bolted facing east reports east at frame centre."""
    east = Camera(width=1280, height=720, hfov_deg=65.0, az_deg=90.0)
    az, el = east.bearing(640.0, 360.0)
    assert az == pytest.approx(90.0)
    assert el == pytest.approx(0.0)

    # Right edge is 32.5 further clockwise.
    az_right, _ = east.bearing(1280.0, 360.0)
    assert az_right == pytest.approx(122.5)


def test_tilted_camera_adds_elevation_at_the_bore():
    up = Camera(width=1280, height=720, hfov_deg=65.0, el_deg=30.0)
    az, el = up.bearing(640.0, 360.0)
    assert az == pytest.approx(0.0)
    assert el == pytest.approx(30.0)


def test_azimuth_wraps_across_north():
    """A camera bored at 350 must report 10, not 370."""
    cam = Camera(width=1280, height=720, hfov_deg=65.0, az_deg=350.0)
    az, _ = cam.bearing(1280.0, 360.0)  # +32.5 from 350
    assert az == pytest.approx(22.5)


def test_range_from_width_inverts_correctly():
    """A Mini 4 Pro is ~0.35 m across. Check the round trip."""
    cam = WEBCAM
    target_m = 0.35
    true_range = 20.0
    # Forward: how many pixels wide would it be at 20 m?
    width_px = target_m * cam.focal_px / true_range
    assert cam.range_from_width(width_px, target_m) == pytest.approx(true_range)
    # Sanity on the absolute number: ~17.6 px at 20 m on this webcam.
    assert width_px == pytest.approx(17.57, abs=0.05)


def test_range_from_width_rejects_degenerate_boxes():
    assert WEBCAM.range_from_width(0.0, 0.35) is None
    assert WEBCAM.range_from_width(-5.0, 0.35) is None
    assert WEBCAM.range_from_width(20.0, 0.0) is None


def test_hfov_from_reference_recovers_a_known_fov():
    """Photograph a 1 m object at 5 m; check we get the FOV we started with."""
    cam = WEBCAM
    object_m, distance_m = 1.0, 5.0
    object_px = object_m * cam.focal_px / distance_m
    recovered = hfov_from_reference(cam.width, object_px, object_m, distance_m)
    assert recovered == pytest.approx(cam.hfov_deg)


def test_hfov_from_reference_rejects_nonsense():
    with pytest.raises(ValueError):
        hfov_from_reference(1280, 0.0, 1.0, 5.0)
    with pytest.raises(ValueError):
        hfov_from_reference(1280, 100.0, 1.0, -5.0)


def test_camera_rejects_impossible_construction():
    with pytest.raises(ValueError):
        Camera(width=0, height=720, hfov_deg=65.0)
    with pytest.raises(ValueError):
        Camera(width=1280, height=720, hfov_deg=200.0)
