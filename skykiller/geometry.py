"""Pixel coordinates to bearing, for a camera of known field of view.

Build 1 uses a pinhole model with no lens-distortion term. With a hand-entered
field of view expect roughly +/-2-3 degrees of absolute error; the relative
motion of a track is far better than that, which is what the camera cue needs.
Run `python -m skykiller calibrate` to replace the guessed FOV with a measured
one, and see docs/engineering/L2-visual-tracking.md for what is still missing.

World frame is ENU: X east, Y north, Z up. Azimuth is degrees clockwise from
north; elevation is degrees above the horizon.

The camera pose is composed as a full rotation rather than by adding offsets to
the bore angles. That is the same arithmetic for a static camera, and it stays
correct when the pan-tilt head arrives and the camera starts looking up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Camera:
    """Intrinsics and mounting pose for one camera.

    `hfov_deg` is the horizontal field of view of the *full* frame. `az_deg` and
    `el_deg` are where the camera is bolted -- for build 1 the mount is static,
    so they are constants from config.
    """

    width: int
    height: int
    hfov_deg: float
    az_deg: float = 0.0
    el_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"frame size must be positive, got {self.width}x{self.height}")
        if not 0.0 < self.hfov_deg < 180.0:
            raise ValueError(f"hfov_deg must be in (0,180), got {self.hfov_deg}")

    @property
    def focal_px(self) -> float:
        """Focal length in pixels, derived from the horizontal field of view."""
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def vfov_deg(self) -> float:
        """Vertical field of view implied by the focal length and frame height."""
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.focal_px))

    def bearing(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Map a pixel to (azimuth, elevation) in degrees, in the world frame.

        Origin is the top-left of the frame, y increasing downward -- the
        convention OpenCV and Ultralytics both use.
        """
        f = self.focal_px
        dx = x_px - self.width / 2.0
        dy = y_px - self.height / 2.0

        a = math.radians(self.az_deg)
        e = math.radians(self.el_deg)

        # Camera basis in world coordinates.
        bore = (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))
        right = (math.cos(a), -math.sin(a), 0.0)
        up = (
            right[1] * bore[2] - right[2] * bore[1],
            right[2] * bore[0] - right[0] * bore[2],
            right[0] * bore[1] - right[1] * bore[0],
        )

        # Ray through the pixel: forward f, right dx, down dy.
        d = tuple(f * bore[i] + dx * right[i] - dy * up[i] for i in range(3))
        norm = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

        el = math.degrees(math.asin(d[2] / norm))
        az = math.degrees(math.atan2(d[0], d[1])) % 360.0
        return az, el

    def range_from_width(self, width_px: float, target_width_m: float) -> float | None:
        """Estimate range from apparent width, assuming a target of known size.

        This is only as true as the assumption. It is correct for the Mini 4 Pro
        the demo flies and wrong for anything else, so the caller decides whether
        to trust it -- see `l2.target_width_m` in the config, which is None by
        default. Returns None for a degenerate box.
        """
        if width_px <= 0 or target_width_m <= 0:
            return None
        return (target_width_m * self.focal_px) / width_px


def hfov_from_reference(width_px: int, object_px: float, object_m: float, distance_m: float) -> float:
    """Measured horizontal FOV from one photo of a known object at a known range.

    Put something of known width (a door, a metre rule) at a measured distance,
    photograph it, and measure how many pixels wide it lands. Exact, not a small
    angle approximation.
    """
    if min(object_px, object_m, distance_m) <= 0:
        raise ValueError("object_px, object_m and distance_m must all be positive")
    focal_px = object_px * distance_m / object_m
    return math.degrees(2.0 * math.atan((width_px / 2.0) / focal_px))
