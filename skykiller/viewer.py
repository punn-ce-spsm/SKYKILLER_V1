"""The operator's view of the L2 lane.

Deliberately not `results.plot()`: the default Ultralytics overlay draws a class
name and a confidence, and the two things this lane exists to produce are a
stable track id and a bearing. Those go on the box.
"""

from __future__ import annotations

import cv2

from .config import Config
from .geometry import Camera
from .schemas import Detection

_TRACK = (86, 180, 233)  # B,G,R -- readable on sky and on treeline
_PROVISIONAL = (60, 160, 240)
_MATCH = (120, 200, 120)      # the identity we were told to look for
_NOT_MATCH = (130, 130, 130)  # a real contact, just not the target
_HUD = (240, 240, 240)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Viewer:
    def __init__(self, cfg: Config) -> None:
        self._window = cfg.window
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

    def show(self, result, dets: list[Detection], cam: Camera, fps: float) -> bool:
        """Draw one frame. Returns False when the operator asks to quit."""
        frame = result.orig_img.copy()

        for det in dets:
            cx, cy, w, h = det.extra["bbox_xywh"]
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            ident = det.extra.get("identity")

            if ident is None:
                colour = _PROVISIONAL if det.extra.get("provisional") else _TRACK
                weight = 2
            else:
                # Non-matching contacts stay on screen, drawn thin and grey. They
                # are real, they are just not the target, and hiding them would
                # tell the operator less than showing them.
                colour = _MATCH if ident["match"] else _NOT_MATCH
                weight = 2 if ident["match"] else 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, weight)

            label = f"T-{det.raw_id} {det.extra['label']} {det.conf:.2f}"
            if ident is not None:
                score = "--" if ident["score"] is None else f"{ident['score']:.2f}"
                label = (f"T-{det.raw_id} {ident['name']} {score}" if ident["match"]
                         else f"T-{det.raw_id} not-target {score}")
            bearing = f"az {det.az:6.1f}  el {det.el:+5.1f}"
            if det.r is not None:
                bearing += f"  r ~{det.r:.0f}m"

            cv2.putText(frame, label, (x1, max(y1 - 22, 12)), _FONT, 0.5, colour, 1, cv2.LINE_AA)
            cv2.putText(frame, bearing, (x1, max(y1 - 6, 26)), _FONT, 0.45, colour, 1, cv2.LINE_AA)

        # Boresight cross, so the operator can see where zero azimuth is.
        ccx, ccy = cam.width // 2, cam.height // 2
        cv2.drawMarker(frame, (ccx, ccy), (120, 120, 120), cv2.MARKER_CROSS, 14, 1)

        matched = sum(1 for d in dets if (d.extra.get("identity") or {}).get("match"))
        gated = any(d.extra.get("identity") is not None for d in dets)
        counts = f"tracks {len(dets)}" + (f"  emitting {matched}" if gated else "")
        hud = f"L2 VISUAL  {fps:5.1f} fps  {counts}  fov {cam.hfov_deg:.0f}x{cam.vfov_deg:.0f}"
        cv2.putText(frame, hud, (10, 22), _FONT, 0.55, _HUD, 1, cv2.LINE_AA)
        if any(d.extra.get("provisional") for d in dets):
            cv2.putText(
                frame, "PROVISIONAL DETECTOR - not drone-trained", (10, 44),
                _FONT, 0.5, _PROVISIONAL, 1, cv2.LINE_AA,
            )

        cv2.imshow(self._window, frame)
        return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)

    def close(self) -> None:
        cv2.destroyAllWindows()
