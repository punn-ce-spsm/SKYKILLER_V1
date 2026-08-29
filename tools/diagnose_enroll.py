"""Why did enrolment find no face? Reports every step, so we stop guessing.

    .venv/bin/python tools/diagnose_enroll.py me.jpg

Prints what the file is, what OpenCV loaded, whether EXIF rotation was applied,
and how many faces YuNet finds at each orientation and confidence threshold.
Reads only; changes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skykiller.config import IdentityCfg  # noqa: E402
from skykiller.identity import (  # noqa: E402
    GOOD_FACE_PX, GOOD_SHARPNESS, YUNET_PATH, FaceMatcher, ensure_models,
)

ROTATIONS = [
    ("as loaded", None),
    ("rot  90 cw", cv2.ROTATE_90_CLOCKWISE),
    ("rot 180", cv2.ROTATE_180),
    ("rot  90 ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
]


def _detector(threshold: float) -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), threshold, 0.3, 5000)


def _count(det: cv2.FaceDetectorYN, img: np.ndarray) -> tuple[int, float]:
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return 0, 0.0
    return len(faces), float(max(f[-1] for f in faces))


def main(path_str: str) -> int:
    ensure_models()
    path = Path(path_str)

    print(f"=== 1. the file: {path}")
    if not path.exists():
        print(f"    MISSING. Nothing at that path. Working dir is {Path.cwd()}")
        return 1
    head = path.open("rb").read(12)
    kind = (
        "JPEG" if head[:2] == b"\xff\xd8" else
        "PNG" if head[:8] == b"\x89PNG\r\n\x1a\n" else
        "HEIC/HEIF" if head[4:8] == b"ftyp" else
        f"unknown ({head[:4]!r})"
    )
    print(f"    {path.stat().st_size / 1e6:.2f} MB, looks like {kind}")
    if kind.startswith("HEIC"):
        print("    >>> OpenCV cannot read HEIC. Re-export as JPEG.")

    print("\n=== 2. what OpenCV loaded")
    img = cv2.imread(str(path))
    if img is None:
        print("    imread returned None -- unreadable or unsupported format.")
        return 1
    h, w = img.shape[:2]
    print(f"    {w} x {h}, {'portrait' if h > w else 'landscape'}")
    print(f"    mean brightness {img.mean():.0f}/255, contrast (std) {img.std():.0f}")
    if img.mean() < 40:
        print("    >>> very dark; that alone can defeat detection")

    print("\n=== 3. did EXIF rotation get applied?")
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if raw is not None and raw.shape != img.shape:
        print(f"    yes -- stored {raw.shape[1]}x{raw.shape[0]}, presented {w}x{h}")
    else:
        print("    no rotation applied (either none needed, or no EXIF tag present)")
        print("    NOTE: a stripped EXIF tag leaves a portrait photo lying on its side.")

    print("\n=== 4. faces found, by orientation and threshold")
    print(f"    {'orientation':>12} " + " ".join(f"{t:>7}" for t in (0.9, 0.7, 0.5, 0.3)))
    best: tuple[int, str, float] = (0, "", 0.0)
    for label, rot in ROTATIONS:
        cells = []
        for thr in (0.9, 0.7, 0.5, 0.3):
            im = img if rot is None else cv2.rotate(img, rot)
            n, score = _count(_detector(thr), im)
            cells.append(f"{n:>7}")
            if n > 0 and (best[0] == 0 or score > best[2]):
                best = (n, label, score)
        print(f"    {label:>12} " + " ".join(cells))

    print("\n=== 5. how comfortably will this enrol?")
    print("    Measured: almost anything the detector can see still tells you apart")
    print("    from other people. These numbers mark where the margin narrows.")
    cfg = IdentityCfg()
    m = FaceMatcher(np.zeros((1, 128), np.float32), "probe",
                    detect_threshold=min(0.3, cfg.detect_threshold))
    upright = img if best[1] in ("", "as loaded") else cv2.rotate(
        img, dict(ROTATIONS)[best[1]])
    q = m.quality(upright)
    if q is None:
        print("    no face to measure")
    else:
        size_ok = "comfortable" if q.face_px >= GOOD_FACE_PX else "marginal"
        sharp_ok = "comfortable" if q.sharpness >= GOOD_SHARPNESS else "marginal"
        print(f"    face width : {q.face_px:>5} px      (want {GOOD_FACE_PX}+)   {size_ok}")
        print(f"    sharpness  : {q.sharpness:>5.0f}         (want {GOOD_SHARPNESS}+)    {sharp_ok}")
        print(f"    confidence : {q.confidence:>5.2f}         (need {cfg.detect_threshold}+)")

    print("\n=== verdict")
    if q is not None and best[0] and not q.comfortable:
        print(f"    Enrolment will WORK, but the photo is marginal: {q.advice}.")
        print("    Try it as-is first -- measurements say photos this weak usually")
        print("    still tell you apart from other people. Retake only if recognition")
        print("    turns out flaky.")
        return 0

    if best[0] == 0:
        print("    No face at any orientation or threshold.")
        print("    The photo itself is the problem: retake it front-on, well lit,")
        print("    face filling a good part of the frame, eyes open, no heavy shadow.")
        return 2
    if best[1] == "as loaded":
        print(f"    Good: a face is detectable as loaded (confidence {best[2]:.2f}) "
              f"and is comfortably\n    above the size and sharpness marks. Enrolment "
              f"should work well.")
        return 0
    print(f"    >>> ROOT CAUSE: the image is sideways.")
    print(f"    No face 'as loaded', but {best[0]} at '{best[1]}' (score {best[2]:.2f}).")
    print("    Enrolment must try all four orientations. That is a code fix, not a")
    print("    photo problem -- tell Claude this and it will be handled.")
    return 3


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
