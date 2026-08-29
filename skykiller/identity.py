"""Identity filtering: emit only the track that matches a known identity.

The counter-UAS system needs exactly this shape — *"emit only the aircraft whose
Remote ID serial is on this list"* — so the stage is written against an abstract
`IdentityMatcher` and given one implementation, faces, which can be tested
indoors today. The L1b Remote ID matcher slots into the same interface later
without touching the lane.

Faces use OpenCV's own YuNet detector and SFace embedder: no new dependencies,
227 KB and 37 MB of ONNX. Measured separation on a two-face reference image is
1.000 self-similarity against 0.038 for a different person, so the documented
0.363 cosine threshold has a lot of daylight either side of it.

An enrolled embedding is biometric data. It stays on this machine, `models/` is
gitignored, and nothing here uploads anything.
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import REPO_ROOT

#: opencv_zoo keeps weights in Git LFS, so these must be the media endpoint --
#: the raw.githubusercontent URLs return a 131-byte LFS pointer that loads as a
#: corrupt ONNX file.
_LFS = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
YUNET_URL = f"{_LFS}/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = f"{_LFS}/face_recognition_sface/face_recognition_sface_2021dec.onnx"

FACE_DIR = REPO_ROOT / "models" / "face"
YUNET_PATH = FACE_DIR / "yunet.onnx"
SFACE_PATH = FACE_DIR / "sface.onnx"

#: OpenCV's documented cosine threshold for SFace. Above this is the same person.
DEFAULT_THRESHOLD = 0.363

#: How confident YuNet must be that a region is a face. Lower catches marginal
#: photos; it also invents faces. Measured on a 2-face reference image:
#: 0.9 -> 2 detections, 0.7 -> 2, 0.5 -> 3, 0.3 -> 4. Everything above 2 is a
#: false positive, which is why enrolment picks by confidence and never by
#: whichever box happens to be largest at a permissive setting.
DEFAULT_DETECT_THRESHOLD = 0.5

#: YuNet gets unreliable on tiny crops; below this we upscale before detecting.
_MIN_FACE_INPUT = 160

#: Advisory quality thresholds for enrolment -- warnings, never refusals.
#:
#: Measured by degrading a photo, enrolling from it, then asking the question
#: that actually matters: does the reference still accept the true person and
#: reject a different one at the 0.363 match threshold?
#:
#:   face 102px sharp 249 -> me 1.000 other 0.029  works
#:   face  33px sharp  52 -> me 0.888 other 0.053  works
#:   face  16px sharp  16 -> me 0.743 other 0.027  works
#:   face 110px sharp   4 -> me 0.557 other 0.031  works
#:   face 111px sharp   3 -> me 0.386 other -0.105 works, margin only 0.023
#:   face  83px sharp   2 -> me 0.291 other -0.102 BROKEN
#:
#: Almost anything the detector can see at all still discriminates. An earlier
#: version of this refused photos below 28px/sharpness 5 -- that would have
#: rejected three of the working rows above. Detection is the real gate; these
#: two only mark the region where the margin starts to narrow.
GOOD_FACE_PX = 30
GOOD_SHARPNESS = 6


def sharpness(image: np.ndarray, face: "_Face") -> float:
    """Variance of the Laplacian over the face box, size-normalised.

    Resized to 112x112 first so this measures blur alone and does not simply
    restate face size -- that is what MIN_FACE_PX is for.
    """
    x, y, w, h = (int(v) for v in face.row[:4])
    crop = image[max(y, 0):y + h, max(x, 0):x + w]
    if crop.size == 0:
        return 0.0
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(cv2.resize(grey, (112, 112)), cv2.CV_64F).var())


@dataclass(slots=True)
class Quality:
    """How comfortably a photo will enrol. Advisory only -- see the note above."""

    face_px: int
    sharpness: float
    confidence: float

    @property
    def comfortable(self) -> bool:
        """True when the photo is clear of the region where margins narrow."""
        return self.face_px >= GOOD_FACE_PX and self.sharpness >= GOOD_SHARPNESS

    @property
    def advice(self) -> str:
        notes = []
        if self.face_px < GOOD_FACE_PX:
            notes.append(f"the face is {self.face_px}px wide (comfortable is "
                         f"{GOOD_FACE_PX}+) -- get closer or crop tighter")
        if self.sharpness < GOOD_SHARPNESS:
            notes.append(f"it is soft (sharpness {self.sharpness:.0f}, comfortable is "
                         f"{GOOD_SHARPNESS}+) -- more light, hold still")
        return "; ".join(notes) if notes else "good"


class IdentityMatcher(Protocol):
    """Decides whether a cropped detection is the identity we are looking for."""

    name: str

    def score(self, crop: np.ndarray) -> float | None:
        """Similarity in [0,1], or None when no comparable subject was found."""
        ...


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000:
        return
    print(f"[identity] downloading {dest.name} ...", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 -- constant https URL
    print(f"[identity] {dest.name}: {dest.stat().st_size / 1e6:.1f} MB", file=sys.stderr)


def ensure_models() -> None:
    """Fetch YuNet and SFace if they are not already present."""
    _download(YUNET_URL, YUNET_PATH)
    _download(SFACE_URL, SFACE_PATH)


@dataclass(slots=True)
class _Face:
    row: np.ndarray  # YuNet's 15 values: x,y,w,h, 5 landmarks, score
    area: float

    @property
    def score(self) -> float:
        return float(self.row[-1])


class FaceMatcher:
    """Matches a crop against one enrolled face embedding."""

    def __init__(
        self,
        reference: np.ndarray,
        name: str,
        threshold: float = DEFAULT_THRESHOLD,
        detect_threshold: float = DEFAULT_DETECT_THRESHOLD,
    ) -> None:
        ensure_models()
        self.name = name
        self.threshold = threshold
        self.detect_threshold = detect_threshold
        self._ref = reference
        self._det = cv2.FaceDetectorYN.create(
            str(YUNET_PATH), "", (320, 320), detect_threshold, 0.3, 5000
        )
        self._rec = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")

    # -- detection -------------------------------------------------------
    def faces(self, image: np.ndarray) -> list[_Face]:
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return []
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(image)
        if faces is None or len(faces) == 0:
            return []
        return [_Face(r, float(r[2]) * float(r[3])) for r in faces]

    def _largest_face(self, image: np.ndarray) -> _Face | None:
        """The subject of this crop: the largest *credible* face.

        Largest, not highest-scoring, because inside a person crop a background
        face may score well while the subject is the big one. But "credible"
        matters once the detection threshold is permissive: at 0.5 a two-face
        image yields a third detection at 0.61 against real faces at 0.94 and
        0.90. Without this band a spurious box that happens to be large would
        win, embed as nobody, and flip a confirmed track to not-target.
        """
        found = self.faces(image)
        if not found:
            return None
        best = max(f.score for f in found)
        credible = [f for f in found if f.score >= best * 0.8]
        return max(credible, key=lambda f: f.area)

    def quality(self, image: np.ndarray) -> Quality | None:
        """Measure whether the subject of this image is enrollable."""
        face = self._largest_face(image)
        if face is None:
            return None
        return Quality(int(face.row[2]), sharpness(image, face), face.score)

    def best_face_score(self, image: np.ndarray) -> float:
        """Confidence of the most confident face, or 0.0 if there is none.

        Used to choose between orientations. Confidence, not area, because a
        wrongly-rotated image produces large *low-confidence* false positives --
        measured: an upside-down two-face image yields three detections.
        """
        found = self.faces(image)
        return max((f.score for f in found), default=0.0)

    def embed(self, image: np.ndarray) -> np.ndarray | None:
        """Embed the largest face in an image, or None if there isn't one."""
        # Upscale small crops: YuNet's confidence collapses on a 60px head, and a
        # person box from across a room is exactly that.
        scale = 1.0
        h, w = image.shape[:2]
        if max(h, w) < _MIN_FACE_INPUT:
            scale = _MIN_FACE_INPUT / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        face = self._largest_face(image)
        if face is None:
            return None
        return self._rec.feature(self._rec.alignCrop(image, face.row))

    # -- matching --------------------------------------------------------
    def score(self, crop: np.ndarray) -> float | None:
        feat = self.embed(crop)
        if feat is None:
            return None
        return float(self._rec.match(self._ref, feat, cv2.FaceRecognizerSF_FR_COSINE))


#: Enrolment tries every orientation. YuNet is not rotation-invariant -- a
#: sideways image yields zero faces where an upright one yields two, measured.
#: Phone photos routinely store a portrait shot as landscape pixels plus an EXIF
#: orientation tag, and a stripped or absent tag leaves the pixels on their side.
#:
#: Deliberately *not* done on the live path: camera frames arrive upright, and
#: paying four detections per crop per frame to guard against a case that cannot
#: happen would be pure cost.
_ENROLL_ROTATIONS = (
    (None, "as provided"),
    (cv2.ROTATE_90_CLOCKWISE, "rotated 90 clockwise"),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, "rotated 90 anticlockwise"),
    (cv2.ROTATE_180, "rotated 180"),
)


def enroll(
    image_path: str | Path,
    name: str,
    out_path: str | Path,
    detect_threshold: float = DEFAULT_DETECT_THRESHOLD,
) -> Path:
    """Store one face embedding from a photograph. Raises if there is no face.

    Tries all four orientations, because the photo may be sideways and the
    caller has no way to know that from the failure, and warns (never refuses)
    when the photo is in the region where recognition margins start to narrow.
    """
    ensure_models()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"could not read image: {image_path}")

    matcher = FaceMatcher(np.zeros((1, 128), np.float32), name, detect_threshold=detect_threshold)

    # Score every orientation and take the most confident, rather than the first
    # that returns anything. A 180-degree image yields confident-looking garbage,
    # so first-hit ordering enrols a face that then matches nobody.
    scored = []
    for rotation, label in _ENROLL_ROTATIONS:
        candidate = img if rotation is None else cv2.rotate(img, rotation)
        scored.append((matcher.best_face_score(candidate), rotation, label, candidate))
    best_score, rotation, label, candidate = max(scored, key=lambda t: t[0])

    feat = matcher.embed(candidate) if best_score > 0 else None
    if feat is not None and rotation is not None:
        print(
            f"[identity] the photo was sideways -- the face is upright with it {label} "
            f"(confidence {best_score:.2f}). Enrolled from that.",
            file=sys.stderr,
        )

    if feat is not None:
        q = matcher.quality(candidate)
        if q is not None and not q.comfortable:
            print(
                f"[identity] marginal photo: {q.advice}.\n"
                f"[identity] Enrolling anyway -- measurements say this usually still "
                f"works. If recognition is flaky, retake it before touching any "
                f"threshold.",
                file=sys.stderr,
            )

    if feat is None:
        raise ValueError(
            f"no face found in {image_path}, at any orientation.\n"
            f"  Retake it front-on and well lit, with your face filling a good part "
            f"of the frame.\n"
            f"  To see exactly what the detector saw, run:\n"
            f"    python tools/diagnose_enroll.py {image_path}"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feat)
    return out


def load(
    reference_path: str | Path,
    name: str,
    threshold: float,
    detect_threshold: float = DEFAULT_DETECT_THRESHOLD,
) -> FaceMatcher:
    p = Path(reference_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"no enrolled identity at {p}. Run:\n"
            f"  python -m skykiller enroll --image <your-photo.jpg> --name {name}"
        )
    return FaceMatcher(np.load(p), name, threshold, detect_threshold)


class TrackVerdicts:
    """Per-track identity decisions, so a 37 MB model is not run every frame.

    Identity is a property of the *track*, not the frame: once a track is
    confirmed, kinematics carry it while the subject turns away and no face is
    visible at all. Re-checked every `recheck_every` frames so a track id
    recycled onto a different person cannot keep a stale confirmation.
    """

    def __init__(self, matcher: IdentityMatcher, recheck_every: int = 15) -> None:
        self._matcher = matcher
        self._recheck = max(1, recheck_every)
        self._verdicts: dict[str, tuple[bool, float | None, int]] = {}

    def verdict(self, track_id: str, crop: np.ndarray | None, frame_no: int) -> tuple[bool, float | None]:
        """Return `(is_match, score)`. Score is None when nothing was comparable.

        Three situations produce no score and must not be confused with a
        negative: no crop (a box clipped to nothing at the frame edge), no face
        in the crop, and a crop too small to detect in. All of them mean "no
        evidence this frame", so a previously confirmed track keeps its verdict.
        """
        cached = self._verdicts.get(track_id)
        if cached is not None and frame_no - cached[2] < self._recheck:
            return cached[0], cached[1]

        score = None if crop is None or crop.size == 0 else self._matcher.score(crop)
        if score is None:
            # No face this frame. Keep any previous decision rather than
            # dropping a confirmed track the moment the subject turns their head.
            if cached is not None:
                self._verdicts[track_id] = (cached[0], cached[1], frame_no)
                return cached[0], cached[1]
            return False, None

        is_match = score >= getattr(self._matcher, "threshold", DEFAULT_THRESHOLD)
        self._verdicts[track_id] = (is_match, score, frame_no)
        return is_match, score

    def forget(self, keep: set[str]) -> None:
        """Drop verdicts for tracks that no longer exist, bounding memory."""
        for tid in [t for t in self._verdicts if t not in keep]:
            del self._verdicts[tid]
