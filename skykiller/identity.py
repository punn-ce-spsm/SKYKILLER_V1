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

#: YuNet gets unreliable on tiny crops; below this we upscale before detecting.
_MIN_FACE_INPUT = 160


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


class FaceMatcher:
    """Matches a crop against one enrolled face embedding."""

    def __init__(self, reference: np.ndarray, name: str, threshold: float = DEFAULT_THRESHOLD) -> None:
        ensure_models()
        self.name = name
        self.threshold = threshold
        self._ref = reference
        self._det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.7, 0.3, 5000)
        self._rec = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")

    # -- detection -------------------------------------------------------
    def _largest_face(self, image: np.ndarray) -> _Face | None:
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return None
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(image)
        if faces is None or len(faces) == 0:
            return None
        rows = [(_Face(r, float(r[2]) * float(r[3]))) for r in faces]
        return max(rows, key=lambda f: f.area)

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


def enroll(image_path: str | Path, name: str, out_path: str | Path) -> Path:
    """Store one face embedding from a photograph. Raises if there is no face."""
    ensure_models()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"could not read image: {image_path}")

    matcher = FaceMatcher(np.zeros((1, 128), np.float32), name)
    feat = matcher.embed(img)
    if feat is None:
        raise ValueError(
            f"no face found in {image_path}. Use a clear, front-on, well-lit photo "
            f"where the face is a decent fraction of the frame."
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feat)
    return out


def load(reference_path: str | Path, name: str, threshold: float) -> FaceMatcher:
    p = Path(reference_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"no enrolled identity at {p}. Run:\n"
            f"  python -m skykiller enroll --image <your-photo.jpg> --name {name}"
        )
    return FaceMatcher(np.load(p), name, threshold)


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
