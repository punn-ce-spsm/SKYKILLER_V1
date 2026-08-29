"""Identity gate: caching, staleness, and the emit decision.

Most of this runs with a stub matcher so it needs no models, no camera and no
network. The one test that loads the real ONNX models skips itself when they are
absent, so a fresh clone still gets a green suite.
"""

import numpy as np
import pytest

from skykiller.config import Config
from skykiller.identity import DEFAULT_THRESHOLD, TrackVerdicts
from skykiller.l2_visual import should_emit
from skykiller.schemas import Detection, SRC_VISUAL

CROP = np.zeros((64, 64, 3), np.uint8)


class _StubMatcher:
    """Returns a scripted sequence of scores, and counts how often it was asked."""

    name = "stub"
    threshold = DEFAULT_THRESHOLD

    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = 0

    def score(self, crop):
        self.calls += 1
        return self._scores.pop(0) if self._scores else None


def _det(identity: dict | None) -> Detection:
    extra = {"label": "person", "bbox_xywh": [1.0, 2.0, 3.0, 4.0], "provisional": False}
    if identity is not None:
        extra["identity"] = identity
    return Detection(src=SRC_VISUAL, az=0.0, el=0.0, conf=0.9, raw_id="1", extra=extra)


# ---------------------------------------------------------------- caching


def test_verdict_is_cached_between_rechecks():
    """A 37 MB embedder must not run on every box of every frame."""
    m = _StubMatcher([0.9])
    v = TrackVerdicts(m, recheck_every=10)
    for frame in range(9):
        assert v.verdict("1", CROP, frame) == (True, 0.9)
    assert m.calls == 1, "cached verdict should have been reused"


def test_verdict_is_rechecked_after_the_window():
    """A recycled track id must not carry a stale confirmation forever."""
    m = _StubMatcher([0.9, 0.05])
    v = TrackVerdicts(m, recheck_every=5)
    assert v.verdict("1", CROP, 0) == (True, 0.9)
    assert v.verdict("1", CROP, 5) == (False, 0.05), "should have re-verified and flipped"
    assert m.calls == 2


def test_a_frame_with_no_visible_face_keeps_the_previous_verdict():
    """Turning your head must not drop a confirmed track."""
    m = _StubMatcher([0.9, None])
    v = TrackVerdicts(m, recheck_every=1)
    assert v.verdict("1", CROP, 0) == (True, 0.9)
    assert v.verdict("1", CROP, 1) == (True, 0.9), "no face != not the target"


def test_an_unknown_track_with_no_face_is_not_a_match():
    """Absence of evidence is not confirmation -- default closed."""
    v = TrackVerdicts(_StubMatcher([None]), recheck_every=1)
    assert v.verdict("99", CROP, 0) == (False, None)


def test_forget_drops_tracks_that_no_longer_exist():
    m = _StubMatcher([0.9, 0.9])
    v = TrackVerdicts(m, recheck_every=100)
    v.verdict("1", CROP, 0)
    v.verdict("2", CROP, 0)
    v.forget({"1"})
    v.verdict("1", CROP, 1)
    assert m.calls == 2, "track 1 should still be cached"
    v.verdict("2", CROP, 1)
    assert m.calls == 3, "track 2 was forgotten and must be re-verified"


def test_threshold_boundary_is_inclusive():
    v = TrackVerdicts(_StubMatcher([DEFAULT_THRESHOLD]), recheck_every=1)
    assert v.verdict("1", CROP, 0)[0] is True


# ------------------------------------------------------------- emit gate


def test_identity_disabled_emits_everything():
    cfg = Config()
    assert cfg.identity.enabled is False
    assert should_emit(_det(None), cfg) is True
    assert should_emit(_det({"name": None, "match": False, "score": 0.01}), cfg) is True


def test_identity_enabled_emits_only_the_match():
    cfg = Config()
    cfg.identity.enabled = True
    assert should_emit(_det({"name": "me", "match": True, "score": 0.9}), cfg) is True
    assert should_emit(_det({"name": None, "match": False, "score": 0.1}), cfg) is False


def test_identity_enabled_but_no_verdict_does_not_emit():
    """Fails closed: an unjudged track is not the target."""
    cfg = Config()
    cfg.identity.enabled = True
    assert should_emit(_det(None), cfg) is False


# ------------------------------------------------- the real models, if present


@pytest.mark.skipif(
    not (__import__("skykiller.identity", fromlist=["YUNET_PATH"]).YUNET_PATH.exists()
         and __import__("skykiller.identity", fromlist=["SFACE_PATH"]).SFACE_PATH.exists()),
    reason="face models not downloaded; run `python -m skykiller enroll` once",
)
def test_real_models_separate_two_different_people():
    """Two faces from Ultralytics' bundled zidane.jpg must not be confusable.

    Measured separation is ~0.94 for the same face against ~0.07 for a different
    one, so the 0.363 threshold has wide margins on both sides. If this test ever
    narrows, the threshold needs revisiting before the gate can be trusted.
    """
    import pathlib

    import cv2
    import ultralytics

    from skykiller.identity import FaceMatcher

    img = cv2.imread(str(pathlib.Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))
    det = cv2.FaceDetectorYN.create(
        str(__import__("skykiller.identity", fromlist=["YUNET_PATH"]).YUNET_PATH),
        "", (320, 320), 0.7, 0.3, 5000,
    )
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    assert faces is not None and len(faces) >= 2

    faces = sorted(faces, key=lambda r: r[0])
    crops = []
    for f in faces[:2]:
        x, y, fw, fh = (int(v) for v in f[:4])
        m = int(max(fw, fh) * 0.6)
        crops.append(img[max(y - m, 0):y + fh + m, max(x - m, 0):x + fw + m])

    ref = FaceMatcher(np.zeros((1, 128), np.float32), "tmp").embed(crops[0])
    assert ref is not None
    matcher = FaceMatcher(ref, "person0")

    same = matcher.score(crops[0])
    other = matcher.score(crops[1])
    assert same is not None and other is not None
    assert same > 0.8, f"same face scored only {same:.3f}"
    assert other < 0.3, f"different person scored {other:.3f} -- too close to the threshold"
    assert same > DEFAULT_THRESHOLD > other


def test_a_missing_crop_keeps_the_previous_verdict():
    """A box clipped to nothing at the frame edge is no evidence, not a negative."""
    m = _StubMatcher([0.9])
    v = TrackVerdicts(m, recheck_every=1)
    assert v.verdict("1", CROP, 0) == (True, 0.9)
    assert v.verdict("1", None, 1) == (True, 0.9)
    assert v.verdict("1", np.zeros((0, 0, 3), np.uint8), 2) == (True, 0.9)
    assert m.calls == 1, "an empty crop must not be sent to the embedder"


def test_low_confidence_frames_do_not_erase_a_confirmed_identity():
    """The regression: forgetting on the *emitted* set, not the tracked set.

    A confirmed target whose detection confidence dips below min_conf_emit for a
    single frame must not lose its identity verdict -- if its face happened to be
    turned away on the next frame it would return as not-target.
    """
    import numpy as np

    from skykiller import l2_visual
    from skykiller.config import Config
    from skykiller.geometry import Camera
    from tests.test_pipeline import _FakeBoxes, _FakeResult, _fake_model_yielding

    cfg = Config()
    cfg.identity.enabled = True
    cfg.lane.min_conf_emit = 0.5
    cam = Camera(1280, 720, 65.0)
    loaded = l2_visual.LoadedModel(_fake_model_yielding([]), {0: "person"}, None, False, "fake.pt")

    # Scores: confirm on frame 1, then no face at all from then on.
    verdicts = TrackVerdicts(_StubMatcher([0.9, None, None]), recheck_every=1)

    high = _FakeResult(_FakeBoxes([[640.0, 360.0, 40.0, 40.0]], [1], [0.9], [0]))
    low = _FakeResult(_FakeBoxes([[640.0, 360.0, 40.0, 40.0]], [1], [0.2], [0]))

    d1 = l2_visual._detections_from_result(high, cam, cfg, loaded, verdicts, 1)
    assert d1[0].extra["identity"]["match"] is True

    # Frame 2: confidence dips, so the track is not in the emitted list at all.
    d2 = l2_visual._detections_from_result(low, cam, cfg, loaded, verdicts, 2)
    assert d2 == []

    # Frame 3: confidence recovers, face still not visible. Must still be ours.
    d3 = l2_visual._detections_from_result(high, cam, cfg, loaded, verdicts, 3)
    assert d3[0].extra["identity"]["match"] is True, "verdict was erased by the dip"
