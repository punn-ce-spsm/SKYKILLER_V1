"""End-to-end lane behaviour with the detector mocked out.

Hermetic: no weights, no camera, no network. This is the check that can run in
CI and still fail for a real reason -- it exercises the actual `stream()` code
path, including building the camera from the frame rather than from config.
"""

import numpy as np
import pytest

from skykiller import l2_visual
from skykiller.config import Config
from skykiller.geometry import Camera
from skykiller.schemas import SRC_VISUAL

FRAME_W, FRAME_H = 1280, 720


class _FakeTensor:
    """Stands in for the torch tensors on `results.boxes`."""

    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def int(self):
        return _FakeTensor([int(v) for v in self._data])

    def tolist(self):
        return self._data


class _FakeBoxes:
    def __init__(self, xywh, ids, confs, clss):
        self.xywh = _FakeTensor(xywh)
        self.id = _FakeTensor(ids)
        self.conf = _FakeTensor(confs)
        self.cls = _FakeTensor(clss)

    def __len__(self):
        return len(self.id.tolist())


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.orig_img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _fake_model_yielding(tracks):
    """Build a stand-in model whose .track() replays the given per-frame boxes."""

    class _FakeModel:
        names = {0: "drone"}

        def track(self, **kwargs):
            self.kwargs = kwargs
            for boxes in tracks:
                yield _FakeResult(boxes)

    return _FakeModel()


@pytest.fixture
def cfg():
    c = Config()
    c.show = False
    c.camera.hfov_deg = 65.0
    return c


def _patch_model(monkeypatch, model):
    monkeypatch.setattr(
        l2_visual, "load_model",
        lambda _cfg: l2_visual.LoadedModel(model, {0: "drone"}, None, False, "fake.pt"),
    )


def test_track_crossing_the_frame_yields_one_id_and_rising_azimuth(monkeypatch, cfg):
    """A target moving centre -> right edge: same id throughout, azimuth climbs."""
    frames = [
        _FakeBoxes([[640.0 + step * 64.0, 360.0, 20.0, 20.0]], [7], [0.9], [0])
        for step in range(10)
    ]
    _patch_model(monkeypatch, _fake_model_yielding(frames))

    azimuths, ids = [], set()
    for _result, dets, cam in l2_visual.stream(cfg):
        assert cam.width == FRAME_W and cam.height == FRAME_H
        for det in dets:
            azimuths.append(det.az)
            ids.add(det.raw_id)

    assert ids == {"7"}, "tracker id must be stable across the crossing"
    assert azimuths[0] == pytest.approx(0.0), "first sample is the boresight"
    assert all(b > a for a, b in zip(azimuths, azimuths[1:])), f"not monotonic: {azimuths}"
    assert azimuths[-1] == pytest.approx(
        Camera(FRAME_W, FRAME_H, 65.0).bearing(640.0 + 9 * 64.0, 360.0)[0]
    )


def test_camera_is_built_from_the_frame_not_the_config(monkeypatch, cfg):
    """A webcam that negotiates a different size must not skew every bearing.

    The frame is 1280x720 here. If the code trusted a config value instead, the
    focal length -- and so every azimuth -- would be wrong by the size ratio.
    """
    frames = [_FakeBoxes([[1280.0, 360.0, 10.0, 10.0]], [1], [0.9], [0])]
    _patch_model(monkeypatch, _fake_model_yielding(frames))

    (_result, dets, cam), = list(l2_visual.stream(cfg))
    assert (cam.width, cam.height) == (FRAME_W, FRAME_H)
    # Right edge of a 65-degree frame is exactly half the FOV off the bore.
    assert dets[0].az == pytest.approx(32.5)


def test_frames_without_tracks_emit_nothing(monkeypatch, cfg):
    """`boxes.id` is None until the tracker has something. Must not crash."""

    class _NoIds:
        xywh = _FakeTensor([])
        id = None
        conf = _FakeTensor([])
        cls = _FakeTensor([])

        def __len__(self):
            return 0

    _patch_model(monkeypatch, _fake_model_yielding([_NoIds(), _NoIds()]))
    assert all(dets == [] for _r, dets, _c in l2_visual.stream(cfg))


def test_low_confidence_boxes_are_not_emitted(monkeypatch, cfg):
    cfg.lane.min_conf_emit = 0.5
    frames = [_FakeBoxes([[640.0, 360.0, 20.0, 20.0], [700.0, 360.0, 20.0, 20.0]],
                         [1, 2], [0.9, 0.2], [0, 0])]
    _patch_model(monkeypatch, _fake_model_yielding(frames))

    (_r, dets, _c), = list(l2_visual.stream(cfg))
    assert [d.raw_id for d in dets] == ["1"]


def test_range_is_none_unless_a_target_width_is_configured(monkeypatch, cfg):
    frames = [_FakeBoxes([[640.0, 360.0, 17.57, 17.57]], [1], [0.9], [0])]

    _patch_model(monkeypatch, _fake_model_yielding(frames))
    (_r, dets, _c), = list(l2_visual.stream(cfg))
    assert dets[0].r is None, "an assumed size must not become a confident range"

    cfg.lane.target_width_m = 0.35  # DJI Mini 4 Pro
    _patch_model(monkeypatch, _fake_model_yielding(frames))
    (_r, dets, _c), = list(l2_visual.stream(cfg))
    assert dets[0].r == pytest.approx(20.0, abs=0.1)


def test_emitted_detections_carry_lane_and_provenance(monkeypatch, cfg):
    frames = [_FakeBoxes([[640.0, 360.0, 20.0, 20.0]], [3], [0.77], [0])]
    _patch_model(monkeypatch, _fake_model_yielding(frames))

    (_r, dets, _c), = list(l2_visual.stream(cfg))
    det = dets[0]
    assert det.src == SRC_VISUAL
    assert det.raw_id == "3"
    assert det.extra["label"] == "drone"
    assert det.extra["provisional"] is False
    assert det.extra["bbox_xywh"] == [640.0, 360.0, 20.0, 20.0]
