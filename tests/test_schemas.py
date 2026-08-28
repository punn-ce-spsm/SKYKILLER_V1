"""The `Detection` contract is shared by every lane. These tests pin it.

If one of these fails because a field was renamed, that rename is a breaking
change to fusion, the console and the CoT exporter -- not a local edit.
"""

import json

import pytest

from skykiller.schemas import Detection, SRC_VISUAL


def test_field_names_match_the_spec_board_contract():
    """Section 2 of the spec board: { t_utc, src, az, el, r, conf, raw_id }."""
    det = Detection(src=SRC_VISUAL, az=12.5, el=3.0, conf=0.8)
    payload = json.loads(det.to_json())
    assert set(payload) == {"t_utc", "src", "az", "el", "r", "conf", "raw_id", "extra"}


def test_json_round_trip_preserves_everything():
    original = Detection(
        src=SRC_VISUAL, az=200.25, el=-4.5, conf=0.91, r=137.0,
        raw_id="7", extra={"label": "drone", "bbox_xywh": [1.0, 2.0, 3.0, 4.0]},
    )
    restored = Detection.from_json(original.to_json())
    assert restored == original


def test_absent_range_is_none_not_zero():
    """Fusion must be able to tell 'no range measurement' from 'range is zero'."""
    det = Detection(src=SRC_VISUAL, az=0.0, el=0.0, conf=0.5)
    assert det.r is None
    assert json.loads(det.to_json())["r"] is None


def test_azimuth_is_normalised_into_zero_to_360():
    assert Detection(src=SRC_VISUAL, az=-20.0, el=0.0, conf=0.5).az == pytest.approx(340.0)
    assert Detection(src=SRC_VISUAL, az=370.0, el=0.0, conf=0.5).az == pytest.approx(10.0)
    assert Detection(src=SRC_VISUAL, az=360.0, el=0.0, conf=0.5).az == pytest.approx(0.0)


def test_confidence_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError):
        Detection(src=SRC_VISUAL, az=0.0, el=0.0, conf=1.4)
    with pytest.raises(ValueError):
        Detection(src=SRC_VISUAL, az=0.0, el=0.0, conf=-0.1)


def test_negative_range_is_rejected():
    with pytest.raises(ValueError):
        Detection(src=SRC_VISUAL, az=0.0, el=0.0, conf=0.5, r=-3.0)


def test_timestamp_defaults_to_now():
    import time

    before = time.time()
    det = Detection(src=SRC_VISUAL, az=0.0, el=0.0, conf=0.5)
    assert before <= det.t_utc <= time.time()
