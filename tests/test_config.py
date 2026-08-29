"""Config loading, device selection, and the imgsz rule.

The imgsz tests exist because of a measured bug, not a hypothetical one: with
imgsz=640 against a 1280-wide capture, the drone detector found a 150 px target
in 0 of 60 frames. At imgsz=1280 it found it in 60 of 60. Downscaling throws
away exactly the pixels a distant drone is made of.
"""

import pytest

from skykiller import l2_visual
from skykiller.config import Config, _merge, load, resolve_device


def test_explicit_imgsz_is_honoured(monkeypatch):
    cfg = Config()
    cfg.model.imgsz = 960
    monkeypatch.setattr(l2_visual, "_probe_source", lambda _s: (1920, 1080))
    assert l2_visual._resolve_imgsz(cfg, "clip.mp4") == 960


def test_auto_imgsz_matches_the_capture_width(monkeypatch):
    cfg = Config()  # imgsz defaults to "auto"
    monkeypatch.setattr(l2_visual, "_probe_source", lambda _s: (1280, 720))
    assert l2_visual._resolve_imgsz(cfg, "clip.mp4") == 1280


def test_auto_imgsz_rounds_to_a_multiple_of_32(monkeypatch):
    cfg = Config()
    monkeypatch.setattr(l2_visual, "_probe_source", lambda _s: (1600, 900))
    got = l2_visual._resolve_imgsz(cfg, "clip.mp4")
    assert got % 32 == 0
    assert got == 1600


def test_auto_imgsz_uses_the_longest_side_for_portrait_video(monkeypatch):
    """A phone held upright is taller than it is wide; imgsz is the long side."""
    cfg = Config()
    monkeypatch.setattr(l2_visual, "_probe_source", lambda _s: (720, 1280))
    assert l2_visual._resolve_imgsz(cfg, "clip.mp4") == 1280


def test_auto_imgsz_falls_back_when_the_source_cannot_be_probed(monkeypatch):
    cfg = Config()
    monkeypatch.setattr(l2_visual, "_probe_source", lambda _s: None)
    assert l2_visual._resolve_imgsz(cfg, "0") == 1280


def test_auto_imgsz_never_goes_below_the_floor(monkeypatch):
    cfg = Config()
    monkeypatch.setattr(l2_visual, "_probe_source", lambda _s: (64, 48))
    assert l2_visual._resolve_imgsz(cfg, "tiny.mp4") == 320


def test_device_auto_prefers_mps_on_apple_silicon():
    """This machine is an M5. Selecting only on CUDA would land it on the CPU."""
    assert resolve_device("auto") in {"mps", "cuda", "cpu"}
    assert resolve_device("cpu") == "cpu"


def test_unknown_config_keys_are_rejected_not_ignored():
    """A silently dropped setting is a config that lies about the system."""
    cfg = Config()
    with pytest.raises(ValueError, match="unknown config key"):
        _merge(cfg, {"nonsense": 1})
    with pytest.raises(ValueError, match="unknown config key"):
        _merge(cfg, {"camera": {"hfov": 65.0}})  # real section, wrong key


def test_nested_overlay_leaves_siblings_alone():
    cfg = Config()
    _merge(cfg, {"camera": {"hfov_deg": 42.0}})
    assert cfg.camera.hfov_deg == 42.0
    assert cfg.camera.az_deg == 0.0
    assert cfg.model.conf == 0.25


def test_shipped_config_file_loads_and_matches_defaults():
    """configs/l2.yaml must stay parseable and agree with the dataclasses."""
    cfg = load()
    assert cfg.model.imgsz == "auto"
    assert cfg.tracker == "botsort.yaml"
    assert cfg.lane.target_width_m is None
    assert cfg.sink.mqtt.enabled is False


def test_null_weights_is_a_choice_not_a_missing_file(monkeypatch, capsys):
    """configs/hometest.yaml sets weights: null.

    The CLI used to do `REPO_ROOT / cfg.model.weights` unconditionally, which
    raised TypeError on None before the lane ever started. It must also not
    print the fetch-model hint: choosing COCO is not a missing download.
    """
    import skykiller.l2_visual as lv
    from skykiller import cli

    cfg = load("configs/hometest.yaml")
    assert cfg.model.weights is None
    assert cfg.model.classes == [0, 39, 41, 67]

    seen = {}

    def _fake_run(c):
        seen["cfg"] = c
        return 0

    monkeypatch.setattr(lv, "run", _fake_run)
    assert cli.main(["--config", "configs/hometest.yaml", "--no-show"]) == 0
    assert seen["cfg"].model.weights is None
    assert seen["cfg"].show is False
    assert "hint: run" not in capsys.readouterr().err


def test_missing_weights_still_warns(monkeypatch, capsys, tmp_path):
    """The opposite case: weights named but absent is a real problem, so warn."""
    import skykiller.l2_visual as lv
    from skykiller import cli

    conf = tmp_path / "c.yaml"
    conf.write_text('model:\n  weights: "models/absent.pt"\n')
    monkeypatch.setattr(lv, "run", lambda _c: 0)
    cli.main(["--config", str(conf), "--no-show"])
    assert "hint: run" in capsys.readouterr().err


def test_fetch_model_has_a_destination_even_with_null_weights():
    from skykiller import cli

    cfg = load("configs/hometest.yaml")
    assert (cfg.model.weights or cli.DEFAULT_WEIGHTS_PATH) == "models/drone-yolo11x.pt"
