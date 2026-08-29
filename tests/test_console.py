"""The recorded run and the page built from it.

Hermetic: no browser, no network. What these check is that the artefact handed
to a customer contains the pipeline's own numbers and the real gate's own
decisions -- not a second implementation of either that happens to agree today.
"""

import json
import re

import numpy as np
import pytest

from skykiller import console
from skykiller.effector import PROMPTED
from skykiller.harness import measure
from skykiller.record import GROUND_MAST_M, Scene, record


@pytest.fixture(scope="module")
def run():
    """One recording for the whole module -- it drives the real pipeline twice."""
    return record(Scene())


def _tracks(run):
    for r in run["runs"]:
        for frame in r["frames"]:
            for k in frame["k"]:
                yield r, frame, k


# --- the record is the pipeline's own output --------------------------------

def test_both_runs_are_recorded_at_the_pipelines_own_rate(run):
    """The browser indexes frames as round(t * hz), which is only correct if the
    timestamps really are uniform. Pin it."""
    for r in run["runs"]:
        assert len(r["frames"]) == int(run["meta"]["hz"] * run["meta"]["duration_s"])
        for i, frame in enumerate(r["frames"]):
            assert frame["t"] == pytest.approx(i / run["meta"]["hz"], abs=1e-6)


def test_each_frame_records_its_own_positions(run):
    """The guard against the subtlest bug available here.

    `AirPicture` mutates `Track` objects in place and `Frame.tracks` is a new
    list of the *same* objects, so holding a reference and reading it later
    yields the final position for every frame. If the recorder ever stopped
    serialising inside the loop, every frame would carry the last frame's
    position and the map would sit frozen with nothing raising.
    """
    frames = [f for f in run["runs"][1]["frames"] if f["k"]]
    first, last = frames[0]["k"][0]["e"], frames[-1]["k"][0]["e"]
    assert first != last
    assert abs(first[1] - last[1]) > 500      # it flew a long way, and we saw it


def test_the_error_series_is_the_harnesss_own_numbers(run):
    """The accuracy chart must be `harness.measure`'s series, not a re-derivation.

    Exact equality is expected, not approximate: both come from the same drive
    of the pipeline through `measure`'s `on_frame` hook.
    """
    for r in run["runs"]:
        per_frame = [f["err"] for f in r["frames"] if f["err"] is not None]
        assert per_frame == r["report"]["errors_m"]
        assert per_frame, "a run with no error samples cannot draw a chart"


def test_the_recorded_reports_match_a_fresh_measure(run):
    """The console and `skykiller demo` must never quote different numbers."""
    scene = Scene()
    for r, mast in ((run["runs"][0], GROUND_MAST_M), (run["runs"][1], scene.mast_m)):
        fresh = measure(scene.scenario(mast), "red-1", scene.effector)
        assert r["report"]["first_seen_t"] == pytest.approx(fresh.first_seen_t)
        assert r["report"]["declared_t"] == pytest.approx(fresh.declared_t)
        assert r["report"]["warning_s"] == pytest.approx(fresh.warning_s, abs=0.05)


def test_the_two_runs_differ_only_in_mast_height(run):
    """The demonstration's central claim, enforced rather than trusted."""
    ground, teth = run["runs"]
    assert ground["mast_m"] == GROUND_MAST_M
    assert teth["mast_m"] == Scene().mast_m
    assert len(ground["frames"]) == len(teth["frames"])
    # Same aircraft, same trajectory, same noise: truth is identical frame for frame.
    for a, b in zip(ground["frames"], teth["frames"]):
        assert a["tr"] == b["tr"]


def test_the_altitude_headline_survives_into_the_artefact(run):
    ground, teth = (r["report"] for r in run["runs"])
    assert ground["warning_s"] < 10.0
    assert teth["warning_s"] > 60.0
    assert teth["warning_s"] > 10 * ground["warning_s"]
    assert teth["error_at_envelope_m"] < ground["error_at_envelope_m"]


def test_the_jump_marks_land_on_measured_moments(run):
    marks = {m["label"]: m["t"] for m in run["meta"]["marks"]}
    ground, teth = (r["report"] for r in run["runs"])
    assert marks["tethered holds it"] == pytest.approx(teth["first_seen_t"])
    assert marks["declared hostile"] == pytest.approx(teth["declared_t"])
    assert marks["ground finally sees it"] == pytest.approx(ground["first_seen_t"])
    assert marks["envelope entry"] == pytest.approx(teth["envelope_entry_t"])


# --- the gate's decisions travel intact -------------------------------------

def test_every_recorded_request_is_simulated(run):
    """`EffectRequest.__post_init__` refuses anything else, so rebuilding each
    one is a mechanical re-assertion of the standing rule inside the artefact."""
    from skykiller.schemas import EffectRequest
    seen = 0
    for _, _, k in _tracks(run):
        if k["req"] is None:
            continue
        seen += 1
        assert k["req"]["simulated"] is True
        EffectRequest(pos_cov=[], **{x: y for x, y in k["req"].items()})
    assert seen > 0, "no request was recorded, so ARM would do nothing"


def test_a_request_exists_wherever_arm_will_be_offered(run):
    """The browser enables ARM exactly where `s == PROMPTED`. If any such frame
    lacked a request, AUTHORISE would be a button that does nothing."""
    prompted = 0
    for _, _, k in _tracks(run):
        if k["s"] == PROMPTED:
            prompted += 1
            assert k["req"] is not None, "PROMPTED frame with no request behind it"
    assert prompted > 0


def test_no_request_is_recorded_against_a_track_that_is_not_hostile(run):
    for _, _, k in _tracks(run):
        if k["req"] is not None:
            assert k["f"] == "H"


def test_probing_the_gate_does_not_disturb_the_run(run):
    """The recorder arms and authorises a *copy* of the gate to learn what a
    request would be. If it ever armed the live one, every later frame would
    read AUTHORISED and the operator would have nothing left to do."""
    states = {k["s"] for _, _, k in _tracks(run)}
    assert states <= {"", PROMPTED}, f"the live gate was advanced: {states}"


# --- the file is genuinely self-contained -----------------------------------

def test_the_page_carries_the_run_and_no_placeholder(run):
    page = console.build(run=run)
    assert console.SENTINEL not in page
    payload = re.search(r"const RUN = (.*?);\n", page, re.S).group(1)
    assert json.loads(payload)["meta"]["marks"] == run["meta"]["marks"]


def test_the_page_reaches_for_nothing(run):
    """It has to work on a laptop with no network, opened by double-click.

    A machine with a connection cannot tell you this is broken, which is why it
    is a test and not a browser check.
    """
    page = console.build(run=run)
    assert page.count("<script") == 1
    assert not re.search(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", page, re.I)
    assert "fonts.googleapis" not in page
    assert 'type="module"' not in page


def test_a_scene_with_no_terrain_still_produces_strict_json():
    """`lowest_visible_alt` returns -inf when nothing blocks the view, and
    `json.dumps` would emit a bare `Infinity`: invalid JSON, but valid
    JavaScript, so it would reach the page as a NaN coordinate and silently
    erase geometry."""
    run = record(Scene(treeline_m=0.0))
    assert run["meta"]["ground_floor_m"] is None
    json.dumps(run, allow_nan=False)          # raises if any non-finite survived
    console.build(run=run)


def test_a_recording_with_no_prompt_fails_loudly(monkeypatch):
    """A demonstration whose ARM button never lights is worse than no
    demonstration, so the build should fall over rather than ship one."""
    import skykiller.record as rec

    real = rec.record_run

    def no_prompts(scene, mast_m, label):
        out = real(scene, mast_m, label)
        for frame in out["frames"]:
            frame["p"] = []
        return out

    monkeypatch.setattr(rec, "record_run", no_prompts)
    with pytest.raises(RuntimeError, match="no ARM prompt"):
        rec.record(Scene())


def test_covariance_packing_round_trips():
    from skykiller.record import _pack_cov
    m = np.array([[4.0, 1.5, -0.5], [1.5, 9.0, 2.0], [-0.5, 2.0, 16.0]])
    packed = _pack_cov(m)
    assert packed == [4.0, 1.5, -0.5, 9.0, 2.0, 16.0]
    rebuilt = np.array([[packed[0], packed[1], packed[2]],
                        [packed[1], packed[3], packed[4]],
                        [packed[2], packed[4], packed[5]]])
    assert np.allclose(rebuilt, m)


def test_the_same_scene_records_byte_identically():
    """A half-megabyte artefact that changes on every regeneration makes its own
    git history useless. Identical inputs must give an identical file."""
    a, b = console.build(Scene(seed=3)), console.build(Scene(seed=3))
    assert a == b
    assert console.build(Scene(seed=4)) != a
