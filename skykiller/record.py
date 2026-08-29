"""Record a real pipeline run into a form a browser can replay.

Nothing here simulates a result. It drives `scenario.run` -- the same generator
the tests and the text demo use -- and writes down what actually happened, frame
by frame, including the ground truth the system was never told.

The one part worth explaining is how the rules-of-engagement gate survives the
trip into a browser. The gate is tested Python and it must not be reimplemented
in JavaScript: forking safety logic into an untested copy is exactly the kind of
divergence that gets discovered during a demonstration. So for every frame in
which a track is PROMPTED, a `deepcopy` of the live gate is armed and authorised,
and the genuine `EffectRequest` it returns is recorded against that frame. The
browser holds no ROE logic at all -- it offers ARM only where the recording says
the real gate said PROMPTED, and on authorisation it displays what Python
decided. The probe is a copy, so the live gate is left exactly as it was.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .effector import PROMPTED, Effector, RoeGate
from .geometry import Camera
from .harness import GHOST_RADIUS_M, Report, _nearest_track, measure
from .masking import lowest_visible_alt, treeline
from .scenario import Aircraft, Scenario, orbit, straight_line
from .sites import Site, SiteNetwork

#: Height of the customer's existing ground cameras, metres. The run is recorded
#: twice -- once at this height and once on the tether -- and the pair of runs is
#: the demonstration.
GROUND_MAST_M = 3.0

#: The camera on every post. One model, both runs: the comparison is only honest
#: if the sensor is identical and the height is the only difference.
DEMO_CAMERA = Camera(width=1280, height=720, hfov_deg=65.0)


@dataclass(slots=True)
class Scene:
    """Everything the two runs share. One source of truth for both the text
    demo and the console, so they can never drift into disagreeing."""

    mast_m: float = 200.0
    baseline_m: float = 200.0
    treeline_m: float = 20.0
    treeline_at_m: float = 200.0
    target_alt_m: float = 50.0
    speed_ms: float = 15.0
    sigma_deg: float = 0.5
    seed: int = 0

    @property
    def obstacles(self) -> list:
        return ([treeline("treeline", self.treeline_m, self.treeline_at_m)]
                if self.treeline_m > 0 else [])

    @property
    def effector(self) -> Effector:
        return Effector(id="J1", enu=np.array([self.baseline_m / 2, 0.0, 5.0]),
                        tier="T1", beamwidth_deg=60.0, envelope_m=500.0)

    @property
    def duration_s(self) -> float:
        """Long enough for the whole ingress: 1 km to the envelope edge at the
        target's own speed, plus a margin to measure dwell."""
        return 1000.0 / self.speed_ms + 30.0

    def ground_floor_m(self, range_m: float = 1000.0) -> float:
        """Lowest altitude a 3 m camera can see at a range, behind the terrain."""
        return lowest_visible_alt(np.array([0.0, 0.0, GROUND_MAST_M]),
                                  (0.0, range_m), self.obstacles)

    def scenario(self, mast_m: float) -> Scenario:
        """The scene observed from posts at a given height."""
        return Scenario(
            net=SiteNetwork({
                "A": Site("A", np.array([0.0, 0.0, mast_m]), DEMO_CAMERA,
                          sigma_deg=self.sigma_deg),
                "B": Site("B", np.array([self.baseline_m, 0.0, mast_m]), DEMO_CAMERA,
                          sigma_deg=self.sigma_deg)}),
            aircraft=[
                Aircraft(id="red-1", path=straight_line(
                    [0.0, 1500.0, self.target_alt_m], [0.0, -self.speed_ms, 0.0])),
                Aircraft(id="blue-1", friendly=True,
                         path=orbit([0.0, 400.0], radius=250.0, altitude=100.0)),
            ],
            duration_s=self.duration_s, sigma_deg=self.sigma_deg, seed=self.seed,
            obstacles=self.obstacles)


def _r(v, nd: int = 1) -> float | None:
    """Round, and turn a non-finite value into None.

    This is not defensive tidying. `json.dumps` emits bare `Infinity` and `NaN`
    by default: invalid JSON, and worse, *valid JavaScript* -- so an infinity
    would sail into the page, become a `NaN` SVG coordinate, and make geometry
    silently disappear with nothing logged anywhere. Reachable in practice:
    `lowest_visible_alt` returns `-inf` when there are no obstacles at all,
    which is exactly what `--treeline-m 0` asks for.
    """
    f = float(v)
    return round(f, nd) if math.isfinite(f) else None


def _pack_cov(cov) -> list[float]:
    """The six distinct entries of a symmetric 3x3, in row-major upper order.

    Halves the largest field in the record. The browser unpacks it to draw the
    error ellipse; nothing is lost because a covariance matrix is symmetric by
    construction.
    """
    m = np.asarray(cov, dtype=float)
    return [_r(m[a][b], 2) for a in range(3) for b in range(a, 3)]


def _track_record(track, gate: RoeGate, effector: Effector, now: float) -> dict:
    az, el, range_m = effector.aim(track.enu)
    state = gate.state(track.id) or ""
    rec: dict[str, Any] = {
        "i": track.id,
        "e": [_r(x) for x in track.enu],
        "c": _pack_cov(track.cov),
        "f": track.iff[0],                       # H / F / U
        "s": state,
        "v": [_r(x) for x in track.vel] if track.vel else None,
        "held": _r(track.age_s),
        "sig": _r(track.sigma_m),
        "aim": {"az": _r(az, 2), "el": _r(el, 2), "r": _r(range_m),
                "env": bool(effector.in_envelope(track.enu)),
                "cov": bool(effector.beam_covers(track.enu, track.cov))},
        "req": None,
    }
    if state == PROMPTED:
        rec["req"] = _probe_request(gate, track, now)
    return rec


def _probe_request(gate: RoeGate, track, now: float) -> dict | None:
    """What the real gate would emit if the operator acted on this frame.

    Runs against a `deepcopy`, so the live gate is untouched and the recording
    pass continues exactly as if nobody had asked. Returns the genuine
    `EffectRequest` -- `simulated=True` is enforced by the dataclass itself.
    """
    probe = copy.deepcopy(gate)
    if not probe.arm(track.id, operator="operator", now=now):
        return None
    request = probe.authorise(track, operator="operator", now=now)
    if request is None:
        return None
    d = asdict(request)
    # The covariance is already on the track this request belongs to; carrying
    # it twice is a third of the file for nothing.
    d.pop("pos_cov", None)
    for k in ("aim_az", "aim_el", "range_m", "armed_at", "authorised_at", "t_utc"):
        d[k] = _r(d[k], 2)
    return d


def record_run(scene: Scene, mast_m: float, label: str) -> dict:
    """One run: every frame, plus the measured report, from a single pass.

    The frames and the numbers come out of the *same* drive of the pipeline via
    `measure`'s `on_frame` hook. Recording and measuring separately would run it
    twice -- 13 s each -- and would leave room for the console and the text demo
    to quote figures that disagree.
    """
    effector = scene.effector
    scenario = scene.scenario(mast_m)
    frames: list[dict] = []

    def capture(frame, gate: RoeGate) -> None:
        truth = scenario.truth(frame.t)
        # The same helper, on the same frame, that `measure` uses to build
        # `errors_m` -- so the per-frame series and the report's series are the
        # same numbers rather than two implementations that agree by luck. The
        # time is what the chart needs: the ground run holds the target for only
        # the last third of the run, and plotting its errors by index would
        # stretch them across the whole width and flatter it enormously.
        held = _nearest_track(frame.tracks, truth["red-1"])
        err = (float(np.linalg.norm(np.array(held.enu) - truth["red-1"]))
               if held is not None else None)
        frames.append({
            "t": _r(frame.t),
            "err": _r(err) if err is not None else None,
            "k": [_track_record(t, gate, effector, frame.t) for t in frame.tracks],
            "p": [t.id for t in frame.prompts],
            "amb": bool(frame.ambiguous),
            "n": frame.n_fixes,
            "tr": {name: [_r(x) for x in pos] for name, pos in truth.items()},
        })

    report = measure(scenario, "red-1", effector, on_frame=capture)
    return {"label": label, "mast_m": mast_m, "frames": frames,
            "report": _report_record(report)}


def _report_record(report: Report) -> dict:
    """The measured report, rounded for display.

    Rounding matters here: `warning_s` is a difference of two floats and comes
    out as 4.3999999999999915, which is not a number to put in front of a
    customer.
    """
    d = {k: (_r(v) if isinstance(v, float) else v)
         for k, v in asdict(report).items() if k != "errors_m"}
    d["errors_m"] = [_r(e) for e in report.errors_m]      # the accuracy chart
    for name in ("warning_s", "dwell_s", "error_p50", "error_p95"):
        value = getattr(report, name)
        d[name] = None if value is None else _r(value)
    return d


def _marks(ground: dict, tethered: dict) -> list[dict]:
    """The four moments worth jumping to, taken from the measured reports."""
    g, t = ground["report"], tethered["report"]
    candidates = [
        (t.get("first_seen_t"), "tethered holds it"),
        (t.get("declared_t"), "declared hostile"),
        (g.get("first_seen_t"), "ground finally sees it"),
        (t.get("envelope_entry_t"), "envelope entry"),
    ]
    return [{"t": _r(v), "label": lbl} for v, lbl in candidates if v is not None]


def record(scene: Scene | None = None) -> dict:
    """Record the whole demonstration: the same target seen from both heights."""
    scene = scene or Scene()
    ground = record_run(scene, GROUND_MAST_M, f"ground cameras, {GROUND_MAST_M:.0f} m")
    tethered = record_run(scene, scene.mast_m, f"tethered observers, {scene.mast_m:.0f} m")

    # A recording in which the gate never prompts is a demonstration with a dead
    # ARM button, and the whole point of the console is that somebody operates
    # the gate. Whether a prompt happens depends on the target reaching HOSTILE
    # inside the envelope, which depends on the seed and on MIN_CONFIRM_HITS --
    # so it is worth failing the build loudly rather than finding out in a room.
    prompts = sum(len(f["p"]) for r in (ground, tethered) for f in r["frames"])
    if prompts == 0:
        raise RuntimeError(
            "no ARM prompt occurred in either run, so the console would have "
            "nothing to engage. Check the effector envelope and the seed.")

    net = scene.scenario(scene.mast_m).net
    return {
        "meta": {
            # Deliberately no wall-clock stamp. A recorded run of a synthetic
            # scenario has no meaningful real-world time -- every timestamp
            # inside it is scenario-relative -- and stamping one would make a
            # half-megabyte artefact change on every regeneration, so a git diff
            # could never tell you whether anything actually changed.
            "seed": scene.seed,
            "hz": 5.0,
            "duration_s": _r(scene.duration_s),
            "sigma_deg": scene.sigma_deg,
            "baseline_m": scene.baseline_m,
            "mast_m": scene.mast_m,
            "ground_mast_m": GROUND_MAST_M,
            "target_alt_m": scene.target_alt_m,
            "speed_ms": scene.speed_ms,
            "ground_floor_m": _r(scene.ground_floor_m()),
            "sites": {name: [_r(x) for x in s.enu] for name, s in net.sites.items()},
            "detect_ring_m": _r(net.max_range_m()),
            "effector": {
                "id": scene.effector.id,
                "enu": [_r(x) for x in scene.effector.enu],
                "tier": scene.effector.tier,
                "beamwidth_deg": scene.effector.beamwidth_deg,
                "envelope_m": scene.effector.envelope_m,
            },
            "obstacles": [{"name": o.name, "a": [_r(x) for x in o.a],
                           "b": [_r(x) for x in o.b], "crest_m": _r(o.crest_m)}
                          for o in scene.obstacles],
            "marks": _marks(ground, tethered),
        },
        "runs": [ground, tethered],
    }
