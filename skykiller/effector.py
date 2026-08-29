"""Where an effector must point, and the two human acts that let it.

This is the last module in the chain and the only one that produces something a
person acts on, so the interesting part is not the geometry -- that is a
subtraction and an arctangent -- but the gate in front of it.

**Nothing here transmits.** No code path in this file opens a radio, and
`EffectRequest` refuses to be constructed with `simulated=False`. What the
product hands over is an aiming solution; the customer's effector fires, under
the customer's authority. That is both the legal posture and the positioning:
we are the eyes and the decision support, not the weapon.

Two things in here are load-bearing and easy to get wrong.

**The bearing is computed from the effector, not from the observer.** Those are
different places -- a tethered mast and a ground emitter can be hundreds of
metres apart -- and the angle between them at a target is large when the target
is close, which is exactly when someone wants to shoot. Passing an observer's
bearing straight through would point the effector confidently at empty sky.
Re-projecting is the only reason the fusion stack needs a *position* at all
rather than just a bearing; a single-post system could never do it.

**The error ellipse has to fit inside the beam.** A jammer illuminates a cone,
so the question is not "how accurate is the fix" but "does the cone contain the
places the target might actually be". Those come apart: a 300 m error is fine
inside a 60 deg panel at 500 m and useless inside a 10 deg dish. `beam_covers`
answers the real question, and the answer travels in the request so the operator
sees it before deciding rather than after.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .schemas import IFF_HOSTILE, EffectRequest, Track
from .triangulate import to_bearing

#: A track must recede past this multiple of the envelope before the gate
#: considers it to have left. Without the margin a track sitting on the
#: boundary crosses it repeatedly on noise alone and raises a prompt each time,
#: which trains an operator to dismiss prompts.
EXIT_HYSTERESIS = 1.1

#: How long an ARM stays valid before it lapses and the operator must decide
#: again. An arming decision is about a situation, and situations move at
#: 15 m/s.
ARM_VALID_S = 30.0

#: Multiple of the position sigma the beam must cover. 2 sigma across the
#: ellipse's worst perpendicular axis is about 95% of where the target could be.
COVERAGE_SIGMAS = 2.0

# Engagement states. Absence of a state means the track is not engageable.
PROMPTED = "PROMPTED"        # in envelope and hostile; awaiting the operator
ARMED = "ARMED"              # operator armed it; awaiting authorisation
AUTHORISED = "AUTHORISED"    # request emitted


@dataclass(slots=True)
class Effector:
    """A surveyed emitter or interceptor: where it stands and what it covers.

    `beamwidth_deg` is the full cone angle, `envelope_m` the range beyond which
    it does nothing useful. Both come from the customer's equipment, not from
    us -- this module models their effector, it does not contain one.
    """

    id: str
    enu: np.ndarray
    tier: str
    beamwidth_deg: float
    envelope_m: float

    def __post_init__(self) -> None:
        self.enu = np.asarray(self.enu, dtype=float)
        if self.enu.shape != (3,):
            raise ValueError(f"effector {self.id}: enu must be 3 numbers")
        if not 0.0 < self.beamwidth_deg < 180.0:
            raise ValueError(f"effector {self.id}: beamwidth must be in (0,180)")
        if self.envelope_m <= 0:
            raise ValueError(f"effector {self.id}: envelope must be positive")

    def aim(self, target_enu) -> tuple[float, float, float]:
        """Azimuth, elevation and range to a target, *from here*."""
        v = np.asarray(target_enu, dtype=float) - self.enu
        az, el = to_bearing(v)
        return az, el, float(np.linalg.norm(v))

    def in_envelope(self, target_enu) -> bool:
        return self.aim(target_enu)[2] <= self.envelope_m

    def beam_width_m(self, range_m: float) -> float:
        """Diameter the beam subtends at a range. The error budget, literally."""
        return 2.0 * range_m * math.tan(math.radians(self.beamwidth_deg) / 2.0)

    def beam_covers(self, target_enu, cov, sigmas: float = COVERAGE_SIGMAS) -> bool:
        """Does the beam contain the target's error ellipse at this range?

        Only the error *across* the beam matters. Uncertainty along the boresight
        moves the target nearer or further inside the same cone, so projecting it
        out is not a simplification -- counting it would reject good solutions
        for an error that cannot cause a miss.
        """
        az, el, range_m = self.aim(target_enu)
        d = np.array([math.sin(math.radians(az)) * math.cos(math.radians(el)),
                      math.cos(math.radians(az)) * math.cos(math.radians(el)),
                      math.sin(math.radians(el))])
        perp = np.eye(3) - np.outer(d, d)
        across = perp @ np.asarray(cov, dtype=float) @ perp
        sigma_across = math.sqrt(max(float(np.linalg.eigvalsh(across).max()), 0.0))
        return sigmas * sigma_across <= self.beam_width_m(range_m) / 2.0


@dataclass(slots=True)
class _Engagement:
    state: str
    prompted_at: float
    operator: str | None = None
    armed_at: float | None = None


@dataclass(slots=True)
class RoeGate:
    """The human decision gate. Two deliberate acts, and either can be undone.

    The gate raises a prompt; it never arms or authorises anything itself. That
    is the whole point of it existing as a separate object: there is no argument
    and no configuration flag that makes this fire on its own.

    A standing arm is not a token to be spent later. It is continuously revoked
    by `update`: if the track stops being HOSTILE, leaves the envelope, is lost,
    or the operator takes too long, the decision is withdrawn and has to be made
    again against the situation as it now is.
    """

    effector: Effector
    arm_valid_s: float = ARM_VALID_S
    exit_hysteresis: float = EXIT_HYSTERESIS
    _engagements: dict[str, _Engagement] = field(default_factory=dict)

    def state(self, track_id: str) -> str | None:
        e = self._engagements.get(track_id)
        return e.state if e else None

    def update(self, tracks: list[Track], now: float | None = None) -> list[Track]:
        """Advance the gate. Returns tracks that have *newly* raised a prompt.

        Called every cycle with the whole picture, because withdrawal has to be
        driven by the same data as the prompt. A gate that only reacted to new
        detections would leave an arm standing against a track that had since
        been identified as ours.
        """
        now = time.time() if now is None else now
        by_id = {t.id: t for t in tracks}

        for track_id in list(self._engagements):
            track = by_id.get(track_id)
            if self._revoked(track, self._engagements[track_id], now):
                del self._engagements[track_id]

        new = []
        for track in tracks:
            if track.id in self._engagements:
                continue
            if track.iff != IFF_HOSTILE:
                continue
            if not self.effector.in_envelope(track.enu):
                continue
            self._engagements[track.id] = _Engagement(state=PROMPTED, prompted_at=now)
            new.append(track)
        return new

    def _revoked(self, track: Track | None, eng: _Engagement, now: float) -> bool:
        if track is None:
            return True                                   # the track is gone
        if track.iff != IFF_HOSTILE:
            return True                                   # no longer a target
        # Hysteresis only on the way out, so one crossing raises one prompt.
        if self.effector.aim(track.enu)[2] > self.effector.envelope_m * self.exit_hysteresis:
            return True
        if eng.armed_at is not None and now - eng.armed_at > self.arm_valid_s:
            return True                                   # the decision went stale
        return False

    def arm(self, track_id: str, operator: str, now: float | None = None) -> bool:
        """First human act. Returns False if there is no standing prompt to arm."""
        now = time.time() if now is None else now
        eng = self._engagements.get(track_id)
        if eng is None or eng.state != PROMPTED:
            return False
        eng.state, eng.operator, eng.armed_at = ARMED, operator, now
        return True

    def authorise(self, track: Track, operator: str,
                  now: float | None = None) -> EffectRequest | None:
        """Second human act. Returns the request, or None if it is not allowed.

        Deliberately takes the *track*, not an id: the aiming solution is built
        from the position as it is at the moment of authorisation, not from
        whatever it was when the prompt went up. A request carrying a stale
        position would be an audit record of a decision nobody made.
        """
        now = time.time() if now is None else now
        eng = self._engagements.get(track.id)
        if eng is None or eng.state != ARMED:
            return None
        if track.iff != IFF_HOSTILE:
            return None
        if now - (eng.armed_at or now) > self.arm_valid_s:
            del self._engagements[track.id]
            return None

        az, el, range_m = self.effector.aim(track.enu)
        eng.state = AUTHORISED
        return EffectRequest(
            track_id=track.id,
            jammer_id=self.effector.id,
            tier=self.effector.tier,
            aim_az=az, aim_el=el, range_m=range_m,
            in_envelope=range_m <= self.effector.envelope_m,
            beam_covers=self.effector.beam_covers(track.enu, track.cov),
            pos_cov=[list(map(float, row)) for row in track.cov],
            operator=operator,
            armed_at=eng.armed_at,
            authorised_at=now,
            t_utc=now,
        )
