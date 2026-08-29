"""Which of site A's contacts is which of site B's.

The gap this fills is easy to miss and it breaks the product without it.
`SiteNetwork.rays()` converts every detection on the bus into a ray, and
`triangulate()` assumes every ray it is handed looks at the same object. Feed it
one frame containing two aircraft and it silently fuses four rays into one
position that is neither of them.

Worse than silence: with two targets there are four candidate pairings, two real
and two crossed, and the crossed ones intersect too. Measured on a two-post
100 m baseline with targets at 1.0 and 1.6 km, a crossed pairing produced a
'target' at (11, 886, 83) reporting a 98 m sigma -- *tighter* than the real
target's 124 m. A ghost that looks more credible than the thing it is made of
will be tracked, will fail to correlate with the friendly feed, will serve the
hostile hold, and will be handed to a jammer as a target that does not exist.

So pairings are chosen globally, not greedily: score every consistent assignment
by its total miss distance and take the best. That is correct whenever the
geometry separates the targets at all.

**Where it stops working, stated plainly.** Two targets at the same altitude on
close bearings are nearly indistinguishable from two posts. Measured on the hard
case above, the true assignment beat the ghost by 9 m of total miss -- while
bearing noise alone contributes up to about 16 m of miss at 1 km. The signal is
smaller than the noise, and no scoring rule fixes that; it is the geometry.

Two things follow, and both are implemented rather than hoped for:

  - The margin between the best and second-best assignment is *reported*, so an
    ambiguous frame is visible instead of being quietly resolved by a coin toss.
    A fix the operator cannot trust must not reach an effector labelled the same
    as one they can.
  - Temporal continuity is the real discriminant. A ghost jumps frame to frame
    while a real aircraft flies a smooth line, so `AirPicture`'s prediction gate
    starves it -- but only if the ambiguity is not laundered into a confident
    single answer first.

A third post removes the problem outright; see `triangulate`'s notes.

**One deployment consequence, and it is not a corner case.** The degeneracy is
worst when the targets fly at *the same height as the masts*: every ray then
lies in one horizontal plane, every pairing intersects exactly, and the choice
falls to floating-point noise. Measured with 80 m masts and both targets at
80 m, the crossed assignment won every time and the margin was 6e-13 m. Since a
tethered observer sits at 80-120 m and the drones it hunts fly at much the same
height, this is the *expected* geometry, not an unlucky one.

Two mitigations, in order of how much they buy:

  - **Spread the masts.** A 200 m baseline paired correctly in 90-100% of
    frames where 100 m managed roughly half. This is free and it is the single
    biggest lever, the same conclusion `SiteNetwork.max_range_m` reaches from
    the separation gate.
  - **Fly the observers above the traffic.** Any height difference tilts the
    rays out of the common plane and gives the assignment something to work
    with. Equal heights is the one case with no signal at all.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .schemas import Detection
from .sites import SiteNetwork
from .triangulate import DEFAULT_MISS_GATE_SIGMAS, Fix, Ray, _miss_distance, triangulate

#: How large the winning margin must be, in units of the miss that bearing
#: noise produces anyway, before the pairing is called unambiguous. Calibrated
#: by measurement across five geometries at sigma 0.5 deg, 2000 frames:
#:
#:      threshold   frames called confident   of those, correctly paired
#:          0.5              78%                        84%
#:          1.0              63%                        89%
#:          2.0              44%                        97%
#:          3.0              31%                        99%
#:
#: 3.0 is the operating point. 2.0 was tried first, on the reasoning that a
#: ghost slipping through 1.4% of frames could not hold a track anyway. Running
#: the two-aircraft scenario end to end showed that reasoning was wrong: a
#: wrong pairing that recurs even rarely lands near its previous self often
#: enough to accumulate confirmations, and the run still finished with a ghost
#: declared HOSTILE.
#:
#: Being conservative here is nearly free in the other direction. An ambiguous
#: frame is not discarded -- it can still update an established track, it just
#: cannot create one -- so a high threshold costs a short delay before a new
#: target is declarable, and buys a much cleaner picture.
AMBIGUITY_SIGMAS = 3.0

#: Refuse to brute-force beyond this many detections per site. The number of
#: assignments grows factorially, and a frame with this many contacts is a
#: different problem (a swarm) needing a different algorithm.
MAX_CONTACTS_PER_SITE = 8


@dataclass(slots=True)
class Association:
    """One frame's worth of fused targets, and how much to trust the pairing."""

    fixes: list[Fix]
    margin_m: float           # total-miss gap to the next-best assignment
    ambiguous: bool           # margin smaller than the noise that produced it
    unpaired: list[Detection] # seen by one site only: real, but not locatable
    #: Which detection paired with which, as (index into site A, index into
    #: site B) using each site's detections in bus order. Carried so a console
    #: can show the operator what was joined to what, and so the choice can be
    #: checked directly instead of inferred from where the answer landed.
    matching: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_targets(self) -> int:
        return len(self.fixes)


def associate(
    net: SiteNetwork,
    dets: list[Detection],
    miss_gate_sigmas: float = DEFAULT_MISS_GATE_SIGMAS,
) -> Association:
    """Pair one frame's detections across two posts and fuse each pair.

    Detections from sites the network does not know are dropped upstream by
    `SiteNetwork.rays`; here they are dropped the same way and reported as
    unpaired, so a misconfigured site never becomes an invented target.
    """
    known = [d for d in dets if d.src in net.sites]
    by_site: dict[str, list[Detection]] = {}
    for d in known:
        by_site.setdefault(d.src, []).append(d)

    if len(by_site) < 2:
        # One post sees everything, or nothing does. Bearings without parallax
        # are not positions, and saying so beats returning an empty success.
        return Association(fixes=[], margin_m=0.0, ambiguous=False, unpaired=known)
    if len(by_site) > 2:
        raise NotImplementedError(
            f"associate() pairs two posts; got {sorted(by_site)}. A third post "
            f"needs sequential assignment, which is a real extension, not a "
            f"loop -- see the module docstring.")

    (name_a, a_dets), (name_b, b_dets) = sorted(by_site.items())
    for name, ds in ((name_a, a_dets), (name_b, b_dets)):
        if len(ds) > MAX_CONTACTS_PER_SITE:
            raise NotImplementedError(
                f"site {name} reported {len(ds)} contacts; the brute-force "
                f"assignment is capped at {MAX_CONTACTS_PER_SITE}.")

    site_a, site_b = net.sites[name_a], net.sites[name_b]
    # Cost of every candidate pairing, once. An unpairable combination costs
    # infinity so it can never be chosen, however good the rest of the
    # assignment looks.
    cost: dict[tuple[int, int], float] = {}
    for i, da in enumerate(a_dets):
        for j, db in enumerate(b_dets):
            rays = [site_a.ray(da), site_b.ray(db)]
            fix = triangulate(rays, miss_gate_sigmas=miss_gate_sigmas)
            cost[(i, j)] = float("inf") if fix is None else fix.miss_m

    # Leaving a detection unpaired is a legitimate outcome -- one post can be
    # occluded, or looking at a false positive -- so it needs a finite price.
    # Set at the gate: a pairing worse than the gate loses to no pairing.
    unpaired_price = _gate_scale(site_a.ray(a_dets[0]), site_b.ray(b_dets[0]),
                                 miss_gate_sigmas)
    # A separate, smaller scale for the ambiguity flag. Reusing the unpaired
    # price here was the first attempt and it called 90-99% of frames ambiguous,
    # which is a flag that tells the operator nothing.
    ambiguity_floor = _gate_scale(site_a.ray(a_dets[0]), site_b.ray(b_dets[0]),
                                  AMBIGUITY_SIGMAS)

    scored = sorted(
        (_score(m, cost, len(a_dets), len(b_dets), unpaired_price), m)
        for m in _matchings(len(a_dets), len(b_dets))
    )
    best_cost, best = scored[0]
    margin = (scored[1][0] - best_cost) if len(scored) > 1 else float("inf")

    fixes, paired_a, paired_b, matching = [], set(), set(), []
    for i, j in best:
        fix = triangulate([site_a.ray(a_dets[i]), site_b.ray(b_dets[j])],
                          miss_gate_sigmas=miss_gate_sigmas)
        if fix is not None:
            fixes.append(fix)
            matching.append((i, j))
            paired_a.add(i)
            paired_b.add(j)

    unpaired = ([d for i, d in enumerate(a_dets) if i not in paired_a]
                + [d for j, d in enumerate(b_dets) if j not in paired_b])
    unpaired += [d for d in dets if d.src not in net.sites]

    return Association(
        fixes=fixes,
        margin_m=margin,
        # Ambiguous when the winning margin is inside the miss that honest
        # bearing noise produces anyway -- i.e. when the ranking is noise.
        ambiguous=margin < ambiguity_floor,
        unpaired=unpaired,
        matching=matching,
    )


def _gate_scale(ray_a: Ray, ray_b: Ray, sigmas: float) -> float:
    """The miss that noise alone would produce for this geometry, times the gate.

    Reuses the same scaling `triangulate` gates on, so 'ambiguous' means
    'the assignment ranking is within the solver's own noise floor'.
    """
    fix = triangulate([ray_a, ray_b], miss_gate_sigmas=float("inf"))
    if fix is None:
        return float("inf")
    rms = math.sqrt(sum((math.radians(r.sigma_deg)
                         * float(((fix.enu - r.origin) ** 2).sum() ** 0.5)) ** 2
                        for r in (ray_a, ray_b)) / 2)
    return sigmas * rms


def _matchings(n: int, m: int) -> list[list[tuple[int, int]]]:
    """Every partial one-to-one pairing of n things with m things.

    Partial, not total: a detection may legitimately go unpaired, and forcing a
    complete matching is what makes a greedy tracker invent targets when one
    post is occluded.
    """
    out: list[list[tuple[int, int]]] = []
    for k in range(min(n, m) + 1):
        for a_sel in itertools.combinations(range(n), k):
            for b_sel in itertools.permutations(range(m), k):
                out.append(list(zip(a_sel, b_sel)))
    return out


def _score(matching: list[tuple[int, int]], cost: dict[tuple[int, int], float],
           n: int, m: int, unpaired_price: float) -> float:
    total = sum(cost[p] for p in matching)
    if math.isinf(total):
        return float("inf")
    return total + unpaired_price * (n + m - 2 * len(matching))
