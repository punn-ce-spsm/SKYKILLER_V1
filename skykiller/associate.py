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

Globally, but not by enumeration. This is a linear assignment problem, and it is
solved as one -- `scipy.optimize.linear_sum_assignment`, a modified
Jonker-Volgenant, over a square-padded cost matrix. The first version of this
file enumerated every partial matching instead, which is factorial and broke
inside the stated requirement: ten contacts a post is 234,662,231 matchings, and
even eight took 1.27 s against a 200 ms frame budget. See `_pad` for why the
padding is exactly equivalent to the partial matching it replaced, and
`_second_best` for the one thing the solver does not hand back.

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

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schemas import Detection
from .sites import SiteNetwork
from .triangulate import DEFAULT_MISS_GATE_SIGMAS, Fix, Ray, triangulate

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

#: Refuse more than this many detections per site in one frame.
#:
#: The assignment is no longer what limits this. What costs is *building* the
#: cost matrix -- one `triangulate` per candidate pair, at a measured 56 us,
#: which is quadratic. Whole-call timings, two posts, 200 m baseline:
#:
#:      contacts/post   total    cost matrix   solve + second-best
#:            2         0.4 ms      0.2 ms          0.01 ms
#:           10         5.0 ms      4.6 ms          0.06 ms
#:           20        19.4 ms     18.7 ms          0.36 ms
#:           30        44.1 ms     43.9 ms          0.65 ms
#:
#: So the solver is under 2% of the call at every size, and the cap follows the
#: geometry: 30 contacts a post is 44 ms of the 200 ms frame budget at 5 Hz,
#: three times the stated envelope of ten drones in view. Beyond it a frame is a
#: swarm, which is a different problem -- it wants bearing-space pre-gating so
#: that most pairs are never triangulated at all, not a faster assignment.
MAX_CONTACTS_PER_SITE = 30


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
                f"site {name} reported {len(ds)} contacts; association is "
                f"capped at {MAX_CONTACTS_PER_SITE} per site.")

    site_a, site_b = net.sites[name_a], net.sites[name_b]
    rays_a = [site_a.ray(d) for d in a_dets]
    rays_b = [site_b.ray(d) for d in b_dets]
    cost, fixes_by_pair = _cost_matrix(rays_a, rays_b, miss_gate_sigmas)

    # Leaving a detection unpaired is a legitimate outcome -- one post can be
    # occluded, or looking at a false positive -- so it needs a finite price.
    # Set at the gate: a pairing worse than the gate loses to no pairing.
    unpaired_price = _gate_scale(rays_a[0], rays_b[0], miss_gate_sigmas)
    # A separate, smaller scale for the ambiguity flag. Reusing the unpaired
    # price here was the first attempt and it called 90-99% of frames ambiguous,
    # which is a flag that tells the operator nothing.
    ambiguity_floor = _gate_scale(rays_a[0], rays_b[0], AMBIGUITY_SIGMAS)

    square, price = _pad(cost, unpaired_price)
    matching, best = _solve(square, cost)
    margin = _second_best(square, cost, matching, best, price) - best

    fixes = [fixes_by_pair[p] for p in matching]
    paired_a = {i for i, _ in matching}
    paired_b = {j for _, j in matching}
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


def _cost_matrix(rays_a: list[Ray], rays_b: list[Ray],
                 miss_gate_sigmas: float) -> tuple[np.ndarray, dict]:
    """Miss distance for every candidate pairing, and the fix behind each.

    An unpairable combination costs infinity here. That is the honest value and
    it is what the rest of this module reasons about; `_pad` swaps it for a
    finite sentinel only because the solver cannot be handed one.

    The fixes are kept rather than recomputed. The previous version triangulated
    every pair to score it and then triangulated the winners a second time to
    build the answer, which is the same expensive call twice and, worse, two
    places a gate could be applied differently.
    """
    n, m = len(rays_a), len(rays_b)
    cost = np.full((n, m), np.inf)
    fixes: dict[tuple[int, int], Fix] = {}
    for i, ra in enumerate(rays_a):
        for j, rb in enumerate(rays_b):
            fix = triangulate([ra, rb], miss_gate_sigmas=miss_gate_sigmas)
            if fix is not None:
                cost[i, j] = fix.miss_m
                fixes[(i, j)] = fix
    return cost, fixes


def _pad(cost: np.ndarray, unpaired_price: float) -> tuple[np.ndarray, float]:
    """Square the cost matrix so a partial matching becomes a full assignment.

    `linear_sum_assignment` returns a *complete* pairing, and forcing every
    contact to pair with something is exactly what makes a tracker invent
    targets when one post is occluded. The standard fix is to give each real row
    a dummy column to retire into, and each real column a dummy row, both priced
    at `unpaired_price`, with the dummy-dummy block free:

              cols 0..m-1        dummy cols (n of them)
        rows   +-------------+  +----------------------+
        0..n-1 |  miss cost  |  |    unpaired_price    |
               +-------------+  +----------------------+
        dummy  |unpaired_pric|  |          0           |
        (m)    +-------------+  +----------------------+

    An assignment with k real pairs then costs `sum(pairs) + (n+m-2k)*price` --
    identical, term for term, to the score the enumerating version computed. The
    dummy blocks are filled uniformly rather than diagonally on purpose: which
    dummy a row retires into is free choice with the same price either way, so
    constraining it would only add cells without changing any total.

    Two things the solver cannot be handed:

      - **Infinity.** `linear_sum_assignment` raises on a matrix with no finite
        feasible assignment, and it would happily let one infinity poison an
        otherwise good frame. Infeasible pairs get a finite sentinel that
        exceeds the whole all-unpaired assignment, so a single one costs more
        than giving up on every contact in the frame and can never be chosen.
      - **A degenerate reference geometry.** `_gate_scale` returns infinity when
        the two rays it measures will not triangulate at all. The sentinel takes
        over as the price, which says: pair on any finite evidence rather than
        retire everything on the strength of a scale we could not measure.
    """
    n, m = cost.shape
    finite = cost[np.isfinite(cost)]
    scale = max(float(finite.max()) if finite.size else 0.0,
                unpaired_price if math.isfinite(unpaired_price) else 0.0, 1.0)
    # Strictly greater than (n+m)*scale, which bounds every sentinel-free
    # assignment, so one sentinel cell always loses to the whole board retiring.
    sentinel = (n + m + 1) * (scale + 1.0)
    price = unpaired_price if math.isfinite(unpaired_price) else sentinel

    square = np.zeros((n + m, n + m))
    square[:n, :m] = np.where(np.isfinite(cost), cost, sentinel)
    square[:n, m:] = price
    square[n:, :m] = price
    return square, price


def _solve(square: np.ndarray, cost: np.ndarray,
           forbid: tuple[int, int] | None = None) -> tuple[list, float]:
    """The cheapest assignment, as real pairs and a total.

    `forbid` blocks one real pairing, which is how `_second_best` walks. The
    total is always read from the *unblocked* board, so a blocked cell could
    never flatter a runner-up -- it cannot be chosen anyway, because the dummy
    blocks always offer a complete alternative.
    """
    board = square
    if forbid is not None:
        board = square.copy()
        board[forbid] = np.inf
    rows, cols = linear_sum_assignment(board)
    n, m = cost.shape
    total = float(square[rows, cols].sum())
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols)
             if i < n and j < m and math.isfinite(cost[i, j])]
    return sorted(pairs), total


def _second_best(square: np.ndarray, cost: np.ndarray, matching: list,
                 best: float, price: float) -> float:
    """The cost of the best assignment that pairs things up differently.

    `margin_m` is the gap to this, and `ambiguous` gates track creation on it.
    Jonker-Volgenant returns the optimum and nothing else, so dropping it in
    without this would leave `margin_m` undefined and `ambiguous` permanently
    False -- silently removing the gate that stops a ghost reaching HOSTILE,
    which is the exact defect this file was written to fix.

    Any assignment other than the winner either drops one of its pairs or keeps
    all of them and adds another, so both are checked:

      - **Drops a pair.** Forbid each winning pair in turn and re-solve; the
        cheapest of those is the best assignment missing at least one. This is
        the first step of Murty's algorithm, at (k+1) solves.
      - **Adds a pair.** Every addition costs `cost[i,j] - 2*price`, which is
        non-negative because the winner would otherwise have taken it, so the
        cheapest strict superset adds exactly one pair -- the cheapest spare.

    Only real pairings are partitioned on. The dummy cells are interchangeable
    at the same price, so forbidding one yields a different permutation with an
    identical total, and a margin of zero on every frame.
    """
    rival = math.inf
    for pair in matching:
        rival = min(rival, _solve(square, cost, forbid=pair)[1])

    spare_a = set(range(cost.shape[0])) - {i for i, _ in matching}
    spare_b = set(range(cost.shape[1])) - {j for _, j in matching}
    for i in spare_a:
        for j in spare_b:
            if math.isfinite(cost[i, j]):
                rival = min(rival, best + cost[i, j] - 2 * price)
    return rival


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
