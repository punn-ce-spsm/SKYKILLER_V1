"""Turn a recorded run into one HTML file that opens by double-click.

Deliberately not a web app. A demonstration you can only give with a terminal
open and a server running is a demonstration you cannot leave behind, and this
one has to survive being emailed to somebody who will open it on a laptop with
no network. So: no server, no dependency, no fetch, no external font, no
module script -- one `<script>` with the run pasted into it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .record import Scene, record

TEMPLATE = Path(__file__).parent / "assets" / "console.html"
SENTINEL = "/*__RUN__*/null"

#: Anything that would make the page reach out to the network. Checked rather
#: than assumed, because a machine with a connection cannot tell you it failed.
_EXTERNAL = re.compile(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", re.I)


def build(scene: Scene | None = None, run: dict | None = None) -> str:
    """The finished page as a string."""
    run = run if run is not None else record(scene)
    # allow_nan=False is the assertion, not the formatting. json.dumps happily
    # emits bare `Infinity`, which is invalid JSON but valid JavaScript, and
    # would arrive in the page as a NaN coordinate that draws nothing.
    payload = json.dumps(run, separators=(",", ":"), allow_nan=False)

    template = TEMPLATE.read_text()
    if SENTINEL not in template:
        raise RuntimeError(f"{TEMPLATE} has no {SENTINEL} placeholder to fill")
    page = template.replace(SENTINEL, payload)

    external = _EXTERNAL.findall(page)
    if external:
        raise RuntimeError(
            f"the page references {len(external)} external resource(s); it must "
            f"work with no network at all")
    return page


def write(path: str | Path, scene: Scene | None = None,
          run: dict | None = None) -> Path:
    """Write the page and return where it went."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(scene, run))
    return out
