"""Find where the draft bar is on screen, from what the frames look like.

No fixed timestamp is assumed. The bar is recognised by the same test the
slot reader uses to accept a frame: five player-name plates on each panel with
a plausible slot geometry. Gameplay, title cards, camera cut-aways that cover
the bar, and loading screens fail that test and are skipped.
"""
from __future__ import annotations

from . import geometry as G


def layout_present(canon) -> bool:
    return all(G.side_geometry(canon, s).ok for s in ("left", "right"))


def find_draft_start(sample, lo: float, hi: float, step: float = 15.0, confirm_gap: float = 2.0, burst=None):
    """Walk lo..hi in `step` seconds and return the first time the layout is
    present AND still present `confirm_gap` / `2*confirm_gap` seconds later
    (>= 2 of those 3 samples), or None.

    sample(t) -> canonical frame or None. burst(t, gaps) -> list of frames at
    t+gap, ideally from ONE decoder launch (see Source.burst); without it the
    confirmation falls back to separate sample() calls.
    Returns (found_at | None, probes); probes = [(t, bool)] is the audit trail."""
    probes = []
    gaps = [confirm_gap, 2 * confirm_gap]
    t = lo
    while t <= hi + 1e-9:
        fr = sample(t)
        ok = fr is not None and layout_present(fr)
        probes.append((round(t, 1), bool(ok)))
        if ok:
            frames = burst(t, gaps) if burst is not None else [sample(t + g) for g in gaps]
            hits = 1
            for g, f2 in zip(gaps, frames):
                good = f2 is not None and layout_present(f2)
                probes.append((round(t + g, 1), bool(good)))
                hits += int(good)
            if hits >= 2:
                return t, probes
        t += step
    return None, probes
