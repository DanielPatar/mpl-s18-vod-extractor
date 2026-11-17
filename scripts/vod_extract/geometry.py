"""Slot geometry of the MPL ID draft bar, measured per frame (canonical 1280x720).

Why this is measured per frame
------------------------------
Real frames of m1113-g4 show that slot widths change with state (a slot whose
turn is active is wide, e.g. 72/72/70/138/138 px at 05:30; 78/143/114/77/78 px
mid-draft; 98 px each when idle). A fixed 5-way split of the strip is therefore
wrong. What does stay put is the player-name plate under every slot, centred in
its slot in every state. With centres c_k and outer edges b_0, b_5:

    c_k = (b_{k-1} + b_k) / 2      (k = 1..5)

which is solved by least squares for b_1..b_4. On the two real frames this
reproduced boundaries measured independently from image gradients to within a
few pixels.

What is validated and what is not
---------------------------------
Per frame the fit is accepted only if there are exactly 5 name clusters, the
fit residual is small and the implied slot widths are plausible. Otherwise the
side has NO geometry in that frame and nothing is read from it. The nominal
outer edges below were measured on two real frames (720p, and 1080p scaled to
720p) of one broadcast layout; other layouts must be re-measured. Use
`extract_picks.py --probe-at` to print this geometry on any VOD.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Outer edges of each 5-slot strip in canonical px (left edge, right edge).
NOMINAL_OUTER = {"left": (8.0, 500.0), "right": (781.0, 1273.0)}
NAME_ROWS = (697, 709)         # player-name plate text rows
NAME_THR = 170                 # white text: min(B,G,R) above this
NAME_MIN_AREA = 40
NAME_MIN_W = 18
MIN_SLOT_W, MAX_SLOT_W = 45.0, 190.0
MAX_RMS = 7.0                  # px, residual of the centre equations
LABEL_LEFT_OFFSET = 4.0        # label x is ~4 px right of its slot's left edge (7 labels: offset -2.0 +-0.2 at 6.0)
ASSIGN_GATE = 20.0             # max px between (label.x - offset) and the chosen left boundary


@dataclass
class SideGeometry:
    side: str                          # "left" | "right" (screen side)
    ok: bool
    reason: str | None = None
    centers: list = field(default_factory=list)
    bounds: list = field(default_factory=list)   # b_0..b_5
    rms: float | None = None


def name_centers(canon: np.ndarray, x0: int, x1: int) -> list[float]:
    roi = canon[NAME_ROWS[0]:NAME_ROWS[1], x0:x1]
    if roi.size == 0:
        return []
    m = ((roi.min(axis=2) > NAME_THR) * 255).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = [x0 + x + w / 2.0 for x, _y, w, _h, a in st[1:] if a > NAME_MIN_AREA and w >= NAME_MIN_W]
    return sorted(out)


def fit_bounds(centers: list[float], outer: tuple[float, float]):
    """Least-squares b_1..b_4 from b_{k-1}+b_k = 2 c_k with b_0, b_5 fixed."""
    b0, b5 = outer
    A = np.zeros((5, 4))
    y = np.array([2.0 * c for c in centers])
    for k in range(5):
        if k - 1 >= 0:
            A[k, k - 1] = 1.0
        if k <= 3:
            A[k, k] = 1.0
    y[0] -= b0
    y[4] -= b5
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    bounds = [b0, *[float(v) for v in sol], b5]
    pred = A @ sol
    rms = float(np.sqrt(np.mean((pred - y) ** 2)) / 2.0)       # in px of centre error
    return bounds, rms


def side_geometry(canon: np.ndarray, side: str) -> SideGeometry:
    b0, b5 = NOMINAL_OUTER[side]
    cs = name_centers(canon, int(b0), int(b5))
    if len(cs) != 5:
        return SideGeometry(side, False, f"expected 5 name plates, found {len(cs)}", cs)
    bounds, rms = fit_bounds(cs, (b0, b5))
    widths = np.diff(bounds)
    if rms > MAX_RMS:
        return SideGeometry(side, False, f"centre fit residual {rms:.1f}px > {MAX_RMS}", cs, bounds, rms)
    if (widths < MIN_SLOT_W).any() or (widths > MAX_SLOT_W).any():
        return SideGeometry(side, False, "implausible slot widths " + str([round(float(w)) for w in widths]),
                            cs, bounds, rms)
    return SideGeometry(side, True, None, cs, bounds, rms)


def slot_of_label(geom: SideGeometry, label_x: float):
    """Return (slot_index 1..5 counted from the LEFT of the strip, offset_px) or
    (None, offset_px) when the label does not sit near any slot's left edge.

    A label is drawn at the left edge of its own slot, so the slot is the one
    whose LEFT boundary is nearest to (label_x - LABEL_LEFT_OFFSET). Adjacent
    left boundaries are >= 45 px apart, so a few px of fit error cannot move a
    label to a neighbour, while the gate rejects labels that sit mid-slot."""
    implied = label_x - LABEL_LEFT_OFFSET
    lefts = geom.bounds[:5]
    k = int(np.argmin([abs(implied - b) for b in lefts]))
    off = implied - lefts[k]
    return (k + 1 if abs(off) <= ASSIGN_GATE else None), float(off)
