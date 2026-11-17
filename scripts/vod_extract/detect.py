"""Per-frame detection: which hero-name labels are on screen, on which side."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import labels as L

# Prior measured on real frames (true labels sat at y=557-558, h=43-52 in the
# canonical frame). Used only to discard bright-art false positives early.
LABEL_Y_RANGE = (550, 568)
LABEL_H_RANGE = (36, 60)

MIN_SCORE = 0.70        # best vocabulary similarity required to accept a read
MIN_MARGIN = 0.15       # best minus second best
 

@dataclass
class Detection:
    side: str                 # blue | red (after applying left_is)
    hero: str                 # best vocabulary match (lower-case name as in heroes.json)
    score: float
    margin: float
    in_candidates: bool       # hero is one of the 5 source picks for that side
    box: tuple                # x, y, w, h (canonical px)
    raws: list = field(default_factory=list)


class LabelCache:
    """Remember the OCR result for a label so it is read once, not every frame.

    A locked label stays on screen for the rest of the draft, so almost every
    frame after the first is a cache hit. A hit needs the same place (+-3 px)
    and near-identical pixels; a different word at that place misses and is
    read again. Garbage boxes (bright art) are cached too (value None)."""

    def __init__(self, tol: float = 14.0):
        self.tol = tol
        self.items = []          # (x, y, w, h, patch, value)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _patch(canon, box):
        x, y, w, h = box
        return cv2.resize(canon[y:y + h, x:x + w].min(axis=2), (10, 40), interpolation=cv2.INTER_AREA).astype(np.float32)

    def get(self, canon, box):
        p = self._patch(canon, box)
        for x, y, w, h, q, val in self.items:
            if abs(x - box[0]) <= 3 and abs(y - box[1]) <= 3 and abs(w - box[2]) <= 3 and abs(h - box[3]) <= 4:
                if float(np.abs(p - q).mean()) < self.tol:
                    self.hits += 1
                    return True, val
        self.misses += 1
        return False, None

    def put(self, canon, box, val):
        self.items.append((*box, self._patch(canon, box), val))


def scan_boxes(canon: np.ndarray, vocab: list[str], candidates: dict[str, set],
               left_is: str = "blue", cache: "LabelCache | None" = None):
    """Every label-shaped box on the bar with its OCR outcome.

    Returns [(strip, (x, y, w, h), Detection | None)]. `strip` is the screen
    side ("left"/"right"). A None Detection means the box looked like a label
    but could not be read with enough confidence ("unreadable"); this differs
    from "no box at all" (nothing there), and callers must not conflate them."""
    out = []
    for box in L.find_label_boxes(canon):
        if not (LABEL_Y_RANGE[0] <= box.y <= LABEL_Y_RANGE[1] and LABEL_H_RANGE[0] <= box.h <= LABEL_H_RANGE[1]):
            continue
        key = (box.x, box.y, box.w, box.h)
        side = L.side_of(box, left_is)
        if cache is not None:
            hit, val = cache.get(canon, key)
            if hit:
                out.append((box.side_strip, key, None if val is None else _with_side(val, side, candidates)))
                continue
        raws, ranked = L.read_and_rank(canon, box, vocab)
        val = _accept(raws, ranked, candidates, side, key)
        if cache is not None:
            cache.put(canon, key, val)
        out.append((box.side_strip, key, val))
    return out


def detect_labels(canon: np.ndarray, vocab: list[str], candidates: dict[str, set],
                  left_is: str = "blue", cache: "LabelCache | None" = None) -> list[Detection]:
    """Readable labels only (kept for the evaluation script and tests)."""
    return [d for _s, _k, d in scan_boxes(canon, vocab, candidates, left_is, cache) if d is not None]


def _accept(raws, ranked, candidates, side, key):
    if not raws:
        return None
    (best, s1), (_, s2) = ranked[0], ranked[1]
    # Prefer a candidate of this side when it is nearly as good as the best
    # overall match (OCR confusions between look-alike names).
    cand_here = [(n, s) for n, s in ranked if n.lower() in candidates.get(side, set())]
    if cand_here and cand_here[0][0] != best and s1 - cand_here[0][1] < 0.05:
        best, s1 = cand_here[0]
        s2 = max((s for n, s in ranked if n != best), default=0.0)
    if s1 < MIN_SCORE or (s1 - s2) < MIN_MARGIN:
        return None
    return Detection(side, best.lower(), round(s1, 3), round(s1 - s2, 3),
                     best.lower() in candidates.get(side, set()), key, raws)


def _with_side(det, side, candidates):
    """Re-label a cached detection for the current frame (side/candidate flags)."""
    return Detection(side, det.hero, det.score, det.margin,
                     det.hero in candidates.get(side, set()), det.box, det.raws)
