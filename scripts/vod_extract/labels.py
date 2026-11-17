"""Detect and read the vertical hero-name labels in the MPL ID draft bar.

Why labels (and not hero-art matching)
--------------------------------------
Observed on real frames of m1113-g4 (see docs/VOD_EXTRACTOR.md):

* a pick slot goes placeholder -> preview (wide, hero art, NO name text) ->
  locked (compact, vertical name text);
* slot positions during the draft differ from the final lane order;
* matching broadcast portraits to Liquipedia avatars was measured at only
  4-8/10 correct on a 10-candidate set, so it is not used.

The name label appears only for a locked pick, is written in the same domain
as the video, and can be matched against the hero vocabulary. It does not
depend on slot geometry, which is why it is the signal used here.

All coordinates are for the canonical 1280x720 frame.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

import cv2
import numpy as np

CANON_W, CANON_H = 1280, 720

# Strip search areas (x0, y0, x1, y1) in canonical px. Measured on the 1280x720
# frame data/audit/m1113_720p_probe.jpg; the bar occupies y 545-695.
STRIPS = {
    "left": (0, 545, 510, 650),
    "right": (770, 545, 1280, 650),
}

try:  # pytesseract is optional at import time so pure-logic tests still run
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


@dataclass
class LabelBox:
    side_strip: str          # "left" | "right"  (screen side, NOT blue/red)
    x: int                   # canonical coordinates
    y: int
    w: int
    h: int


def canonicalize(frame: np.ndarray) -> np.ndarray:
    """Resize a 16:9 frame to 1280x720. Raises on other aspect ratios so a
    letterboxed or cropped VOD is noticed instead of silently mis-measured."""
    h, w = frame.shape[:2]
    if abs(w / h - 16 / 9) > 0.02:
        raise ValueError(f"frame is {w}x{h}, not 16:9; calibrate STRIPS first")
    if (w, h) == (CANON_W, CANON_H):
        return frame
    interp = cv2.INTER_AREA if w > CANON_W else cv2.INTER_CUBIC
    return cv2.resize(frame, (CANON_W, CANON_H), interpolation=interp)


def _white_mask(bgr: np.ndarray, thr: int) -> np.ndarray:
    return ((bgr.min(axis=2) > thr).astype(np.uint8)) * 255


def find_label_boxes(canon: np.ndarray, thr: int = 165) -> list[LabelBox]:
    """Find tall thin clusters of bright glyph blobs (a vertical word)."""
    out: list[LabelBox] = []
    for name, (x0, y0, x1, y1) in STRIPS.items():
        roi = canon[y0:y1, x0:x1]
        m = _white_mask(roi, thr)
        # glue the letters of one word together vertically
        glued = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))
        n, _, st, _ = cv2.connectedComponentsWithStats(glued, 8)
        for k in range(1, n):
            x, y, w, h, area = st[k]
            if 6 <= w <= 20 and 22 <= h <= 85 and area >= 0.18 * w * h:
                out.append(LabelBox(name, x0 + int(x), y0 + int(y), int(w), int(h)))
    return out


def _tight_text(canon: np.ndarray, box: LabelBox, pad: int = 4) -> list[np.ndarray]:
    """Return OCR-ready binary images (black text on white) for several
    thresholds. Rotation is counter-clockwise: the text on screen reads
    top-to-bottom with glyph tops facing right."""
    x0, y0 = max(0, box.x - pad), max(0, box.y - pad)
    crop = canon[y0:box.y + box.h + pad, x0:box.x + box.w + pad]
    crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    k = max(3.0, 46.0 / max(1, box.w))
    crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    mn = crop.min(axis=2)
    outs = []
    for thr in (140, 160, 180, 200):
        b = ((mn > thr) * 255).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
        keep = np.zeros_like(b)
        for i in range(1, n):
            _, _, ww, hh, a = st[i]
            if 20 <= hh <= 110 and a >= 50 and ww <= 110:
                keep[lab == i] = 255
        ys, xs = np.where(keep > 0)
        if len(xs) < 50:
            continue
        ymed = np.median(ys)
        sel = np.abs(ys - ymed) < 45
        if sel.sum() < 50:
            continue
        t = keep[max(0, ys[sel].min() - 5):ys[sel].max() + 6, max(0, xs[sel].min() - 5):xs[sel].max() + 6]
        t = cv2.dilate(t, np.ones((2, 2), np.uint8))
        t = cv2.resize(t, None, fx=60 / t.shape[0], fy=60 / t.shape[0])
        outs.append(255 - cv2.copyMakeBorder(t, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=0))
    return outs


def read_and_rank(canon: np.ndarray, box: LabelBox, vocab: list[str],
                  stop_score: float = 0.95, stop_margin: float = 0.25):
    """OCR one label, trying variants until the reading is clearly one name.

    Early exit matters: every pytesseract call spawns a process (~0.1 s), and a
    full 4-threshold x 2-psm read costs ~1 s per label.
    Returns (raws, ranked) where ranked = [(name, score)] best-first.
    """
    if pytesseract is None:
        raise RuntimeError("pytesseract/tesseract not available")
    cfg = "--psm {} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    raws, ranked = [], [(v, 0.0) for v in vocab]
    for img in _tight_text(canon, box):
        for psm in (7, 8):
            s = pytesseract.image_to_string(img, config=cfg.format(psm)).strip().lower()
            if not s:
                continue
            raws.append(s)
            ranked = rank_vocab(raws, vocab)
            if ranked[0][1] >= stop_score and ranked[0][1] - ranked[1][1] >= stop_margin:
                return raws, ranked
    return raws, ranked


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalpha())


def rank_vocab(raws: list[str], vocab: list[str]) -> list[tuple[str, float]]:
    """Score every vocabulary name by its best similarity to any OCR string.
    Returns [(name, score)] sorted best-first."""
    scored = []
    for name in vocab:
        n = _norm(name)
        best = max((difflib.SequenceMatcher(None, r, n).ratio() for r in raws), default=0.0)
        scored.append((name, best))
    scored.sort(key=lambda t: -t[1])
    return scored


def side_of(box: LabelBox, left_is: str = "blue") -> str:
    other = "red" if left_is == "blue" else "blue"
    return left_is if box.side_strip == "left" else other
