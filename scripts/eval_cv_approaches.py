#!/usr/bin/env python3
"""Reproduce the numbers used to choose the pick-order extraction method.

Writes data/audit/cv_method_comparison.json. Ground truth = the hero names
printed on the broadcast in data/audit/final_frame_samples (portrait_01..10) and
in the two full frames, read by a person. Everything is m1113-g4 only (n=10
heroes), and the label-reading parameters were tuned on these same images, so
treat the numbers as development-set results, not held-out accuracy.
"""
import difflib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pytesseract  # noqa: E402
from vod_extract import detect as D, labels as L  # noqa: E402

TRUTH = ["masha", "marcel", "kimmy", "suyou", "bruno", "claude", "atlas", "hirara", "zhuxin", "freya"]
AUD = ROOT / "data/audit"


def avatar(n):
    a = cv2.imread(str(ROOT / f"data/reference/hero_avatars_cache/{n}.png"), cv2.IMREAD_UNCHANGED)
    if a.shape[2] == 4:
        al = a[:, :, 3:4] / 255.0
        a = (a[:, :, :3] * al).astype(np.uint8)
    return a


def center(p):
    h, w = p.shape[:2]
    return p[int(.1 * h):int(.9 * h), int(.3 * w):int(.95 * w)]


def hist(img):
    h = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(h, h, 1, 0, cv2.NORM_L1)
    return h


def run_avatar_methods():
    P = [cv2.imread(str(AUD / f"final_frame_samples/portrait_{i + 1:02d}.jpg")) for i in range(10)]
    A = [avatar(n) for n in TRUTH]
    sift = cv2.SIFT_create()

    def s_hist(p, a):
        return cv2.compareHist(hist(center(p)), hist(a), cv2.HISTCMP_CORREL)

    def s_sift(p, a):
        g1 = cv2.cvtColor(cv2.resize(center(p), None, fx=3, fy=3), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(a, (300, 300)), cv2.COLOR_BGR2GRAY)
        k1, d1 = sift.detectAndCompute(g1, None)
        k2, d2 = sift.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
            return 0
        m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        return sum(1 for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance)

    def s_ncc(p, a):
        x = cv2.cvtColor(cv2.resize(center(p), (64, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        y = cv2.cvtColor(cv2.resize(a, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        best = -1
        for s in (.6, .8, 1.0):
            t = cv2.resize(y, None, fx=s, fy=s)
            if t.shape[0] > x.shape[0] or t.shape[1] > x.shape[1]:
                t = cv2.resize(t, (min(t.shape[1], x.shape[1]), min(t.shape[0], x.shape[0])))
            best = max(best, cv2.matchTemplate(x, t, cv2.TM_CCOEFF_NORMED).max())
        return best

    res = {}
    for name, f in (("hsv_hist", s_hist), ("sift_matches", s_sift), ("ncc_template", s_ncc)):
        M = np.array([[f(P[i], A[j]) for j in range(10)] for i in range(10)], float)
        hung = 0
        for s in (0, 5):
            r, c = linear_sum_assignment(-M[s:s + 5, s:s + 5])
            hung += int(sum(r == c))
        res[name] = {"top1_of_10_candidates": int(sum(M[i].argmax() == i for i in range(10))),
                     "hungarian_within_side_of_5": hung, "n": 10}
    return res


def run_label_reading_on_frames():
    heroes = json.loads((ROOT / "data/reference/heroes.json").read_text(encoding="utf-8"))["heroes"]
    vocab = [h["name"] for h in heroes]
    games = json.loads((ROOT / "data/processed/mpl_id_s18_games.json").read_text(encoding="utf-8"))["games"]
    g = next(x for x in games if x["game_id"].endswith("m1113-g4"))
    cands = {s: {h["hero"].lower() for h in g["draft"]["picks"][s]} for s in ("blue", "red")}
    truth = {
        "m1113_720p_probe.jpg": {("blue", "masha"), ("blue", "marcel"), ("blue", "kimmy"),
                                 ("red", "atlas"), ("red", "hirara"), ("red", "zhuxin"), ("red", "freya")},
        "m1113_panel_reference.jpg": {("blue", "masha"), ("red", "zhuxin"), ("red", "freya")},
    }
    out = {}
    for f, want in truth.items():
        dets = D.detect_labels(L.canonicalize(cv2.imread(str(AUD / f))), vocab, cands, "blue")
        got = {(d.side, d.hero) for d in dets}
        out[f] = {"visible_labels": len(want), "read_correctly": len(got & want),
                  "false_positives": len(got - want), "vocabulary": len(vocab)}
    return out


def run_label_reading_on_480p_crops():
    """Legacy crop-level test: portrait crops come from a ~480p capture. Closed
    set of the 5 source picks per side. Kept to show resolution sensitivity."""
    games = json.loads((ROOT / "data/processed/mpl_id_s18_games.json").read_text(encoding="utf-8"))["games"]
    g = next(x for x in games if x["game_id"].endswith("m1113-g4"))
    cand = {"blue": [h["hero"].lower() for h in g["draft"]["picks"]["blue"]],
            "red": [h["hero"].lower() for h in g["draft"]["picks"]["red"]]}
    ok, wrong = 0, []
    for i in range(10):
        p = cv2.imread(str(AUD / f"final_frame_samples/portrait_{i + 1:02d}.jpg"))
        h, w = p.shape[:2]
        c = cv2.rotate(p[int(.03 * h):int(.75 * h), :int(.5 * w)], cv2.ROTATE_90_COUNTERCLOCKWISE)
        c = cv2.resize(c, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        mn = c.min(axis=2)
        raws = []
        for thr in (140, 160, 180, 200):
            b = ((mn > thr) * 255).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
            keep = np.zeros_like(b)
            for k in range(1, n):
                if 25 <= st[k][3] <= 120 and st[k][4] >= 60 and st[k][2] <= 110:
                    keep[lab == k] = 255
            ys, xs = np.where(keep > 0)
            if len(xs) < 50:
                continue
            sel = np.abs(ys - np.median(ys)) < 45
            if sel.sum() < 50:
                continue
            t = keep[max(0, ys[sel].min() - 5):ys[sel].max() + 6, max(0, xs[sel].min() - 5):xs[sel].max() + 6]
            t = cv2.dilate(t, np.ones((2, 2), np.uint8))
            t = cv2.resize(t, None, fx=60 / t.shape[0], fy=60 / t.shape[0])
            t = 255 - cv2.copyMakeBorder(t, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=0)
            for psm in (7, 8):
                raws.append(pytesseract.image_to_string(
                    t, config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip().lower())
        side = "blue" if i < 5 else "red"
        sc = {c_: max([difflib.SequenceMatcher(None, r, c_).ratio() for r in raws] + [0]) for c_ in cand[side]}
        best = sorted(sc.items(), key=lambda t: -t[1])
        if best[0][0] == TRUTH[i]:
            ok += 1
        else:
            wrong.append({"truth": TRUTH[i], "predicted": best[0][0], "margin": round(best[0][1] - best[1][1], 2)})
    return {"correct": ok, "n": 10, "closed_set_size": 5, "wrong": wrong}


if __name__ == "__main__":
    result = {
        "note": "m1113-g4 only; development-set numbers (parameters tuned on the same images); not held-out accuracy.",
        "approach_1_avatar_matching_cross_domain": run_avatar_methods(),
        "approach_2_label_reading_real_frames_full_vocab": run_label_reading_on_frames(),
        "approach_2_label_reading_480p_crops_closed_set": run_label_reading_on_480p_crops(),
    }
    out = AUD / "cv_method_comparison.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
