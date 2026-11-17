"""Slot-position-based chronological extraction.

The pick order is NOT inferred from when names appear. It is read from WHERE
they appear: the draft bar fills its slots in turn order, outer edge first.

    A1 B1 B2 A2 A3 B3 B4 A4 A5 B5
    A = blue team, slot n counted from its panel's outer edge (left panel: from the left)
    B = red team, slot n counted from its panel's outer edge (right panel: from the right)

For each slot we decide, over several frames, whether it is LOCKED and which
hero it holds. The identity is frozen at the moment of confirmation, so later
changes of that slot (post-draft swaps between teammates, re-layout) cannot
alter the draft history; they are only reported.

Time is used for exactly one purpose: a consistency check that a later turn
did not lock before an earlier one. That check can only quarantine; it never
orders or reorders anything.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_lib import PICK_TURN_ORDER  # noqa: E402

from . import detect as D, geometry as G  # noqa: E402

TURN_GROUPS = [["A1"], ["B1", "B2"], ["A2", "A3"], ["B3", "B4"], ["A4", "A5"], ["B5"]]
CONFIRM_K = 3            # reads of the same hero needed ...
CONFIRM_WINDOW = 6       # ... within this many consecutive frames
TURN_TOLERANCE_S = 2.0   # label-appearance lag allowed in the consistency check
COLLAPSE_SPAN_S = 5.0


def slot_address(position: str, left_is: str = "blue"):
    """('A1') -> dict(side, strip, slot_from_outer, strip_index_from_left)."""
    side = "blue" if position[0] == "A" else "red"
    n = int(position[1])
    strip = "left" if side == left_is else "right"
    return {"position": position, "side": side, "strip": strip, "slot_from_outer": n,
            "strip_index": n if strip == "left" else 6 - n}


def observe_frame(canon, vocab, cands, left_is, cache):
    """One frame -> (geometry per strip, [(strip, slot_index, Detection|None)], unassigned)."""
    geoms = {s: G.side_geometry(canon, s) for s in ("left", "right")}
    items, unassigned = [], []
    for strip, key, det in D.scan_boxes(canon, vocab, cands, left_is, cache):
        g = geoms[strip]
        if not g.ok:
            unassigned.append((strip, key, "no_geometry"))
            continue
        idx, _off = G.slot_of_label(g, key[0])
        if idx is None:
            unassigned.append((strip, key, "off_slot_edge"))
            continue
        items.append((strip, idx, det))
    return geoms, items, unassigned


class SlotTracker:
    def __init__(self, left_is="blue"):
        self.left_is = left_is
        self.reads = defaultdict(list)      # (strip, idx) -> [(frame, t, hero|None, score, margin)]
        self.times = []
        self.geom_ok = Counter()
        self.unassigned = Counter()

    def add_frame(self, t, geoms, items, unassigned=()):
        i = len(self.times)
        self.times.append(t)
        for s, g in geoms.items():
            if g.ok:
                self.geom_ok[s] += 1
        for strip, idx, det in items:
            self.reads[(strip, idx)].append(
                (i, t, None, 0.0, 0.0) if det is None else (i, t, det.hero, det.score, det.margin))
        for _s, _k, why in unassigned:
            self.unassigned[why] += 1

    def slot(self, position):
        a = slot_address(position, self.left_is)
        return resolve_slot(self.reads.get((a["strip"], a["strip_index"]), []))

    def all_confirmed(self):
        return all(self.slot(p)["status"] == "confirmed" for _, _, p in PICK_TURN_ORDER)


def resolve_slot(reads, k=CONFIRM_K, window=CONFIRM_WINDOW):
    """Decide one slot from its reads over time.

    confirmed: some hero is read >= k times within `window` consecutive frames
    and no other hero is read >= 2 times in that window. The identity is then
    frozen. Otherwise a specific reason: empty / label_unreadable /
    identity_conflict / insufficient_evidence."""
    hero_reads = [r for r in reads if r[2] is not None]
    unread = [r for r in reads if r[2] is None]
    for (i, _t, _h, _s, _m) in hero_reads:
        win = [r for r in hero_reads if i - window < r[0] <= i]
        cnt = Counter(r[2] for r in win)
        top, c = cnt.most_common(1)[0]
        if c >= k and not [h for h, n in cnt.items() if h != top and n >= 2]:
            mine = [r for r in win if r[2] == top]
            later = Counter(r[2] for r in hero_reads if r[0] > i and r[2] != top)
            changed = sorted(h for h, n in later.items() if n >= k)
            return {"status": "confirmed", "hero": top, "n_reads": len(hero_reads),
                    "first_read_t": mine[0][1], "confirm_frame": i,
                    "mean_score": round(sum(r[3] for r in mine) / len(mine), 3),
                    "min_margin": round(min(r[4] for r in mine), 3),
                    "changed_after_lock": changed}
    heroes = Counter(r[2] for r in hero_reads)
    if not hero_reads:
        status = "label_unreadable" if len(unread) >= k else "empty"
    elif len([h for h, n in heroes.items() if n >= 2]) > 1 or (len(heroes) > 1 and len(unread) >= k):
        status = "identity_conflict"
    else:
        status = "insufficient_evidence"
    return {"status": status, "hero": None, "n_reads": len(hero_reads), "n_unreadable": len(unread),
            "heroes_seen": dict(heroes)}


def build_result(tracker: SlotTracker, cands, lane_order=None, turn_check=True):
    """Turn slot states into the ordered pick list plus checks/issues."""
    issues = []
    left_is = tracker.left_is
    n_frames = len(tracker.times)
    layout_ok = True
    for strip in ("left", "right"):
        ok = tracker.geom_ok[strip]
        if n_frames == 0 or ok == 0:
            layout_ok = False
            issues.append({"code": "layout_not_validated", "strip": strip,
                           "detail": "no frame had a valid slot geometry (5 name plates + plausible fit); "
                                     "the layout may differ from the one this was measured on"})
        elif ok / n_frames < 0.5:
            layout_ok = False
            issues.append({"code": "geometry_unstable", "strip": strip,
                           "detail": f"valid geometry in only {ok}/{n_frames} frames"})

    slots = {}
    for _side, _phase, pos in PICK_TURN_ORDER:
        r = tracker.slot(pos)
        r["address"] = slot_address(pos, left_is)
        slots[pos] = r
        if r["status"] != "confirmed":
            issues.append({"code": f"slot_{r['status']}", "position": pos,
                           "strip": r["address"]["strip"], "strip_index": r["address"]["strip_index"],
                           "detail": {k: v for k, v in r.items() if k in ("heroes_seen", "n_reads", "n_unreadable")}})
        elif r["changed_after_lock"]:
            issues.append({"code": "slot_changed_after_lock", "position": pos, "level": "info",
                           "kept": r["hero"], "later_read": r["changed_after_lock"],
                           "detail": "identity frozen at lock; later change ignored (post-draft swap or re-layout)"})

    confirmed = all(s["status"] == "confirmed" for s in slots.values())
    checks = {"layout_validated": layout_ok, "all_slots_confirmed": confirmed}
    heroes_distinct = matches_source = confident = not_collapsed = not_lane = turn_ok = False
    order = []
    if confirmed:
        order = [(pos, slots[pos]["hero"]) for _s, _p, pos in PICK_TURN_ORDER]
        hs = [h for _p, h in order]
        heroes_distinct = len(set(hs)) == 10
        if not heroes_distinct:
            for h, n in Counter(hs).items():
                if n > 1:
                    issues.append({"code": "duplicate_hero", "hero": h,
                                   "positions": [p for p, x in order if x == h]})
        matches_source = True
        for pos, h in order:
            side = slot_address(pos, left_is)["side"]
            other = "red" if side == "blue" else "blue"
            if h in cands[side]:
                continue
            matches_source = False
            issues.append({"code": "side_mismatch" if h in cands[other] else "hero_not_in_source",
                           "position": pos, "hero": h,
                           "detail": "differs from the source record; extraction is kept as read, not corrected"})
        for side in ("blue", "red"):
            got = {h for pos, h in order if slot_address(pos, left_is)["side"] == side}
            for h in sorted(cands[side] - got):
                matches_source = False
                issues.append({"code": "source_hero_not_extracted", "hero": h, "side": side})
        low = [p for p, h in order if slots[p]["min_margin"] < 0.2 or slots[p]["mean_score"] < 0.8]
        confident = not low
        for p in low:
            issues.append({"code": "low_ocr_confidence", "position": p, "hero": slots[p]["hero"]})

        times = {p: slots[p]["first_read_t"] for p, _ in order}
        span = max(times.values()) - min(times.values())
        not_collapsed = span > COLLAPSE_SPAN_S
        if not not_collapsed:
            issues.append({"code": "labels_appear_simultaneously", "span_s": round(span, 2),
                           "detail": "all ten slots were already locked when the window began or filled at once "
                                     "(final lineup, or wrong window); draft chronology not observable"})
        if lane_order:
            per = {s: [h for pos, h in order if slot_address(pos, left_is)["side"] == s] for s in ("blue", "red")}
            if all(per[s] == [x.lower() for x in lane_order.get(s, [])] for s in ("blue", "red")):
                issues.append({"code": "suspicious_lane_order",
                               "detail": "extracted order equals the source lane-slot order on both sides "
                                         "(chance ~1 in 14,400); the final lane lineup may have been read"})
            else:
                not_lane = True
            not_lane = not_lane or not lane_order
        else:
            not_lane = True
        if turn_check:
            turn_ok = True
            for a in range(len(TURN_GROUPS)):
                for b in range(a + 1, len(TURN_GROUPS)):
                    earlier = max(times[p] for p in TURN_GROUPS[a])
                    later = min(times[p] for p in TURN_GROUPS[b])
                    if later < earlier - TURN_TOLERANCE_S:
                        turn_ok = False
                        issues.append({"code": "slot_order_inconsistent_with_turn_structure",
                                       "earlier_turn": TURN_GROUPS[a], "later_turn": TURN_GROUPS[b],
                                       "detail": f"later turn locked {earlier - later:.1f}s before the earlier turn "
                                                 "finished; slot order may not be pick order on this layout"})
        else:
            turn_ok = True
    checks.update({"heroes_distinct": heroes_distinct, "matches_source_heroes": matches_source,
                   "ocr_confident": confident, "chronology_not_collapsed": not_collapsed,
                   "not_lane_order_artifact": not_lane, "turn_structure_consistent": turn_ok})
    passed = all(checks.values())
    return {"slots": slots, "order": order, "checks": checks, "issues": issues, "passed": passed}
