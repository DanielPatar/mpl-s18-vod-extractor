"""Tests for the slot-position extractor (scripts/vod_extract, scripts/extract_picks.py).

Kinds of test (see docs/VOD_EXTRACTOR.md):

  RealFrame*   frames captured from the real m1113-g4 VOD (data/audit/); ground truth
               is what is visible on screen, read by a person.
  Synthetic*   video rendered here (flexible slot widths, name plates, labels at slot
               left edges). Tests plumbing, slot mapping and state handling ONLY; says
               nothing about accuracy on real video.
  Stream*      ffmpeg pipe layer (local range-capable HTTP server, fake ffmpeg).
  Slot*/Logic* hand-built reads, no images.

No test runs on a full real VOD: this sandbox cannot reach YouTube.
"""
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

EXTRACTOR_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXTRACTOR_ROOT.parent / "mpl-s18-draft-project"
sys.path.insert(0, str(EXTRACTOR_ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from vod_extract import detect as D, geometry as G, labels as L, slots as SL, stream as S  # noqa: E402
import extract_picks as X  # noqa: E402

GAME = "mpl-id-s18-m1113-g4"
AUDIT = ROOT / "data/audit"
POSITIONS = ["A1", "B1", "B2", "A2", "A3", "B3", "B4", "A4", "A5", "B5"]
# What the draft bar holds in each slot position for m1113-g4 (final frame, outer edge first)
SLOT_HEROES = {"A1": "masha", "B1": "freya", "B2": "zhuxin", "A2": "marcel", "A3": "kimmy",
               "B3": "hirara", "B4": "atlas", "A4": "suyou", "A5": "bruno", "B5": "claude"}
EXPECTED_SEQUENCE = ["masha", "freya", "zhuxin", "marcel", "kimmy", "hirara", "atlas", "suyou", "bruno", "claude"]
LOCK_T = {"A1": 8, "B1": 12, "B2": 13, "A2": 18, "A3": 19, "B3": 24, "B4": 25, "A4": 30, "A5": 31, "B5": 36}


def _vocab_and_cands(game_id=GAME):
    g, heroes = X.load_game(game_id)
    return [h["name"] for h in heroes], {s: {h["hero"].lower() for h in g["draft"]["picks"][s]}
                                         for s in ("blue", "red")}


def _tesseract_ok():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


needs_ocr = unittest.skipUnless(_tesseract_ok(), "tesseract not installed")
needs_ffmpeg = unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe missing")


def _real_canon(name):
    im = cv2.imread(str(AUDIT / name))
    return im if im.shape[1] == 1280 else cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA)


def _tracker_from_frame(canon, left_is="blue", frames=6):
    vocab, cands = _vocab_and_cands()
    geoms, items, un = SL.observe_frame(canon, vocab, cands, left_is, D.LabelCache())
    tr = SL.SlotTracker(left_is)
    for i in range(frames):
        tr.add_frame(i * 0.5, geoms, items, un)
    return tr


# --------------------------------------------------------------------------
# Real frames
# --------------------------------------------------------------------------
@needs_ocr
class RealFrameGeometryTests(unittest.TestCase):
    # Reference boundaries measured INDEPENDENTLY from image gradients / by eye (see docs)
    REF = {("m1113_720p_probe.jpg", "left"): [80, 152, 222, 360],
           ("m1113_panel_reference.jpg", "left"): [85, 228, 343, 422],
           ("m1113_720p_probe.jpg", "right"): [878, 976, 1075, 1173],
           ("m1113_panel_reference.jpg", "right"): [878, 976, 1075, 1173]}

    def test_fitted_boundaries_match_independent_measurements(self):
        for (name, strip), ref in self.REF.items():
            g = G.side_geometry(_real_canon(name), strip)
            self.assertTrue(g.ok, (name, strip, g.reason))
            for got, want in zip(g.bounds[1:5], ref):
                self.assertLessEqual(abs(got - want), 4.5, (name, strip, g.bounds, ref))

    def test_slot_widths_really_differ_between_states(self):
        w_active = np.diff(G.side_geometry(_real_canon("m1113_720p_probe.jpg"), "left").bounds)
        w_idle = np.diff(G.side_geometry(_real_canon("m1113_720p_probe.jpg"), "right").bounds)
        self.assertGreater(w_active.max() / w_active.min(), 1.7)     # 72 vs 138 px
        self.assertLess(w_idle.max() / w_idle.min(), 1.15)           # ~98 px each

    def test_label_off_slot_edge_is_not_assigned(self):
        g = G.side_geometry(_real_canon("m1113_720p_probe.jpg"), "left")
        self.assertEqual(G.slot_of_label(g, 12)[0], 1)
        self.assertIsNone(G.slot_of_label(g, 300)[0])                 # mid-slot: not a label position

    def test_missing_name_plate_invalidates_geometry(self):
        im = _real_canon("m1113_720p_probe.jpg").copy()
        im[690:712, 100:140] = (40, 90, 50)                           # wipe the 2nd player-name plate
        g = G.side_geometry(im, "left")
        self.assertFalse(g.ok)
        self.assertIn("expected 5 name plates", g.reason)


@needs_ocr
class RealFrameSlotTests(unittest.TestCase):
    def _statuses(self, name):
        tr = _tracker_from_frame(_real_canon(name))
        return {p: (tr.slot(p)["hero"] or tr.slot(p)["status"]) for p in POSITIONS}

    def test_0530_frame_maps_each_hero_to_the_right_pick_position(self):
        # On screen: blue MASHA MARCEL KIMMY | slot4 player photo | slot5 wide Bruno preview;
        # red (right panel) logo | ATLAS HIRARA ZHUXIN FREYA.  Outer-edge-first:
        self.assertEqual(self._statuses("m1113_720p_probe.jpg"),
                         {"A1": "masha", "B1": "freya", "B2": "zhuxin", "A2": "marcel", "A3": "kimmy",
                          "B3": "hirara", "B4": "atlas", "A4": "empty", "A5": "empty", "B5": "empty"})

    def test_mid_draft_frame_ignores_previews(self):
        # Only MASHA, ZHUXIN, FREYA are locked; Suyou/Kimmy are wide previews without a label.
        st = self._statuses("m1113_panel_reference.jpg")
        self.assertEqual({p: h for p, h in st.items() if h != "empty"},
                         {"A1": "masha", "B1": "freya", "B2": "zhuxin"})
        self.assertEqual(st["A2"], "empty")
        self.assertEqual(st["A3"], "empty")


@needs_ocr
class RealFramePlumbingTest(unittest.TestCase):
    """CLI on two REAL frames; timestamps assigned by this test (real capture times
    were not recorded), so only the slot mapping and honest partial result count."""

    def test_cli_quarantines_partial_draft_with_specific_reasons(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fd = tmp / "f"
            fd.mkdir()
            for t in (10.0, 10.5, 11.0):
                shutil.copy(AUDIT / "m1113_panel_reference.jpg", fd / f"{t}.jpg")
            for t in (12.0, 12.5, 13.0):
                shutil.copy(AUDIT / "m1113_720p_probe.jpg", fd / f"{t}.jpg")
            out = tmp / "o.json"
            rc = X.main(["--game-id", GAME, "--frames-dir", str(fd), "--start", "10", "--end", "13", "--out", str(out)])
            res = json.loads(out.read_text())
            self.assertEqual(rc, 2)
            self.assertEqual(res["result"], "quarantined")
            self.assertIsNone(res["events"])
            got = {s["position"]: s["hero"] for s in res["diagnostics"]["slots"] if s["hero"]}
            self.assertEqual(got, {"A1": "masha", "B1": "freya", "B2": "zhuxin", "A2": "marcel",
                                   "A3": "kimmy", "B3": "hirara", "B4": "atlas"})
            empty = {i["position"] for i in res["issues"] if i["code"] == "slot_empty"}
            self.assertEqual(empty, {"A4", "A5", "B5"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Synthetic video: flexible slot widths, name plates, labels at slot left edges
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _font(size=12):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return None


@functools.lru_cache(maxsize=None)
def _label_patch(text):
    from PIL import Image, ImageDraw
    im = Image.new("L", (90, 14), 0)
    ImageDraw.Draw(im).text((1, 0), text.upper(), fill=255, font=_font(12))
    return np.array(im.crop(im.getbbox()).rotate(-90, expand=True)) > 128


@functools.lru_cache(maxsize=None)
def _name_patch(text):
    from PIL import Image, ImageDraw
    im = Image.new("L", (90, 14), 0)
    ImageDraw.Draw(im).text((1, 0), text, fill=255, font=_font(11))
    return np.array(im.crop(im.getbbox())) > 128


@functools.lru_cache(maxsize=1)
def _bg():
    rng = np.random.default_rng(0)
    return rng.integers(20, 70, (720, 1280, 1)).astype(np.uint8).repeat(3, axis=2)


NAMES = {"left": ["Karss", "Joshuaa", "Jiizee", "Aprho", "Zeonn"], "right": ["Keven", "Lyoni", "Drichel", "Kevin", "Aran"]}
KIND_COLOR = {"placeholder": (35, 45, 40), "preview": (205, 210, 220), "locked": (110, 120, 135)}


class Plan:
    """Slot states over time: (strip, idx) -> [(t_from, kind, hero)]."""

    def __init__(self, left_is="blue"):
        self.left_is = left_is
        self.ev = defaultdict(list)

    def set(self, pos, t_from, kind, hero=None):
        a = SL.slot_address(pos, self.left_is)
        self.ev[(a["strip"], a["strip_index"])].append((t_from, kind, hero))
        self.ev[(a["strip"], a["strip_index"])].sort(key=lambda e: e[0])

    def state(self, strip, idx, t):
        cur = ("placeholder", None)
        for t_from, kind, hero in self.ev.get((strip, idx), []):
            if t >= t_from:
                cur = (kind, hero)
        return cur


def std_plan(left_is="blue", lock_t=None, heroes=None, preview_s=3.0):
    lock_t, heroes = dict(lock_t or LOCK_T), dict(heroes or SLOT_HEROES)
    p = Plan(left_is)
    for pos in POSITIONS:
        p.set(pos, lock_t[pos] - preview_s, "preview")
        p.set(pos, lock_t[pos], "locked", heroes[pos])
    return p


def _bounds(strip, kinds):
    b0, b5 = G.NOMINAL_OUTER[strip]
    w = [1.92 if k == "preview" else 1.0 for k in kinds]       # active (preview) slots are wide
    xs, acc = [b0], 0.0
    for wi in w:
        acc += wi
        xs.append(b0 + (b5 - b0) * acc / sum(w))
    return xs


def _frame(t, plan, hide_names=(), hide_window=None):
    f = _bg().copy()
    for strip in ("left", "right"):
        states = [plan.state(strip, k, t) for k in range(1, 6)]
        b = _bounds(strip, [s[0] for s in states])
        for k in range(5):
            x0, x1 = int(round(b[k])), int(round(b[k + 1]))
            kind, hero = states[k]
            cv2.rectangle(f, (x0 + 1, 545), (x1 - 1, 693), KIND_COLOR[kind], -1)
            if hero:
                p = _label_patch(hero)
                roi = f[558:558 + p.shape[0], x0 + 4:x0 + 4 + p.shape[1]]
                roi[p] = (245, 245, 245)
        f[694:711, int(b[0]):int(b[5])] = (40, 105, 60)           # player-name band
        hidden = strip in hide_names or (hide_window and hide_window[0] <= t <= hide_window[1])
        if not hidden:
            for k in range(5):
                p = _name_patch(NAMES[strip][k])
                cx = (b[k] + b[k + 1]) / 2
                x = int(round(cx - p.shape[1] / 2))
                roi = f[698:698 + p.shape[0], x:x + p.shape[1]]
                roi[p] = (250, 250, 250)
    return f


def _write_video(path, plan, dur=48.0, fps=4, **kw):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))
    for i in range(int(dur * fps)):
        vw.write(_frame(i / fps, plan, **kw))
    vw.release()


def _events_heroes(res):
    return [e["hero"].lower() for e in res["events"]]


def _issue_codes(res):
    return {i["code"] for i in res["issues"]}


@needs_ocr
@needs_ffmpeg
class SyntheticSlotVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _font() is None:
            raise unittest.SkipTest("no bold TrueType font available")
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, name, plan, extra=(), dur=48.0, video_kw=None, base=("--start", "0", "--end", "47")):
        v = self.tmp / f"{name}.avi"
        _write_video(v, plan, dur=dur, **(video_kw or {}))
        out = self.tmp / f"{name}.json"
        rc = X.main(["--game-id", GAME, "--video", str(v), "--out", str(out), *base, *extra])
        return rc, json.loads(out.read_text())

    # ---- mapping slot -> pick position --------------------------------------------
    def test_order_comes_from_slot_position(self):
        rc, res = self._run("std", std_plan())
        self.assertEqual(rc, 0, res["issues"])
        self.assertEqual(_events_heroes(res), EXPECTED_SEQUENCE)
        self.assertEqual([e["position"] for e in res["events"]], POSITIONS)
        self.assertEqual(res["events"][9]["hero_id"], 65)
        self.assertFalse(res["provenance"]["order_uses_first_seen_time"])
        self.assertTrue(res["provenance"]["order_uses_slot_position"])
        self.assertNotIn("first_seen_s", json.dumps(res))

    def test_lock_time_inside_a_turn_does_not_change_the_order(self):
        t = dict(LOCK_T, B2=12, B1=13, A3=18, A2=19, B4=24, B3=25, A5=30, A4=31)   # second slot of each pair locks first
        rc, res = self._run("pairs", std_plan(lock_t=t))
        self.assertEqual(rc, 0, res["issues"])
        self.assertEqual(_events_heroes(res), EXPECTED_SEQUENCE)

    def test_stops_early_and_writes_only_the_json(self):
        work = Path(tempfile.mkdtemp(dir=self.tmp))
        v = self.tmp / "long.avi"
        _write_video(v, std_plan(), dur=80)
        out = work / "o.json"
        rc = X.main(["--game-id", GAME, "--video", str(v), "--out", str(out)])
        res = json.loads(out.read_text())
        self.assertEqual(rc, 0)
        self.assertTrue(res["diagnostics"]["stopped_early"])
        self.assertLess(res["diagnostics"]["frames_scanned"], 80 * 2)
        self.assertEqual([p.name for p in work.iterdir()], ["o.json"])
        self.assertEqual(res["storage"]["video_or_frame_files_written"], 0)

    def test_red_panel_on_the_left_is_mirrored_correctly(self):
        rc, res = self._run("redleft", std_plan(left_is="red"), ["--left-is", "red"])
        self.assertEqual(rc, 0, res["issues"])
        self.assertEqual(_events_heroes(res), EXPECTED_SEQUENCE)

    def test_wrong_left_is_is_reported_not_guessed(self):
        rc, res = self._run("wrongside", std_plan(left_is="red"))                  # read as blue-on-left
        self.assertEqual(rc, 2)
        self.assertIn("side_mismatch", _issue_codes(res))

    # ---- empty / preview / unreadable / ambiguous -------------------------------
    def test_empty_slot_is_quarantined_with_its_position(self):
        p = std_plan()
        p.ev[("left", 4)] = [e for e in p.ev[("left", 4)] if e[1] != "locked"]       # A4 never locks (stays preview)
        rc, res = self._run("empty", p)
        self.assertEqual(rc, 2)
        self.assertIsNone(res["events"])
        self.assertIn(("slot_empty", "A4"), {(i["code"], i.get("position")) for i in res["issues"]})
        self.assertIn("all_slots_confirmed", res["quarantine_reasons"])

    def test_preview_art_is_never_read_as_a_pick(self):
        p = std_plan()
        p.ev[("right", 1)] = [(30.0, "preview", None)]                                # B5: long wide preview, never locked
        rc, res = self._run("preview", p)
        slots = {s["position"]: s for s in res["diagnostics"]["slots"]}
        self.assertEqual(slots["B5"]["status"], "empty")
        self.assertEqual(slots["B5"]["n_reads"], 0)
        self.assertEqual(rc, 2)

    def test_unreadable_label_is_distinguished_from_empty(self):
        p = std_plan()
        p.ev[("left", 3)] = [(16.0, "preview", None), (19.0, "locked", "xzxzx")]    # A3 shows a label we cannot read
        rc, res = self._run("unread", p)
        self.assertEqual(rc, 2)
        self.assertIn(("slot_label_unreadable", "A3"), {(i["code"], i.get("position")) for i in res["issues"]})

    def test_slot_alternating_between_two_heroes_is_ambiguous(self):
        p = std_plan()
        flip = [(24.0 + 0.5 * k, "locked", "hirara" if k % 2 == 0 else "atlas") for k in range(46)]
        p.ev[("right", 3)] = [(21.0, "preview", None)] + flip                         # B3
        rc, res = self._run("flip", p)
        self.assertEqual(rc, 2)
        self.assertIn(("slot_identity_conflict", "B3"), {(i["code"], i.get("position")) for i in res["issues"]})

    # ---- changes during animation / after lock / swaps --------------------------
    def test_hero_changing_during_the_lock_animation_settles_on_the_stable_hero(self):
        p = std_plan()
        p.ev[("left", 2)] = [(15.0, "preview", None), (18.0, "locked", "kimmy"),       # A2: 1 s of a wrong name ...
                             (19.0, "locked", "marcel")]                               # ... then the real one
        rc, res = self._run("anim", p)
        self.assertEqual(rc, 0, res["issues"])
        self.assertEqual(_events_heroes(res), EXPECTED_SEQUENCE)

    def test_identity_is_frozen_at_lock_when_the_slot_changes_later(self):
        p = std_plan()
        p.ev[("left", 1)].append((40.0, "locked", "bruno"))                            # A1 later shows another hero
        rc, res = self._run("frozen", p, ["--lookahead", "100"])
        self.assertEqual(_events_heroes(res)[0], "masha")
        info = [i for i in res["issues"] if i["code"] == "slot_changed_after_lock"]
        self.assertTrue(info and info[0]["position"] == "A1" and info[0]["kept"] == "masha")
        self.assertEqual(rc, 0, res["issues"])

    def test_post_draft_swaps_between_teammates_do_not_change_the_result(self):
        p = std_plan()
        for a, b in (("A2", "A3"), ("A4", "A5")):
            ha, hb = SLOT_HEROES[a], SLOT_HEROES[b]
            p.set(a, 42.0, "locked", hb)
            p.set(b, 42.0, "locked", ha)
        rc, res = self._run("swap", p, ["--lookahead", "100"])
        self.assertEqual(rc, 0, res["issues"])
        self.assertEqual(_events_heroes(res), EXPECTED_SEQUENCE)
        self.assertEqual(sum(1 for i in res["issues"] if i["code"] == "slot_changed_after_lock"), 4)

    # ---- geometry --------------------------------------------------------------
    def test_geometry_follows_changing_slot_widths(self):
        f = _frame(10.5, std_plan())                       # A1 locked, B1/B2 previews: uneven widths
        for strip in ("left", "right"):
            states = [std_plan().state(strip, k, 10.5)[0] for k in range(1, 6)]
            want = _bounds(strip, states)
            g = G.side_geometry(f, strip)
            self.assertTrue(g.ok, g.reason)
            self.assertLessEqual(max(abs(a - b) for a, b in zip(g.bounds, want)), 3.0)

    def test_short_loss_of_geometry_is_tolerated(self):
        rc, res = self._run("hide_short", std_plan(), video_kw={"hide_window": (10.0, 15.0)})
        self.assertEqual(rc, 0, res["issues"])

    def test_unvalidated_layout_is_quarantined(self):
        rc, res = self._run("hide_right", std_plan(), video_kw={"hide_names": ("right",)})
        self.assertEqual(rc, 2)
        self.assertIn("layout_not_validated", _issue_codes(res))

    # ---- guards that must quarantine, never "fix" ------------------------------
    def test_final_lineup_shown_all_at_once_is_quarantined(self):
        rc, res = self._run("final", std_plan(lock_t={p: 30 for p in POSITIONS}, preview_s=0.0))
        self.assertEqual(rc, 2)
        self.assertIn("labels_appear_simultaneously", _issue_codes(res))

    def test_slots_that_lock_out_of_turn_are_quarantined_and_time_never_orders(self):
        rc, res = self._run("turn", std_plan(lock_t=dict(LOCK_T, B5=4)))
        self.assertEqual(rc, 2)
        self.assertIn("slot_order_inconsistent_with_turn_structure", _issue_codes(res))
        rc2, res2 = self._run("turn_off", std_plan(lock_t=dict(LOCK_T, B5=4)), ["--no-turn-check"])
        self.assertEqual(rc2, 0)
        self.assertEqual(_events_heroes(res2), EXPECTED_SEQUENCE)       # same order: time never decides it

    def test_extraction_equal_to_the_source_lane_order_is_quarantined(self):
        lane = {"A1": "masha", "A2": "suyou", "A3": "kimmy", "A4": "bruno", "A5": "marcel",
                "B1": "freya", "B2": "hirara", "B3": "zhuxin", "B4": "claude", "B5": "atlas"}
        rc, res = self._run("lane", std_plan(heroes=lane))
        self.assertEqual(rc, 2)
        self.assertIn("suspicious_lane_order", _issue_codes(res))

    def test_series_or_livestream_vod_without_start_is_refused(self):
        v = self.tmp / "tiny.avi"
        _write_video(v, std_plan(), dur=2)
        for gid in ("mpl-id-s18-m1087-g1", "mpl-id-s18-m1093-g1"):
            with self.assertRaises(SystemExit, msg=gid):
                X.main(["--game-id", gid, "--video", str(v), "--out", str(self.tmp / "z.json")])

    def test_probe_prints_geometry_and_slot_assignment(self):
        v = self.tmp / "probe.avi"
        _write_video(v, std_plan(), dur=20)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = X.main(["--game-id", GAME, "--video", str(v), "--probe-at", "15"])
        txt = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("left: geometry OK", txt)
        self.assertIn("A1: left panel slot 1 from left -> masha", txt)
        self.assertIn("B1: right panel slot 5 from left -> freya", txt)


# --------------------------------------------------------------------------
# Slot logic (no images)
# --------------------------------------------------------------------------
def _det(hero, side="blue", score=1.0, margin=0.5):
    return D.Detection(side, hero, score, margin, True, (0, 0, 0, 0))


OK = SimpleNamespace(ok=True)
CANDS = {"blue": {"masha", "suyou", "kimmy", "bruno", "marcel"}, "red": {"freya", "hirara", "zhuxin", "claude", "atlas"}}
LANE = {"blue": ["masha", "suyou", "kimmy", "bruno", "marcel"], "red": ["freya", "hirara", "zhuxin", "claude", "atlas"]}


def _tracker(lock_frame, heroes=SLOT_HEROES, left_is="blue", n=80, overrides=None):
    """Frame i (0.5 s each): slot `pos` shows its hero from lock_frame[pos] on."""
    tr = SL.SlotTracker(left_is)
    for i in range(n):
        items = []
        for pos in POSITIONS:
            a = SL.slot_address(pos, left_is)
            h = (overrides or {}).get((pos, i), heroes[pos] if i >= lock_frame[pos] else "-")
            if h == "-":
                continue
            items.append((a["strip"], a["strip_index"], None if h is None else _det(h, a["side"])))
        tr.add_frame(i * 0.5, {"left": OK, "right": OK}, items)
    return tr


FR = {p: int(t * 2) for p, t in LOCK_T.items()}


class SlotMappingTests(unittest.TestCase):
    def test_every_pick_position_maps_to_the_specified_slot(self):
        # A: blue slot n from the left. B: red slot n from the right (= strip index 6-n).
        want = {"A1": ("left", 1), "B1": ("right", 5), "B2": ("right", 4), "A2": ("left", 2), "A3": ("left", 3),
                "B3": ("right", 3), "B4": ("right", 2), "A4": ("left", 4), "A5": ("left", 5), "B5": ("right", 1)}
        for pos, (strip, idx) in want.items():
            a = SL.slot_address(pos, "blue")
            self.assertEqual((a["strip"], a["strip_index"]), (strip, idx), pos)

    def test_mirrored_when_the_red_panel_is_on_the_left(self):
        a = SL.slot_address("B1", "red")
        self.assertEqual((a["strip"], a["strip_index"]), ("left", 1))
        a = SL.slot_address("A5", "red")
        self.assertEqual((a["strip"], a["strip_index"]), ("right", 1))

    def test_turn_order_constant_is_the_required_sequence(self):
        self.assertEqual([p for _s, _ph, p in SL.PICK_TURN_ORDER], POSITIONS)

    def test_full_draft_gives_the_expected_sequence(self):
        res = SL.build_result(_tracker(FR), CANDS, LANE)
        self.assertTrue(res["passed"], res["issues"])
        self.assertEqual([h for _p, h in res["order"]], EXPECTED_SEQUENCE)


class ResolveSlotTests(unittest.TestCase):
    def R(self, seq):                       # seq of hero|None per frame ('-' = nothing)
        reads = [(i, i * 0.5, h, 1.0, 0.5) if h is not None else (i, i * 0.5, None, 0.0, 0.0)
                 for i, h in enumerate(seq) if h != "-"]
        return SL.resolve_slot(reads)

    def test_confirmed(self):
        r = self.R(["-", "masha", "masha", "masha"])
        self.assertEqual((r["status"], r["hero"]), ("confirmed", "masha"))

    def test_empty(self):
        self.assertEqual(self.R(["-", "-", "-"])["status"], "empty")

    def test_few_unreadable_boxes_are_still_empty_many_are_unreadable(self):
        self.assertEqual(self.R([None, "-", None, "-"])["status"], "empty")
        self.assertEqual(self.R([None, None, None, None])["status"], "label_unreadable")

    def test_two_brief_wrong_reads_then_a_stable_hero(self):
        r = self.R(["kimmy", "kimmy", "marcel", "marcel", "marcel", "marcel", "marcel", "marcel"])
        self.assertEqual((r["status"], r["hero"]), ("confirmed", "marcel"))

    def test_interleaved_heroes_are_ambiguous(self):
        r = self.R(["hirara", "atlas"] * 6)
        self.assertEqual(r["status"], "identity_conflict")

    def test_too_few_reads_is_insufficient(self):
        self.assertEqual(self.R(["masha", "masha"])["status"], "insufficient_evidence")

    def test_identity_frozen_and_later_change_reported(self):
        r = self.R(["masha"] * 4 + ["bruno"] * 5)
        self.assertEqual((r["status"], r["hero"]), ("confirmed", "masha"))
        self.assertEqual(r["changed_after_lock"], ["bruno"])

    def test_one_stray_read_does_not_block(self):
        r = self.R(["masha", "masha", "bruno", "masha", "masha"])
        self.assertEqual((r["status"], r["hero"]), ("confirmed", "masha"))


class BuildResultTests(unittest.TestCase):
    def codes(self, res):
        return {i["code"] for i in res["issues"]}

    def test_empty_slot_quarantines_and_names_the_position(self):
        fr = dict(FR, A4=10 ** 6)
        res = SL.build_result(_tracker(fr), CANDS, LANE)
        self.assertFalse(res["passed"])
        self.assertIn(("slot_empty", "A4"), {(i["code"], i.get("position")) for i in res["issues"]})
        self.assertEqual(res["order"], [])                       # nothing is filled in

    def test_ambiguous_slot_quarantines(self):
        ov = {("B3", i): ("hirara" if i % 2 else "atlas") for i in range(FR["B3"], 80)}
        res = SL.build_result(_tracker(FR, overrides=ov), CANDS, LANE)
        self.assertIn("slot_identity_conflict", self.codes(res))
        self.assertFalse(res["passed"])

    def test_hero_change_during_animation_then_stable(self):
        ov = {("A2", FR["A2"]): "kimmy", ("A2", FR["A2"] + 1): "kimmy"}
        res = SL.build_result(_tracker(FR, overrides=ov), CANDS, LANE)
        self.assertTrue(res["passed"], res["issues"])
        self.assertEqual(dict(res["order"])["A2"], "marcel")

    def test_swap_after_the_draft_is_ignored(self):
        ov = {}
        for i in range(70, 80):
            ov[("A2", i)] = "kimmy"
            ov[("A3", i)] = "marcel"
        res = SL.build_result(_tracker(FR, overrides=ov), CANDS, LANE)
        self.assertTrue(res["passed"], res["issues"])
        self.assertEqual([h for _p, h in res["order"]], EXPECTED_SEQUENCE)

    def test_duplicate_hero_quarantines(self):
        h = dict(SLOT_HEROES, A5="masha")
        res = SL.build_result(_tracker(FR, heroes=h), CANDS, LANE)
        self.assertIn("duplicate_hero", self.codes(res))
        self.assertFalse(res["passed"])

    def test_source_mismatch_is_reported_not_corrected(self):
        h = dict(SLOT_HEROES, B5="nolan")
        res = SL.build_result(_tracker(FR, heroes=h), CANDS, LANE)
        self.assertIn("hero_not_in_source", self.codes(res))
        self.assertIn("source_hero_not_extracted", self.codes(res))
        self.assertEqual(dict(res["order"])["B5"], "nolan")     # kept as read

    def test_out_of_turn_locks_quarantine(self):
        fr = dict(FR, B5=2)
        res = SL.build_result(_tracker(fr), CANDS, LANE)
        self.assertIn("slot_order_inconsistent_with_turn_structure", self.codes(res))
        self.assertFalse(res["passed"])
        self.assertTrue(SL.build_result(_tracker(fr), CANDS, LANE, turn_check=False)["passed"])

    def test_lock_order_inside_a_turn_is_free(self):
        fr = dict(FR, B1=FR["B2"] + 1, A3=FR["A2"] - 1, B4=FR["B3"] - 1, A5=FR["A4"] - 1)
        res = SL.build_result(_tracker(fr), CANDS, LANE)
        self.assertTrue(res["passed"], res["issues"])
        self.assertEqual([h for _p, h in res["order"]], EXPECTED_SEQUENCE)

    def test_all_at_once_is_collapsed(self):
        res = SL.build_result(_tracker({p: 4 for p in POSITIONS}), CANDS, LANE)
        self.assertIn("labels_appear_simultaneously", self.codes(res))

    def test_lane_order_tripwire(self):
        lane = {"A1": "masha", "A2": "suyou", "A3": "kimmy", "A4": "bruno", "A5": "marcel",
                "B1": "freya", "B2": "hirara", "B3": "zhuxin", "B4": "claude", "B5": "atlas"}
        res = SL.build_result(_tracker(FR, heroes=lane), CANDS, LANE)
        self.assertIn("suspicious_lane_order", self.codes(res))
        self.assertFalse(res["passed"])

    def test_no_valid_geometry_anywhere_is_a_layout_failure(self):
        tr = SL.SlotTracker("blue")
        bad = SimpleNamespace(ok=False)
        for i in range(10):
            tr.add_frame(i * 0.5, {"left": bad, "right": bad}, [])
        res = SL.build_result(tr, CANDS, LANE)
        self.assertEqual({i["strip"] for i in res["issues"] if i["code"] == "layout_not_validated"}, {"left", "right"})


# --------------------------------------------------------------------------
# Streaming layer
# --------------------------------------------------------------------------
class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    sent = 0

    def log_message(self, *a):
        pass

    def setup(self):
        import socket
        super().setup()
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)   # bytes sent ~ bytes read

    def do_GET(self):
        path = self.translate_path(self.path)
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start, end = int(a or 0), (int(b) if b else size - 1)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    type(self).sent += len(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


@needs_ocr
@needs_ffmpeg
class StreamOverHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _font() is None:
            raise unittest.SkipTest("no bold TrueType font available")
        cls.tmp = Path(tempfile.mkdtemp())
        avi = cls.tmp / "s.avi"
        _write_video(avi, std_plan(), dur=150, fps=4)
        cls.mp4 = cls.tmp / "s.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(avi), "-c:v", "libx264", "-crf", "30",
                        "-g", "50", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(cls.mp4)], check=True)
        cls.size = cls.mp4.stat().st_size
        handler = functools.partial(_RangeHandler, directory=str(cls.tmp))
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/s.mp4"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_http_stream_recovers_the_sequence_and_reads_only_part_of_the_file(self):
        _RangeHandler.sent = 0
        work = Path(tempfile.mkdtemp(dir=self.tmp))
        out = work / "o.json"
        rc = X.main(["--game-id", GAME, "--video", self.url, "--out", str(out)])
        res = json.loads(out.read_text())
        self.assertEqual(rc, 0, res["issues"])
        self.assertEqual(_events_heroes(res), EXPECTED_SEQUENCE)
        self.assertEqual(res["provenance"]["source_kind"], "http_stream")
        self.assertTrue(res["diagnostics"]["stopped_early"])
        self.assertEqual([p.name for p in work.iterdir()], ["o.json"])
        self.assertLess(_RangeHandler.sent, 0.8 * self.size)
        print(f"\n   [http] served {_RangeHandler.sent} of {self.size} bytes")

    def test_seek_into_the_middle_keeps_only_the_bar_strip(self):
        frames = list(S.iter_strip_frames(self.url, 20.0, 22.0, 2.0))
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[0][0], 20.0)
        self.assertEqual(frames[0][1].shape, (720, 1280, 3))
        self.assertEqual(int(frames[0][1][:540].max()), 0)
        self.assertEqual(int(frames[0][1][715:].max()), 0)


class _FakeProc:
    def __init__(self, nframes):
        import io
        self.stdout = io.BytesIO(bytes(nframes * S.CANON_W * S.STRIP_H * 3))
        self.killed = False
        self._done = False

    def poll(self):
        return 0 if self._done else None

    def kill(self):
        self.killed = True

    def wait(self):
        self._done = True


def _info(w=1280, h=720, d=100.0):
    return lambda _src: {"width": w, "height": h, "duration": d}


class StreamLogicTests(unittest.TestCase):
    def test_resumes_after_premature_end_and_re_resolves_the_url(self):
        calls, urls = [], []

        def popen(cmd, **kw):
            calls.append(cmd)
            return _FakeProc(6 if len(calls) == 1 else 14)

        def resolver():
            urls.append("fresh")
            return "http://fresh/url.mp4"

        got = list(S.iter_strip_frames("http://old/url.mp4", 10.0, 20.0, 2.0, resolver=resolver,
                                       popen=popen, probe_fn=_info()))
        self.assertEqual(len(got), 20)
        self.assertEqual(round(got[6][0], 1), 13.0)
        self.assertEqual(urls, ["fresh"])
        self.assertEqual(calls[1][calls[1].index("-ss") + 1], "13.000")
        self.assertIn("http://fresh/url.mp4", calls[1])

    def test_gives_up_loudly_instead_of_returning_a_truncated_result(self):
        with self.assertRaises(S.StreamError):
            list(S.iter_strip_frames("x.mp4", 0.0, 20.0, 2.0, popen=lambda c, **k: _FakeProc(2),
                                     probe_fn=_info(), max_retries=2))

    def test_closing_the_generator_kills_ffmpeg(self):
        procs = []

        def popen(cmd, **kw):
            procs.append(_FakeProc(50))
            return procs[-1]

        it = S.iter_strip_frames("x.mp4", 0.0, 25.0, 2.0, popen=popen, probe_fn=_info())
        next(it)
        it.close()
        self.assertTrue(procs[0].killed)
        self.assertEqual(len(procs), 1)

    def test_rejects_non_16_9_and_sub_720p(self):
        with self.assertRaises(S.StreamError):
            list(S.iter_strip_frames("x", 0, 5, 1, probe_fn=_info(640, 480)))
        with self.assertRaises(S.StreamError):
            list(S.iter_strip_frames("x", 0, 5, 1, probe_fn=_info(854, 480)))

    def test_yt_dlp_resolver_returns_first_url_and_never_downloads(self):
        seen = {}

        def run(cmd, **kw):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="https://rr1.example/v.mp4\nhttps://x/a.m4a\n", stderr="")

        self.assertEqual(S.resolve_stream_url("https://youtu.be/ID", run=run), "https://rr1.example/v.mp4")
        self.assertIn("-g", seen["cmd"])
        self.assertNotIn("-o", seen["cmd"])

    def test_yt_dlp_failure_is_an_error(self):
        run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no 720p format")
        with self.assertRaises(S.StreamError):
            S.resolve_stream_url("https://youtu.be/ID", run=run)


class RealFrameLabelTests(unittest.TestCase):
    """Label reading (kept from the previous design; still the identity channel)."""

    @needs_ocr
    def test_labels_on_real_frames(self):
        vocab, cands = _vocab_and_cands()
        got = {(d.side, d.hero) for d in D.detect_labels(_real_canon("m1113_720p_probe.jpg"), vocab, cands)}
        self.assertEqual(got, {("blue", "masha"), ("blue", "marcel"), ("blue", "kimmy"), ("red", "atlas"),
                               ("red", "hirara"), ("red", "zhuxin"), ("red", "freya")})
        got = {(d.side, d.hero) for d in D.detect_labels(_real_canon("m1113_panel_reference.jpg"), vocab, cands)}
        self.assertEqual(got, {("blue", "masha"), ("red", "zhuxin"), ("red", "freya")})

    def test_non_16_9_frame_is_rejected_not_guessed(self):
        with self.assertRaises(ValueError):
            L.canonicalize(np.zeros((480, 640, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
