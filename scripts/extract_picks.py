#!/usr/bin/env python3
"""Slot-position-based chronological pick extraction for one game.

    python scripts/extract_picks.py --game-id mpl-id-s18-m1113-g4 --yt-dlp
    python scripts/extract_picks.py --game-id mpl-id-s18-m1113-g4 --video <file-or-url> --debug-dir debug
    python scripts/extract_picks.py --game-id mpl-id-s18-m1113-g4 --yt-dlp --probe-at 330

Stages
  1. FIND the draft bar from what frames look like (five player-name plates on
     both panels with a plausible slot geometry); no fixed timestamp.
  2. READ the slots in a dense scan from just before the bar appears.
  3. STOP when all ten slots are confirmed, when the bar has been gone for
     --loss-seconds, or at the end of the window.

Order = draft slot position (A1 B1 B2 A2 A3 B3 B4 A4 A5 B5), NOT the time a name
was first seen. Output: <project>/data/extractions/<game_id>.json with status
`machine_extracted_unreviewed`. Nothing else in the main project is modified.
The streaming path writes no video or frame files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from vod_extract import paths  # noqa: E402  (must come first: puts the main project's scripts on sys.path)

ROOT = paths.ensure_project_scripts_on_path()

from pipeline_lib import PICK_TURN_ORDER  # noqa: E402
from vod_extract import detect as D, search as SE, slots as SL, stream as S  # noqa: E402
from vod_extract.source import Source  # noqa: E402

VERSION = "0.4.0"
METHOD = "slot_position_ocr_label"
MAX_DEBUG_IMAGES = 12
MAX_PROBES_KEPT = 120


def load_game(game_id: str):
    games = json.loads((ROOT / "data/processed/mpl_id_s18_games.json").read_text(encoding="utf-8"))["games"]
    g = next((x for x in games if x["game_id"] == game_id), None)
    if g is None:
        raise SystemExit(f"unknown game id: {game_id}")
    heroes = json.loads((ROOT / "data/reference/heroes.json").read_text(encoding="utf-8"))["heroes"]
    return g, heroes


def _debug_image(canon, geoms, items, path):
    vis = canon[540:715].copy()
    for strip, g in geoms.items():
        if g.ok:
            for b in g.bounds:
                cv2.line(vis, (int(b), 0), (int(b), 174), (255, 128, 0), 1)
    for strip, idx, det in items:
        if det is not None:
            x, y, w, h = det.box
            cv2.rectangle(vis, (x, y - 540), (x + w, y - 540 + h), (0, 255, 0), 1)
            cv2.putText(vis, f"{strip[0]}{idx} {det.hero}", (max(0, x - 6), 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
    cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])


def probe(a, src, vocab, cands):
    """Text-only diagnosis of ONE frame: geometry, labels, slot assignment."""
    canon = src.sample(a.probe_at)
    if canon is None:
        raise SystemExit("no frame at that time")
    geoms, items, unassigned = SL.observe_frame(canon, vocab, cands, a.left_is, D.LabelCache())
    print(f"probe at {a.probe_at:.1f}s (left panel = {a.left_is}); stream height {src.height}; "
          f"bitrate {None if src.info is None else src.info.get('bit_rate')} bps; duration {src.duration}")
    for s in ("left", "right"):
        gg = geoms[s]
        print(f"  {s}: geometry {'OK' if gg.ok else 'INVALID: ' + str(gg.reason)}; "
              f"name centres {[round(c) for c in gg.centers]}; bounds {[round(b) for b in gg.bounds]}; "
              f"fit rms {None if gg.rms is None else round(gg.rms, 1)}px")
    by_slot = {(s, i): d for s, i, d in items}
    for _side, _ph, pos in PICK_TURN_ORDER:
        ad = SL.slot_address(pos, a.left_is)
        k = (ad["strip"], ad["strip_index"])
        d = by_slot.get(k)
        txt = "no label (empty / preview / placeholder)" if k not in by_slot else (
            "label UNREADABLE" if d is None else f"{d.hero} (score {d.score}, margin {d.margin})")
        print(f"  {pos}: {ad['strip']} panel slot {ad['strip_index']} from left -> {txt}")
    if unassigned:
        print("  label-shaped boxes not assigned to a slot:", [(s, k[0], why) for s, k, why in unassigned])
    return 0


def scan_window(src, tracker, vocab, cands, left_is, start, end, fps, lookahead, loss_s, dbg):
    """Dense scan. Returns (frames_scanned, stop_reason)."""
    cache = D.LabelCache()
    saved, saved_slots, n = 0, set(), 0
    seen_valid, last_valid_t = False, None
    for t, canon in src.frames_between(start, end, fps):
        geoms, items, unassigned = SL.observe_frame(canon, vocab, cands, left_is, cache)
        tracker.add_frame(t, geoms, items, unassigned)
        n += 1
        if geoms["left"].ok or geoms["right"].ok:
            seen_valid, last_valid_t = True, t
        confirmed = {pos for _s, _p, pos in PICK_TURN_ORDER if tracker.slot(pos)["status"] == "confirmed"}
        if dbg is not None and saved < MAX_DEBUG_IMAGES:
            for pos in sorted(confirmed - saved_slots):
                dbg.mkdir(parents=True, exist_ok=True)
                _debug_image(canon, geoms, items, dbg / f"slot_{pos}_{tracker.slot(pos)['hero']}.jpg")
                saved += 1
        saved_slots |= confirmed
        if len(confirmed) == 10:
            last_confirm_t = max(tracker.times[tracker.slot(p)["confirm_frame"]] for p in confirmed)
            if t >= last_confirm_t + lookahead:
                return n, "all_confirmed"
        if seen_valid and t - last_valid_t >= loss_s:
            return n, "layout_lost"
    return n, "end_of_window"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game-id", required=True)
    src_g = ap.add_mutually_exclusive_group(required=True)
    src_g.add_argument("--video", help="local file or http(s) media URL (streamed through ffmpeg, never saved)")
    src_g.add_argument("--yt-dlp", action="store_true",
                       help="resolve a direct stream URL from the game's VOD link with yt-dlp (no download)")
    src_g.add_argument("--frames-dir", help="folder of <seconds>.jpg frames (tests / manual use)")
    ap.add_argument("--probe-at", type=float, help="print geometry and slot labels of the frame at this second, then exit")
    ap.add_argument("--start", type=float, help="skip the draft search and start the dense scan here")
    ap.add_argument("--end", type=float)
    ap.add_argument("--search-from", type=float, help="where the draft search begins (default 0 for single-game VODs)")
    ap.add_argument("--search-limit", type=float, default=1200.0, help="how far the draft search may look (default 1200 s)")
    ap.add_argument("--search-step", type=float, default=15.0, help="coarse step of the draft search (default 15 s)")
    ap.add_argument("--max-seconds", type=float, default=900.0, help="longest dense scan (default 900 s)")
    ap.add_argument("--loss-seconds", type=float, default=25.0,
                    help="stop when the draft bar has been gone this long (default 25 s)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--lookahead", type=float, default=3.0)
    ap.add_argument("--left-is", choices=["blue", "red"], default="blue")
    ap.add_argument("--no-turn-check", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--debug-dir", help=f"save up to {MAX_DEBUG_IMAGES} small JPEGs (bar strip when a slot is confirmed)")
    a = ap.parse_args(argv)

    g, heroes = load_game(a.game_id)
    vocab = [h["name"] for h in heroes]
    id_of = {h["name"].lower(): h["hero_id"] for h in heroes}
    name_of = {h["name"].lower(): h["name"] for h in heroes}
    cands = {s: {h["hero"].lower() for h in g["draft"]["picks"][s]} for s in ("blue", "red")}
    lane_order = {s: [h["hero"].lower() for h in g["draft"]["picks"][s]] for s in ("blue", "red")}
    vod = g.get("vod", {})

    start = a.start if a.start is not None else vod.get("start_seconds")
    unknown_start = {"vod_scope_series", "vod_is_full_day_livestream_start_time_unknown"} & set(vod.get("issues") or [])
    if a.probe_at is None and start is None and a.search_from is None:
        if vod.get("scope") != "game" or unknown_start:
            raise SystemExit(f"VOD is not a single-game video (scope={vod.get('scope')!r}, "
                             f"issues={sorted(unknown_start)}): pass --start or --search-from for this game's draft")

    try:
        src = Source(video=a.video, frames_dir=a.frames_dir,
                     resolver=(lambda: S.resolve_stream_url(vod["url"])) if a.yt_dlp else None)
        if a.probe_at is not None:
            return probe(a, src, vocab, cands)

        search = {"mode": "explicit_start" if start is not None else "auto"}
        found = start
        if start is None:
            lo = a.search_from or 0.0
            hi = min(lo + a.search_limit, max(lo, (src.duration or lo + a.search_limit) - 2.0))
            found, probes = SE.find_draft_start(src.sample, lo, hi, a.search_step, burst=src.burst)
            search.update({"range_s": [lo, hi], "step_s": a.search_step, "found_at_s": found,
                           "probes": [[t, ok] for t, ok in probes[:MAX_PROBES_KEPT]]})
        else:
            search["found_at_s"] = start

        tracker = SL.SlotTracker(a.left_is)
        n, stop_reason, scan_range = 0, "draft_not_found", None
        if found is not None:
            lead = a.search_step if start is None else 0.0
            s0 = max(0.0, found - lead)
            end = a.end if a.end is not None else s0 + a.max_seconds
            scan_range = [s0, end]
            n, stop_reason = scan_window(src, tracker, vocab, cands, a.left_is, s0, end, a.fps, a.lookahead,
                                         a.loss_seconds, Path(a.debug_dir) if a.debug_dir else None)
    except S.StreamError as e:
        raise SystemExit(f"stream error: {e}")

    stage_issues = []
    if found is None:
        res = {"slots": {}, "order": [], "passed": False,
               "checks": {"draft_found": False}, "issues": []}
        stage_issues.append({"code": "draft_not_found",
                             "detail": f"no frame with a valid draft-bar layout between {search['range_s']} s "
                                       f"(step {a.search_step} s); wrong VOD/window, different layout, or stream below 720p"})
        slot_diag = []
    else:
        res = SL.build_result(tracker, cands, lane_order, turn_check=not a.no_turn_check)
        res["checks"] = {"draft_found": True, **res["checks"]}
        if not res["checks"]["all_slots_confirmed"]:
            if stop_reason == "layout_lost":
                stage_issues.append({"code": "draft_layout_lost_before_complete",
                                     "detail": f"the draft bar was gone for {a.loss_seconds:.0f}s before all ten slots were confirmed"})
            elif stop_reason == "end_of_window":
                stage_issues.append({"code": "scan_window_exhausted",
                                     "detail": "reached the end of the scan window (or of the video) before all ten slots were confirmed"})
        slot_diag = []
        for _s, _p, pos in PICK_TURN_ORDER:
            r = res["slots"][pos]
            slot_diag.append({"position": pos, "status": r["status"], "hero": r["hero"],
                              "n_reads": r["n_reads"], "mean_score": r.get("mean_score"),
                              "min_margin": r.get("min_margin"), "changed_after_lock": r.get("changed_after_lock"),
                              "heroes_seen": r.get("heroes_seen")})
    issues = stage_issues + res["issues"]
    events = None
    if res["passed"]:
        events = []
        for i, ((side, phase, pos), (_p, hero)) in enumerate(zip(PICK_TURN_ORDER, res["order"]), 1):
            ad = SL.slot_address(pos, a.left_is)
            events.append({"seq": i, "side": side, "team": g["teams"][side], "action": "pick", "phase": phase,
                           "side_index": ad["slot_from_outer"], "position": pos,
                           "hero_id": id_of[hero], "hero": name_of[hero],
                           "slot": {"panel": ad["strip"], "slot_from_outer": ad["slot_from_outer"],
                                    "slot_from_left": ad["strip_index"]},
                           "order_status": "machine_extracted_unreviewed", "origin": "vod_slot_position_extraction"})
    reasons = [] if events else [k for k, v in res["checks"].items() if not v] + [i["code"] for i in stage_issues]
    dbg = Path(a.debug_dir) if a.debug_dir else None
    result = {
        "schema": "vod_extraction/4",
        "game_id": a.game_id,
        "result": "extracted" if events else "quarantined",
        "quarantine_reasons": reasons,
        "verification": {"status": "machine_extracted_unreviewed", "verified_by_vod": False, "human_reviewed": False},
        "provenance": {
            "method": METHOD, "extractor_version": VERSION, "vod_id": vod.get("video_id"),
            "source_kind": src.kind, "stream_height": src.height,
            "stream_bitrate_bps": None if src.info is None else src.info.get("bit_rate"), "left_is": a.left_is,
            "vocabulary_size": len(vocab),
            "order_uses_slot_position": True, "order_uses_first_seen_time": False,
            "order_uses_source_lane_order": False,
            "time_used_for": None if a.no_turn_check else "turn-structure consistency check only (quarantine, never ordering)",
        },
        "events": events,
        "checks": res["checks"],
        "issues": issues,
        "diagnostics": {"draft_search": search, "scan_window_s": scan_range, "stop_reason": stop_reason,
                        "frames_scanned": n, "sampling_fps": a.fps,
                        "frames_with_valid_geometry": dict(tracker.geom_ok),
                        "labels_not_assigned_to_a_slot": dict(tracker.unassigned),
                        "slots": slot_diag},
        "notes": ["Machine output. Confirm in vod-annotator before promoting to verified_by_vod."],
    }
    out = Path(a.out) if a.out else ROOT / "data/extractions" / f"{a.game_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    out.write_text(text, encoding="utf-8")
    result["storage"] = {"video_or_frame_files_written": 0,
                         "persistent_bytes": len(text.encode("utf-8")) + (
                             sum(f.stat().st_size for f in dbg.glob("*.jpg")) if dbg and dbg.exists() else 0)}
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{a.game_id}: {result['result']}; draft bar found at {found}s; frames {n}; stopped: {stop_reason}")
    for _s, _p, pos in PICK_TURN_ORDER:
        r = res["slots"].get(pos)
        if r:
            print(f"  {pos:2s} {r['status']:18s} {r['hero'] or ''}")
    for i in issues:
        print("  !", i["code"], {k: v for k, v in i.items() if k != "code"})
    print("wrote", out, f"({result['storage']['persistent_bytes']} bytes persistent)")
    return 0 if events else 2


if __name__ == "__main__":
    sys.exit(main())
