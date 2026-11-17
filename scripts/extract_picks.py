#!/usr/bin/env python3
"""Slot-position-based chronological pick extraction for one game.

    python scripts/extract_picks.py --game-id mpl-id-s18-m1113-g4 --probe-at 330 --video <file-or-url>
    python scripts/extract_picks.py --game-id mpl-id-s18-m1113-g4 --video <file-or-url> --debug-dir debug

Order = draft slot position (A1 B1 B2 A2 A3 B3 B4 A4 A5 B5), NOT the time a
name was first seen. A = blue team, slot n from its panel's outer edge; B = red
team likewise. A slot counts only when its name label is on screen in
several frames (a locked pick); previews, placeholders and bans never count.
The identity is frozen when confirmed, so post-draft swaps cannot alter it.

Output: data/extractions/<game_id>.json, status `machine_extracted_unreviewed`.
Nothing in data/processed or data/annotations is touched. The streaming path
writes no video or frame files (see scripts/vod_extract/stream.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2] / "mpl-s18-draft-project"
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_lib import PICK_TURN_ORDER  # noqa: E402
from vod_extract import detect as D, geometry as G, labels as L, slots as SL, stream as S, video as V  # noqa: E402

VERSION = "0.3.0"
METHOD = "slot_position_ocr_label"
MAX_DEBUG_IMAGES = 12


def load_game(game_id: str):
    games = json.loads((ROOT / "data/processed/mpl_id_s18_games.json").read_text(encoding="utf-8"))["games"]
    g = next((x for x in games if x["game_id"] == game_id), None)
    if g is None:
        raise SystemExit(f"unknown game id: {game_id}")
    heroes = json.loads((ROOT / "data/reference/heroes.json").read_text(encoding="utf-8"))["heroes"]
    return g, heroes


def frame_iter(a, start, end, resolver):
    if a.frames_dir:
        src = V.FrameDir(a.frames_dir)
        for t in src.times(start, end, 1.0 / a.fps):
            fr = src.frame_at(t)
            if fr is not None:
                yield t, L.canonicalize(fr)
        return
    target = resolver() if a.yt_dlp else a.video
    yield from S.iter_strip_frames(target, start, end, a.fps, resolver=resolver if a.yt_dlp else None)


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


def probe(a, g, vocab, cands):
    """Text-only diagnosis of ONE frame: geometry, labels, slot assignment."""
    t0 = a.probe_at
    it = frame_iter(a, t0, t0 + 2.0, lambda: S.resolve_stream_url(g["vod"]["url"]))
    t, canon = next(it, (None, None))
    it.close() if hasattr(it, "close") else None
    if canon is None:
        raise SystemExit("no frame at that time")
    geoms, items, unassigned = SL.observe_frame(canon, vocab, cands, a.left_is, D.LabelCache())
    print(f"probe at {t:.1f}s (left panel = {a.left_is})")
    for s in ("left", "right"):
        gg = geoms[s]
        print(f"  {s}: geometry {'OK' if gg.ok else 'INVALID: ' + str(gg.reason)}; "
              f"name centres {[round(c) for c in gg.centers]}; bounds {[round(b) for b in gg.bounds]}; "
              f"fit rms {None if gg.rms is None else round(gg.rms, 1)}px")
    by_slot = {(s, i): d for s, i, d in items}
    for _side, _ph, pos in PICK_TURN_ORDER:
        ad = SL.slot_address(pos, a.left_is)
        d = by_slot.get((ad["strip"], ad["strip_index"]))
        txt = "no label (empty / preview / placeholder)" if (ad["strip"], ad["strip_index"]) not in by_slot else (
            "label UNREADABLE" if d is None else f"{d.hero} (score {d.score}, margin {d.margin})")
        print(f"  {pos}: {ad['strip']} panel slot {ad['strip_index']} from left -> {txt}")
    if unassigned:
        print("  label-shaped boxes not assigned to a slot:", [(s, k[0], why) for s, k, why in unassigned])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game-id", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="local file or http(s) media URL (streamed through ffmpeg, never saved)")
    src.add_argument("--yt-dlp", action="store_true",
                     help="opt-in: resolve a direct stream URL from the game's VOD link with yt-dlp (no download)")
    src.add_argument("--frames-dir", help="folder of <seconds>.jpg frames (tests / manual use)")
    ap.add_argument("--probe-at", type=float, help="print geometry and slot labels of the frame at this second, then exit")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--lookahead", type=float, default=3.0,
                    help="seconds to keep scanning after the last slot is confirmed (default 3)")
    ap.add_argument("--left-is", choices=["blue", "red"], default="blue",
                    help="which team's picks are on the LEFT panel (m1113-g4: blue)")
    ap.add_argument("--no-turn-check", action="store_true",
                    help="disable the lock-time consistency check (not recommended)")
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
    if a.probe_at is None and start is None:
        if vod.get("scope") != "game" or unknown_start:
            raise SystemExit(f"VOD is not a single-game video (scope={vod.get('scope')!r}, "
                             f"issues={sorted(unknown_start)}): pass --start for this game's draft")
        start = 0.0
    if a.probe_at is not None:
        try:
            return probe(a, g, vocab, cands)
        except S.StreamError as e:
            raise SystemExit(f"stream error: {e}")
    end = a.end if a.end is not None else start + a.max_seconds

    tracker = SL.SlotTracker(a.left_is)
    cache = D.LabelCache()
    dbg = Path(a.debug_dir) if a.debug_dir else None
    saved, saved_slots, stopped_early, n = 0, set(), False, 0
    last_confirm_t = None
    try:
        for t, canon in frame_iter(a, start, end, lambda: S.resolve_stream_url(vod["url"])):
            geoms, items, unassigned = SL.observe_frame(canon, vocab, cands, a.left_is, cache)
            tracker.add_frame(t, geoms, items, unassigned)
            n += 1
            confirmed = {pos for _s, _p, pos in PICK_TURN_ORDER if tracker.slot(pos)["status"] == "confirmed"}
            if dbg is not None and saved < MAX_DEBUG_IMAGES:
                for pos in sorted(confirmed - saved_slots):
                    dbg.mkdir(parents=True, exist_ok=True)
                    _debug_image(canon, geoms, items, dbg / f"slot_{pos}_{tracker.slot(pos)['hero']}.jpg")
                    saved += 1
            saved_slots |= confirmed
            if len(confirmed) == 10:
                last_confirm_t = max(tracker.times[tracker.slot(p)["confirm_frame"]] for p in confirmed)
                if t >= last_confirm_t + a.lookahead:
                    stopped_early = True
                    break
    except S.StreamError as e:
        raise SystemExit(f"stream error: {e}")

    res = SL.build_result(tracker, cands, lane_order, turn_check=not a.no_turn_check)
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
    failed = [k for k, v in res["checks"].items() if not v]
    slot_diag = []
    for _s, _p, pos in PICK_TURN_ORDER:
        r = res["slots"][pos]
        slot_diag.append({"position": pos, "status": r["status"], "hero": r["hero"],
                          "n_reads": r["n_reads"], "mean_score": r.get("mean_score"),
                          "min_margin": r.get("min_margin"), "changed_after_lock": r.get("changed_after_lock"),
                          "heroes_seen": r.get("heroes_seen")})
    result = {
        "schema": "vod_extraction/3",
        "game_id": a.game_id,
        "result": "extracted" if events else "quarantined",
        "quarantine_reasons": failed if not events else [],
        "verification": {"status": "machine_extracted_unreviewed", "verified_by_vod": False, "human_reviewed": False},
        "provenance": {
            "method": METHOD, "extractor_version": VERSION, "vod_id": vod.get("video_id"),
            "source_kind": "frames_dir" if a.frames_dir else ("yt_dlp_stream" if a.yt_dlp else
                           ("http_stream" if str(a.video).startswith("http") else "local_file")),
            "left_is": a.left_is, "vocabulary_size": len(vocab),
            "order_uses_slot_position": True,
            "order_uses_first_seen_time": False,
            "order_uses_source_lane_order": False,
            "time_used_for": None if a.no_turn_check else "turn-structure consistency check only (quarantine, never ordering)",
        },
        "events": events,
        "checks": res["checks"],
        "issues": res["issues"],
        "diagnostics": {"frames_scanned": n, "stopped_early": stopped_early, "scan_start_s": start,
                        "sampling_fps": a.fps,
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

    print(f"{a.game_id}: {result['result']}; frames {n}{' (stopped early)' if stopped_early else ''}")
    for _s, _p, pos in PICK_TURN_ORDER:
        r = res["slots"][pos]
        print(f"  {pos:2s} {r['status']:18s} {r['hero'] or ''}")
    for i in res["issues"]:
        print("  !", i["code"], {k: v for k, v in i.items() if k != "code"})
    print("wrote", out, f"({result['storage']['persistent_bytes']} bytes persistent)")
    return 0 if events else 2


if __name__ == "__main__":
    sys.exit(main())
