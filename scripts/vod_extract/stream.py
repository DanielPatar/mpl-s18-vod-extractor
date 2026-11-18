"""Storage-free frame source: ffmpeg decodes a local file or an http(s) stream
and pipes ONLY the draft-bar strip to this process. Nothing is written to disk.

    URL/file --ffmpeg(-ss, fps, scale 1280x720, crop bar strip)--> pipe --> numpy

Properties
* Seeking is done on the input side (`-ss` before `-i`), which uses HTTP range
  requests when the server supports them; otherwise ffmpeg reads from the start
  (correct but slower).
* Early stop: closing the generator kills ffmpeg at once, so the rest of the
  video is never fetched past what the pipe already buffered.
* Interruptions: ffmpeg's own reconnect options are enabled, and if the stream
  still ends well before the requested end, decoding resumes from the last
  frame, re-resolving the URL first when a resolver is given (signed URLs
  expire). After `max_retries` it raises StreamError instead of returning a
  silently truncated result.
* Only 16:9 input is accepted (strip coordinates assume it).
"""
from __future__ import annotations

import json
import subprocess

import numpy as np

CANON_W, CANON_H = 1280, 720
STRIP_Y0, STRIP_H = 540, 175          # bar rows kept: 540..714 (labels y~557, player-name plates y~697-709)


class StreamError(RuntimeError):
    pass


def _is_url(src: str) -> bool:
    return src.startswith(("http://", "https://"))


def _net_opts(src: str) -> list[str]:
    if not _is_url(src):
        return []
    return ["-rw_timeout", "20000000", "-reconnect", "1", "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5"]


def probe(src: str, run=subprocess.run) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,bit_rate:format=duration,bit_rate", "-of", "json", src]
    if _is_url(src):
        cmd[1:1] = ["-rw_timeout", "20000000"]
    r = run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise StreamError(f"ffprobe failed: {r.stderr.strip()[-300:]}")
    j = json.loads(r.stdout or "{}")
    st = (j.get("streams") or [{}])[0]
    if not st.get("width"):
        raise StreamError("no video stream found")
    dur = j.get("format", {}).get("duration")
    br = st.get("bit_rate") or j.get("format", {}).get("bit_rate")
    return {"width": int(st["width"]), "height": int(st["height"]),
            "duration": float(dur) if dur not in (None, "N/A") else None,
            "bit_rate": int(br) if br not in (None, "N/A") else None}


def resolve_stream_url(video_url: str, min_height: int = 720, max_height: int = 720,
                       run=subprocess.run) -> str:
    """Ask yt-dlp for a direct media URL WITHOUT downloading anything.

    Prefers exactly 720p (labels were validated on a real 720p frame, and 720p
    costs roughly half the bytes of 1080p); falls back to anything from
    min_height up to 1080p. The URL is signed and expires: call again for a new one."""
    h = max(max_height, 1080)
    fmt = (f"bv*[height<={max_height}][height>={min_height}]"
           f"/bv*[height<=?{h}][height>={min_height}]/b[height<=?{h}][height>={min_height}]")
    r = run(["yt-dlp", "-g", "--no-playlist", "-f", fmt, video_url],
            capture_output=True, text=True, timeout=120)
    lines = [x for x in (r.stdout or "").splitlines() if x.startswith("http")]
    if r.returncode != 0 or not lines:
        raise StreamError(f"yt-dlp gave no stream URL (>= {min_height}p): {(r.stderr or '').strip()[-300:]}")
    return lines[0]


def _read_exact(f, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def iter_strip_frames(src: str, start: float, end: float | None, fps: float,
                      resolver=None, max_retries: int = 3, popen=subprocess.Popen,
                      probe_fn=probe, info: dict | None = None):
    """Yield (t_seconds, canon) where canon is a 720x1280x3 BGR array whose rows
    540..699 hold the bar strip and everything else is zero."""
    info = info or probe_fn(src)
    if abs(info["width"] / info["height"] - 16 / 9) > 0.02:
        raise StreamError(f"video is {info['width']}x{info['height']}, not 16:9")
    if info["height"] < 720:
        raise StreamError(f"video is {info['height']}p; 720p or higher is required for reliable OCR")
    dur = info["duration"]
    stop = end if end is not None else dur
    if stop is not None and dur is not None:
        stop = min(stop, dur)
    nbytes = CANON_W * STRIP_H * 3
    done = 0                                        # frames already yielded
    attempts = 0
    while True:
        t0 = start + done / fps
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t0:.3f}",
               *_net_opts(src), "-i", src]
        if stop is not None:
            cmd += ["-t", f"{max(0.0, stop - t0):.3f}"]
        cmd += ["-an", "-sn", "-vf",
                f"fps={fps},scale={CANON_W}:{CANON_H}:flags=area,crop={CANON_W}:{STRIP_H}:0:{STRIP_Y0}",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=nbytes * 2)
        try:
            while True:
                raw = _read_exact(proc.stdout, nbytes)
                if raw is None:
                    break
                canon = np.zeros((CANON_H, CANON_W, 3), np.uint8)
                canon[STRIP_Y0:STRIP_Y0 + STRIP_H] = np.frombuffer(raw, np.uint8).reshape(STRIP_H, CANON_W, 3)
                yield start + done / fps, canon
                done += 1
        finally:                                    # also runs when the consumer stops early
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            try:
                proc.stdout.close()
            except Exception:
                pass
        reached = start + done / fps
        if stop is None or reached >= stop - 1.5:
            return
        attempts += 1
        if attempts > max_retries:
            raise StreamError(f"stream ended at {reached:.1f}s, expected {stop:.1f}s, after {max_retries} retries")
        if resolver is not None:
            src = resolver()


def grab_frame(src: str, t: float, info: dict | None = None, popen=subprocess.Popen, probe_fn=probe):
    """One (t, canon) frame at about t, or None past the end. Used by the draft search."""
    it = iter_strip_frames(src, t, t + 1.0, 1.0, info=info, popen=popen, probe_fn=probe_fn, max_retries=0)
    try:
        return next(it, None)
    finally:
        it.close()
