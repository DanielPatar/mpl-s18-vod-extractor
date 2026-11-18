"""One video source (local file, http(s) URL, yt-dlp-resolved URL, or frames folder).

Hides three things from the extractor: the stream URL (signed, expires: it is
re-resolved once if a read fails), the one-time ffprobe, and the difference
between a seekable video and a folder of frames.
"""
from __future__ import annotations

from . import labels as L, stream as S, video as V


class Source:
    def __init__(self, video=None, frames_dir=None, resolver=None, popen=None, probe_fn=None):
        self.resolver = resolver
        self.frames = V.FrameDir(frames_dir) if frames_dir else None
        self.target = None if frames_dir else (resolver() if (resolver and not video) else video)
        self._kw = {k: v for k, v in (("popen", popen), ("probe_fn", probe_fn)) if v is not None}
        self.info = None
        if self.frames is not None:
            self.duration = self.frames.duration
        else:
            self.info = self._probe()
            self.duration = self.info["duration"]
        self.kind = "frames_dir" if frames_dir else ("yt_dlp_stream" if resolver and not video else
                    ("http_stream" if str(video).startswith("http") else "local_file"))

    def _probe(self):
        fn = self._kw.get("probe_fn", S.probe)
        try:
            return fn(self.target)
        except S.StreamError:
            if not self.resolver:
                raise
            self.target = self.resolver()
            return fn(self.target)

    @property
    def height(self):
        return None if self.info is None else self.info["height"]

    def _with_reresolve(self, fn):
        try:
            return fn()
        except S.StreamError:
            if not self.resolver:
                raise
            self.target = self.resolver()          # expired signed URL: one fresh attempt
            return fn()

    def sample(self, t: float):
        """canon frame at ~t or None."""
        if self.frames is not None:
            fr = self.frames.frame_at(t)
            return None if fr is None else L.canonicalize(fr)
        got = self._with_reresolve(lambda: S.grab_frame(self.target, t, info=self.info, **self._kw))
        return None if got is None else got[1]

    def burst(self, t: float, gaps):
        """Frames at t+gap for each gap (evenly spaced), using ONE ffmpeg launch.
        Every launch costs ~1 MB of header/seek traffic, so confirmation of a
        candidate must not launch once per frame. Returns a list (None = missing)."""
        gaps = sorted(gaps)
        if self.frames is not None:
            return [self.sample(t + g) for g in gaps]
        step = gaps[1] - gaps[0] if len(gaps) > 1 else 1.0
        got = {}

        def run():
            for tt, canon in S.iter_strip_frames(self.target, t + gaps[0], t + gaps[-1] + 0.4 * step,
                                                 1.0 / step, info=self.info, max_retries=0, **self._kw):
                got[round(tt - t, 3)] = canon
        self._with_reresolve(run)
        return [got.get(round(g, 3)) for g in gaps]

    def frames_between(self, start: float, end: float, fps: float):
        """Yield (t, canon) sequentially."""
        if self.frames is not None:
            for t in self.frames.times(start, end, 1.0 / fps):
                fr = self.frames.frame_at(t)
                if fr is not None:
                    yield t, L.canonicalize(fr)
            return
        yield from S.iter_strip_frames(self.target, start, end, fps, resolver=self.resolver,
                                       info=self.info, **self._kw)
