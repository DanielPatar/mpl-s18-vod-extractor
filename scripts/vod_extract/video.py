"""Frame sources. Reads a LOCAL video file or a folder of frames; it never
downloads anything (docs/ANNOTATION_WORKFLOW.md section 7)."""
from __future__ import annotations

import re
from pathlib import Path

import cv2


class VideoSource:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.duration = (n / fps) if fps else 0.0

    def frame_at(self, t: float):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = self.cap.read()
        return fr if ok else None

    def times(self, start, end, step):
        end = self.duration if end is None else min(end, self.duration)
        t = start
        while t <= end + 1e-9:
            yield round(t, 3)
            t += step


class FrameDir:
    """Folder of images whose file stem is the time in seconds, e.g. 366.5.jpg
    or 0366.50.png. Sampling picks the nearest available frame."""

    def __init__(self, path: str):
        self.items = []
        for p in sorted(Path(path).iterdir()):
            m = re.fullmatch(r"\d+(?:\.\d+)?", p.stem)
            if m and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self.items.append((float(p.stem), p))
        if not self.items:
            raise FileNotFoundError(f"no <seconds>.jpg/png frames in {path}")
        self.items.sort()
        self.duration = self.items[-1][0]

    def frame_at(self, t: float):
        ts = [x[0] for x in self.items]
        i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
        if abs(ts[i] - t) > 1.0:
            return None
        return cv2.imread(str(self.items[i][1]))

    def times(self, start, end, step):
        end = self.duration if end is None else min(end, self.duration)
        seen, t = set(), start
        while t <= end + 1e-9:
            seen.add(round(t, 3)); t += step
        return sorted(seen)


def open_source(video=None, frames_dir=None):
    if bool(video) == bool(frames_dir):
        raise ValueError("give exactly one of --video / --frames-dir")
    return VideoSource(video) if video else FrameDir(frames_dir)
