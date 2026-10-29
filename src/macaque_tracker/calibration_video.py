"""Shared recorded-video controls; never used by the live tracking pipeline."""

from __future__ import annotations

import time

from .calibration_dialogs import choose_video
from .camera import VideoFileCamera


class CalibrationVideo:
    """Switchable preview with one retained raw frame and exact paused stepping."""

    def __init__(self, config, fallback):
        self.config = config
        self.fallback = fallback
        self.camera = None
        self.playing = False
        self.image = None
        self._next_frame = 0.0
        self._at_end = False

    def load(self, path):
        camera = VideoFileCamera(path, self.config, realtime=False, loop=False)
        try:
            camera.start()
            image, _ = camera.capture_preview()
        except BaseException:
            camera.close()
            raise
        self.close()
        self.camera, self.image = camera, image

    def buttons(self):
        buttons = [("Load video", "load_video")]
        if self.camera is not None:
            buttons += [("Pause" if self.playing else "Play", "play_video"),
                        ("Frame -1", "frame_back"), ("Frame +1", "frame_forward")]
        return buttons

    def action(self, action, *, path=None, origin=None):
        if action == "load_video":
            path = path or choose_video()
            if not path:
                return False
            self.load(path)
        elif action == "play_video":
            if origin is not None:
                self.playing = False
                if origin != self.camera.frame_index:
                    self.image, _ = self.camera.seek_preview(origin)
                    self._at_end = False
            self.toggle()
        else:
            return self.step(-1 if action == "frame_back" else 1, origin=origin)
        return True

    @property
    def description(self):
        if self.camera is None:
            return "Load video: O"
        count = self.camera.frame_count
        return (f"{self.camera.video_path.name} | Frame {self.camera.frame_index + 1}"
                f"{' / ' + str(count) if count else ''} | "
                f"{'Playing' if self.playing else 'Paused'} | Space play/pause | [ / ] step")

    def latest(self):
        if self.camera is None:
            return self.fallback()
        now = time.monotonic()
        if self.playing and now >= self._next_frame:
            count = self.camera.frame_count
            due = min(4, 1 + int((now - self._next_frame) * self.camera.video_fps))
            for _ in range(due):
                if count and self.camera.frame_index >= count - 1:
                    self.playing = False
                    self._at_end = True
                    break
                try:
                    self.image, _ = self.camera.capture_preview()
                except EOFError:
                    self.playing = False
                    self._at_end = True
                    break
            self._next_frame = now + 1 / self.camera.video_fps
        return self.image

    def step(self, direction, *, origin=None):
        if self.camera is None:
            raise ValueError("Load a video to step between frames")
        self.playing = False
        self._at_end = False
        index = self.camera.frame_index if origin is None else origin
        try:
            self.image, _ = self.camera.seek_preview(index + direction)
        except EOFError:
            self._at_end = True
        return self.camera.frame_index != index

    def toggle(self):
        if self.camera is None:
            raise ValueError("Load a video to use playback controls")
        count = self.camera.frame_count
        if not self.playing and (self._at_end or (count and self.camera.frame_index == count - 1)):
            self.image, _ = self.camera.seek_preview(0)
        self._at_end = False
        self.playing = not self.playing
        self._next_frame = time.monotonic() + 1 / self.camera.video_fps

    def close(self):
        if self.camera is not None:
            self.camera.close()
        self.camera = None
        self.playing = False
        self.image = None
        self._at_end = False
