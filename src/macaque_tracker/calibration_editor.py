"""Toolkit-independent control-image editing and cancellable offline calibration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np

from .calibration import (
    ControlSample,
    load_control_samples,
    optimize_calibration,
    validate_control_samples,
)


def load_controls(path, context):
    """A damaged optional label archive must not prevent fresh calibration."""
    try:
        return load_control_samples(path, context=context), ""
    except (OSError, ValueError) as exc:
        return (), f"Saved control images unavailable; recapture them: {exc}"


def control_context(image, boxes, rotations) -> dict:
    """Reject saved labels when their raw crop coordinates or orientation change."""
    return {
        "source_size": list(image.shape[::-1]),
        "boxes": [[box.x, box.y, box.width, box.height] for box in boxes],
        "rotations": list(rotations),
    }


class ControlImageEditor:
    """Keep original pixels, explicit labels, and worker results off the live path."""

    def __init__(self, eye_count: int, samples=()) -> None:
        self.samples = [list(eye) for eye in samples] if samples else [[] for _ in range(eye_count)]
        if len(self.samples) != eye_count:
            raise ValueError("Control images must match the selected eyes")
        self.labelled = [[True] * len(eye) for eye in self.samples]
        self.eye = 0
        self.index: int | None = None
        self.brush = 4
        self.erase = False
        self.show_prediction = True
        self.status = "Capture varied gaze directions, pupil sizes, and a blink: 2-6 images per eye recommended; more welcome."
        self.revision = 0
        self.results = [None] * eye_count
        self._executor = None
        self._pending = None
        self._cancel = Event()

    @property
    def busy(self) -> bool:
        return self._pending is not None

    @property
    def sample(self):
        return None if self.index is None else self.samples[self.eye][self.index]

    def _editable(self) -> None:
        if self.busy:
            raise ValueError("Cancel optimization before editing control images")

    def _changed(self) -> None:
        self.results[self.eye] = None
        self.revision += 1

    def select_eye(self, eye: int) -> None:
        self._editable()
        if not 0 <= eye < len(self.samples):
            raise ValueError("Unknown eye")
        self.eye, self.index = eye, None
        self.revision += 1

    def capture(self, crop: np.ndarray, *, replace_unlabelled: bool = False) -> None:
        self._editable()
        sample = ControlSample(crop, np.zeros_like(crop))
        if (replace_unlabelled and self.index is not None
                and not self.labelled[self.eye][self.index]):
            samples = list(self.samples[self.eye])
            samples[self.index] = sample
            validate_control_samples(samples)
            self.samples[self.eye] = samples
            self._changed()
            return
        validate_control_samples((*self.samples[self.eye], sample))
        self.samples[self.eye].append(sample)
        self.labelled[self.eye].append(False)
        self.index = len(self.samples[self.eye]) - 1
        self.status = "Paint the whole pupil with left drag; erase with right drag. No pupil labels a blink."
        self._changed()

    def browse(self, direction: int) -> None:
        if self.samples[self.eye]:
            self.index = (
                (0 if direction > 0 else len(self.samples[self.eye]) - 1)
                if self.index is None
                else (self.index + direction) % len(self.samples[self.eye])
            )
            self.revision += 1

    def live(self) -> None:
        self.index = None
        self.revision += 1

    def delete(self) -> None:
        self._editable()
        if self.index is not None:
            self.samples[self.eye].pop(self.index)
            self.labelled[self.eye].pop(self.index)
            self.index = min(self.index, len(self.samples[self.eye]) - 1)
            if self.index < 0:
                self.index = None
            self._changed()

    def no_pupil(self) -> None:
        self._editable()
        if self.sample is None:
            raise ValueError("Capture or select a control image first")
        sample = self.sample
        self.samples[self.eye][self.index] = sample.with_mask(np.zeros_like(sample.mask))
        self.labelled[self.eye][self.index] = True
        self.status = "Labelled No pupil. Include open-eye images before optimizing."
        self._changed()

    def paint(self, start, end, *, erase=False) -> None:
        self._editable()
        if self.sample is None:
            return
        import cv2

        sample = self.sample
        mask = sample.mask.copy()
        color = 0 if erase or self.erase else 255
        cv2.line(mask, start, end, color, self.brush * 2, cv2.LINE_8)
        cv2.circle(mask, end, self.brush, color, -1)
        self.samples[self.eye][self.index] = sample.with_mask(mask)
        # An erased-to-empty image is unfinished until No pupil is explicit.
        self.labelled[self.eye][self.index] = bool(np.any(mask))
        self._changed()

    def export(self):
        if any(not all(labels) for labels in self.labelled):
            raise ValueError("Paint, delete, or mark No pupil on every captured image before saving")
        return tuple(tuple(eye) for eye in self.samples)

    def optimize(self, settings, tracker_config) -> None:
        self._editable()
        samples = self.samples[self.eye]
        if len(samples) < 2:
            raise ValueError("Capture at least 2 control images for this eye before optimizing")
        if not all(self.labelled[self.eye]):
            raise ValueError("Paint or mark No pupil on every selected-eye image before optimizing")
        if not any(np.any(sample.mask) for sample in samples):
            raise ValueError("Include at least one painted pupil before optimizing")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pupil-calibration")
        self._cancel.clear()
        snapshot = tuple(samples)  # ControlSample owns immutable pixels; edits replace samples.
        future = self._executor.submit(
            optimize_calibration, snapshot, settings, tracker_config, cancelled=self._cancel.is_set,
        )
        self._pending = (self.eye, future)
        self.status = f"Optimizing eye {self.eye} from {len(samples)} control images... Cancel keeps current settings."
        self.revision += 1

    def poll(self):
        """Return a completed result on the UI thread; cancelled work never applies."""
        if self._pending is None or not self._pending[1].done():
            return None
        eye, future = self._pending
        self._pending = None
        self.revision += 1
        if self._cancel.is_set():
            self.status = "Optimization cancelled; settings kept."
            return None
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - keep the editor open after worker failures
            self.status = f"Optimization failed: {exc}"
            return None
        self.results[eye] = result
        self.eye, self.index = eye, 0
        self.status = (
            f"Eye {eye} example agreement {result.before_score:.0%} -> {result.after_score:.0%}. "
            "Review each image; green = your pupil, yellow = prediction. Save calibration to keep."
        )
        return eye, result

    def cancel(self) -> None:
        if self.busy:
            self._cancel.set()
            self.status = "Cancelling optimization; settings kept."

    def close(self) -> None:
        self.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
