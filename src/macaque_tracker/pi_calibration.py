"""Embedded adapter for the same ROI and pupil-tuning editors used by configure."""

from __future__ import annotations

from dataclasses import replace

from .config import RoiLayout
from .models import EyeImageSettings, NormalizedRoi
from .roi_tool import EyeTuningEditor, RoiEditor, _load_existing_layout, _pupil_size_bias_slider


class CalibrationSession:
    def __init__(self, config, experiment, feed) -> None:
        self.config = config
        self.experiment = experiment
        self.feed = feed
        image = feed.latest()
        boxes, self.initial_settings = _load_existing_layout(
            experiment.calibration_path,
            image,
            config.tracker,
        )
        self.roi = RoiEditor(
            image,
            boxes,
            camera_config=config.camera,
            control_limits=feed.camera.image_control_limits(),
            frame_source=feed.latest,
            apply_exposure=lambda camera: feed.camera.set_image_controls(
                exposure_us=camera.exposure_us,
            ),
            maximum_display_width=1280,
            maximum_display_height=600,
        )
        self.tuning = None
        self.phase = "boxes"
        self.selected = ()

    @property
    def editor(self):
        return self.roi if self.phase == "boxes" else self.tuning

    def render(self):
        editor = self.editor
        if self.phase == "boxes":
            editor._refresh_live_image()
        else:
            editor._refresh_live_crops()
        return editor._frame()

    def mouse(self, event: int, x: int, y: int) -> None:
        if self.phase == "boxes":
            self.roi._mouse(event, x, y, 0, None)

    def next(self) -> None:
        if not 1 <= len(self.roi.boxes) <= 2:
            raise ValueError("Draw one or two eye boxes before continuing")
        if self.roi.camera_config != self.roi.applied_camera_config:
            raise ValueError("Exposure was not applied; move its slider to retry")
        self.selected = tuple(self.roi.boxes)

        def crops():
            image = self.feed.latest()
            return tuple(box.extract(image) for box in self.selected)

        initial = tuple(
            self.initial_settings.get(
                index,
                EyeImageSettings(pupil_size_bias=self.config.tracker.pupil_size_bias),
            )
            for index in range(len(self.selected))
        )
        self.tuning = EyeTuningEditor(
            crops(),
            initial,
            self.config.tracker,
            frame_source=crops,
            maximum_display_width=1280,
            maximum_display_height=540,
        )
        self.phase = "tuning"

    def previous(self) -> None:
        if self.tuning is not None:
            self.initial_settings = dict(enumerate(self.tuning.settings))
        self.phase = "boxes"

    def sliders(self):
        """Return label, shared slider mapping, value, callback for toolkit controls."""
        if self.phase == "boxes":
            slider = self.roi._exposure
            if slider is not None:
                yield (
                    slider.label,
                    slider,
                    self.roi.camera_config.exposure_us,
                    self.roi._exposure_changed,
                )
        else:
            for index, settings in enumerate(self.tuning.settings):
                bias = _pupil_size_bias_slider()
                yield (
                    f"Eye {index}: pupil size bias",
                    bias,
                    settings.pupil_size_bias,
                    lambda position, i=index: self.tuning._size_bias_changed(i, position),
                )
                for slider in self.tuning._sliders:
                    yield (
                        f"Eye {index}: {slider.config_name}",
                        slider,
                        getattr(settings, slider.config_name),
                        lambda position, i=index, s=slider: self.tuning._software_changed(
                            i, s, position
                        ),
                    )

    def save(self) -> None:
        if self.phase != "tuning" or self.tuning is None:
            raise ValueError("Complete pupil tuning before saving calibration")
        height, width = self.roi.image.shape
        layout = RoiLayout(
            rois=tuple(
                NormalizedRoi.from_pixels(
                    eye_id=index,
                    label=f"eye_{index}",
                    roi=box,
                    frame_width=width,
                    frame_height=height,
                    settings=self.tuning.settings[index],
                )
                for index, box in enumerate(self.selected)
            ),
            source_width=width,
            source_height=height,
        )
        layout.validate_for_frame(width, height, tracker_config=self.config.tracker)
        updated = replace(self.config, camera=self.roi.applied_camera_config)
        # Config/ROI writers use atomic replacement; only the current experiment
        # is touched. Returning with Back instead discards unsaved adjustments.
        updated.save(self.experiment.config_path, shared=True)
        layout.save(self.experiment.calibration_path, shared=True)
