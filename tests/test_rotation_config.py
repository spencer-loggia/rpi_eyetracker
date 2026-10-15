"""Persistent orientation must preserve legacy raw-coordinate calibration files."""

import json

import pytest

from macaque_tracker.config import ConfigError, RoiLayout
from macaque_tracker.models import EyeImageSettings, NormalizedRoi


def eye_box(x=0.55, width=0.15):
    return NormalizedRoi(
        0, "eye_0", x, 0.2, width, 0.6,
        EyeImageSettings(gain=1.5, brightness=-0.1, pupil_size_bias=0.25),
    )


@pytest.mark.parametrize("shared", [False, True])
def test_camera_rotation_round_trip_preserves_raw_boxes_and_eye_tuning(tmp_path, shared):
    box = eye_box()
    layout = RoiLayout((box,), 800, 200, camera_rotations=[0, 90, 180, 270])
    path = layout.save(tmp_path / "calibration.json", shared=shared)
    assert layout.camera_rotations == (0, 90, 180, 270)
    loaded = RoiLayout.load(path)
    assert loaded == layout
    assert loaded.rois == (box,)
    assert loaded.rotation_for_roi(box) == 180
    assert json.loads(path.read_text())["camera_rotations"] == [0, 90, 180, 270]


def test_legacy_calibration_defaults_to_zero_rotation_without_altering_boxes(tmp_path):
    layout = RoiLayout((eye_box(x=0.2, width=0.3),), 800, 200)
    path = layout.save(tmp_path / "legacy.json")
    payload = json.loads(path.read_text())
    del payload["camera_rotations"]
    path.write_text(json.dumps(payload))
    loaded = RoiLayout.load(path)
    assert loaded == layout
    assert loaded.camera_rotations == (0, 0, 0, 0)
    # Legacy unrotated boxes may span camera boundaries.
    loaded.validate_for_frame(800, 200)
    assert loaded.rotation_for_roi(loaded.rois[0]) == 0


@pytest.mark.parametrize("angles", [
    None, "0,90,180,270", [], [0] * 3, [0] * 5, [0, 90.0, 180, 270],
    [True, 90, 180, 270], [0, "90", 180, 270], [0, -90, 180, 270],
    [0, 45, 180, 270], [0, 90, 180, 360],
])
def test_invalid_rotation_values_fail_in_memory_and_when_loading_json(tmp_path, angles):
    with pytest.raises(ConfigError, match="rotations|rotations must"):
        RoiLayout((eye_box(),), 800, 200, camera_rotations=angles)
    path = RoiLayout((eye_box(),), 800, 200).save(tmp_path / "invalid.json")
    payload = json.loads(path.read_text())
    payload["camera_rotations"] = angles
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="rotations|rotations must"):
        RoiLayout.load(path)


@pytest.mark.parametrize("camera", range(4))
def test_camera_ownership_uses_raw_coordinates_including_exact_camera_edges(camera):
    box = eye_box(x=camera / 4, width=0.25)
    layout = RoiLayout((box,), 800, 200, camera_rotations=(0, 90, 180, 270))
    assert layout.rotation_for_roi(box) == camera * 90
    layout.validate_for_frame(800, 200)


def test_rotated_layout_validation_rejects_roi_spanning_two_camera_views():
    box = eye_box(x=0.2, width=0.1)
    layout = RoiLayout((box,), 800, 200, camera_rotations=(0, 90, 0, 0))
    with pytest.raises(ConfigError, match="single camera"):
        layout.rotation_for_roi(box)
    with pytest.raises(ConfigError, match="single camera"):
        layout.validate_for_frame(800, 200)
