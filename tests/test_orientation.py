"""Pixel-level tests for orientation independent of OpenCV and tracker fitting."""

from itertools import product

import numpy as np
import pytest

from macaque_tracker.models import PixelRoi
from macaque_tracker.orientation import (
    CameraCanvas,
    rotate_image,
    rotate_roi,
    rotated_size,
    validate_rotations,
)

ANGLES = (0, 90, 180, 270)


def reference_rotation(image, angle):
    """Place individual pixels using clockwise corner geometry, without rot90."""
    height, width = image.shape[:2]
    shape = (width, height) if angle in (90, 270) else (height, width)
    result = np.empty((*shape, *image.shape[2:]), dtype=image.dtype)
    for y, x in product(range(height), range(width)):
        target_x, target_y = {
            0: (x, y),
            90: (height - 1 - y, x),
            180: (width - 1 - x, height - 1 - y),
            270: (y, width - 1 - x),
        }[angle]
        result[target_y, target_x] = image[y, x]
    return result


@pytest.mark.parametrize("width", [4, 5, 7, 13, 20, 23])
def test_all_camera_rotation_combinations_preserve_every_pixel_and_inverse_mapping(width):
    """Cover all 256 mixed orientations, including uneven thumbnail partitions."""
    height = 3
    image = np.arange(1, height * width + 1, dtype=np.uint16).reshape(height, width)
    # Unique nonzero pixels distinguish padding from real image pixels.
    for angles in product(ANGLES, repeat=4):
        canvas = CameraCanvas(width, height, angles)
        composed = canvas.compose(image)
        assert composed.dtype == image.dtype
        assert composed.shape == (canvas.height, canvas.width)
        assert sorted(composed[composed != 0].tolist()) == image.reshape(-1).tolist()
        left = 0
        for index, angle in enumerate(angles):
            start, end = round(index * width / 4), round((index + 1) * width / 4)
            expected = reference_rotation(image[:, start:end], angle)
            np.testing.assert_array_equal(
                composed[:expected.shape[0], left:left + expected.shape[1]], expected,
            )
            left += expected.shape[1]
        assert left == canvas.width
        for y, x in product(range(canvas.height), range(canvas.width)):
            source = canvas.source_point(x, y)
            if composed[y, x] == 0:
                assert source is None
            else:
                assert source is not None
                raw_x, raw_y = source
                assert 0 <= raw_x < width and 0 <= raw_y < height
                assert composed[y, x] == image[raw_y, raw_x]
        for x, y in [(-1, 0), (0, -1), (canvas.width, 0), (0, canvas.height)]:
            assert canvas.source_point(x, y) is None


@pytest.mark.parametrize("angle", ANGLES)
def test_rotate_image_and_every_rectangle_agree_with_pixel_geometry(angle):
    image = np.arange(35, dtype=np.uint8).reshape(5, 7)
    rotated = rotate_image(image, angle)
    np.testing.assert_array_equal(rotated, reference_rotation(image, angle))
    assert np.shares_memory(rotated, image)
    assert rotated_size(7, 5, angle) == (rotated.shape[1], rotated.shape[0])
    if angle == 0:
        assert rotated is image
    for x, y in product(range(7), range(5)):
        for x2, y2 in product(range(x + 1, 8), range(y + 1, 6)):
            source = PixelRoi(x, y, x2 - x, y2 - y)
            target = rotate_roi(source, 7, 5, angle)
            np.testing.assert_array_equal(
                target.extract(rotated), reference_rotation(source.extract(image), angle),
            )
            assert rotate_roi(target, rotated.shape[1], rotated.shape[0], -angle) == source


def test_display_roi_corresponds_to_rotated_region_in_every_camera():
    image = np.arange(115, dtype=np.uint16).reshape(5, 23)
    for angles in product(ANGLES, repeat=4):
        canvas = CameraCanvas(23, 5, angles)
        composed = canvas.compose(image)
        for tile in canvas.tiles:
            for box in (
                tile.source,
                PixelRoi(tile.source.x + 1, 1, tile.source.width - 2, 3),
                PixelRoi(tile.source.x2 - 1, 4, 1, 1),
            ):
                displayed = canvas.display_roi(box)
                np.testing.assert_array_equal(
                    displayed.extract(composed),
                    reference_rotation(box.extract(image), tile.rotation),
                )


def test_color_overview_rotates_without_interpolating_or_mixing_channels():
    image = np.arange(3 * 20 * 3, dtype=np.uint8).reshape(3, 20, 3)
    canvas = CameraCanvas(20, 3, ANGLES)
    actual = canvas.compose(image)
    assert actual.shape == (5, 16, 3)
    for tile in canvas.tiles:
        source, target = tile.source, tile.destination
        np.testing.assert_array_equal(
            actual[:target.height, target.x:target.x2],
            reference_rotation(image[:, source.x:source.x2], tile.rotation),
        )
        assert not actual[target.height:, target.x:target.x2].any()


def test_default_orientation_returns_original_overview_and_cross_camera_roi():
    image = np.zeros((3, 13), dtype=np.uint8)
    canvas = CameraCanvas(13, 3)
    box = PixelRoi(1, 0, 10, 3)
    assert canvas.compose(image) is image
    assert canvas.display_roi(box) is box


def test_rotated_overview_rejects_cross_camera_boxes_and_wrong_image_size():
    canvas = CameraCanvas(20, 3, ANGLES)
    with pytest.raises(ValueError, match="one camera"):
        canvas.display_roi(PixelRoi(4, 0, 3, 2))
    with pytest.raises(ValueError, match="dimensions"):
        canvas.compose(np.zeros((3, 21), dtype=np.uint8))


@pytest.mark.parametrize("values", [
    None, 90, "0,90,180,270", [], [0] * 3, [0] * 5,
    [False, 0, 0, 0], [0.0, 90, 180, 270], [0, "90", 180, 270],
    [0, -90, 180, 270], [0, 45, 180, 270], [0, 90, 180, 360],
])
def test_invalid_camera_rotations_are_rejected(values):
    with pytest.raises(ValueError):
        validate_rotations(values)
    with pytest.raises(ValueError):
        CameraCanvas(20, 3, values)


def test_camera_rotations_are_copied_to_an_immutable_tuple():
    angles = list(ANGLES)
    canvas = CameraCanvas(20, 3, angles)
    angles[0] = 180
    assert canvas.rotations == ANGLES

