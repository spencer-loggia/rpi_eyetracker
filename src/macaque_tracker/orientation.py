"""Lossless camera orientation and raw-to-display geometry shared by every viewer.

ROI files keep sensor coordinates. Only crops used for analysis and recording,
and the lower-rate overview, are rotated; the stitched analysis frame is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import PixelRoi

DEFAULT_ROTATIONS = (0, 0, 0, 0)


def validate_rotations(values) -> tuple[int, ...]:
    """Validate clockwise quarter turns for cameras in physical left-to-right order."""
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ValueError("camera_rotations must contain four clockwise angles")
    if any(type(value) is not int or value not in (0, 90, 180, 270) for value in values):
        raise ValueError("Camera rotations must be 0, 90, 180, or 270 degrees")
    return tuple(values)


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Return a lossless clockwise rotation view; callers own any required copy."""
    return image if degrees % 360 == 0 else np.rot90(image, -(degrees // 90))


def rotated_size(width: int, height: int, degrees: int) -> tuple[int, int]:
    """Return image dimensions after a clockwise quarter turn."""
    return (height, width) if degrees % 180 else (width, height)


def rotate_roi(roi: PixelRoi, width: int, height: int, degrees: int) -> PixelRoi:
    """Rotate a half-open rectangle without off-by-one shifts at image boundaries."""
    turn = degrees % 360
    if turn == 90:
        return PixelRoi(height - roi.y2, roi.x, roi.height, roi.width)
    if turn == 180:
        return PixelRoi(width - roi.x2, height - roi.y2, roi.width, roi.height)
    if turn == 270:
        return PixelRoi(roi.y, width - roi.x2, roi.height, roi.width)
    return roi


@dataclass(frozen=True)
class CameraTile:
    """One physical camera's raw extent and rotated location in an overview."""

    source: PixelRoi
    destination: PixelRoi
    rotation: int


class CameraCanvas:
    """Arrange four oriented views in one row, padding below shorter views.

    Pixel aspect ratio is preserved. Geometry also works with small thumbnails
    whose width is not exactly divisible by four.
    """

    def __init__(self, width: int, height: int, rotations=DEFAULT_ROTATIONS) -> None:
        self.rotations = validate_rotations(rotations)
        self.raw_width, self.raw_height = width, height
        tiles = []
        left = 0
        for index, angle in enumerate(self.rotations):
            x, x2 = round(index * width / 4), round((index + 1) * width / 4)
            tile_width, tile_height = rotated_size(x2 - x, height, angle)
            tiles.append(CameraTile(
                PixelRoi(x, 0, x2 - x, height),
                PixelRoi(left, 0, tile_width, tile_height), angle,
            ))
            left += tile_width
        self.tiles = tuple(tiles)
        self.width = left
        self.height = max(tile.destination.height for tile in self.tiles)

    def compose(self, image: np.ndarray) -> np.ndarray:
        """Orient an overview; return the original array for the default orientation."""
        if image.shape[:2] != (self.raw_height, self.raw_width):
            raise ValueError("Overview dimensions do not match camera geometry")
        if not any(self.rotations):
            return image
        canvas = np.zeros((self.height, self.width, *image.shape[2:]), dtype=image.dtype)
        for tile in self.tiles:
            source, target = tile.source, tile.destination
            canvas[target.y:target.y2, target.x:target.x2] = rotate_image(
                image[source.y:source.y2, source.x:source.x2], tile.rotation,
            )
        return canvas

    def display_roi(self, roi: PixelRoi) -> PixelRoi:
        """Map a raw ROI to the overview; rotated ROIs must belong to one camera."""
        if not any(self.rotations):
            return roi
        for tile in self.tiles:
            if tile.source.x <= roi.x and roi.x2 <= tile.source.x2:
                local = PixelRoi(roi.x - tile.source.x, roi.y, roi.width, roi.height)
                rotated = rotate_roi(local, tile.source.width, tile.source.height, tile.rotation)
                return PixelRoi(rotated.x + tile.destination.x, rotated.y,
                                rotated.width, rotated.height)
        raise ValueError("An eye box must fit within one camera before rotating views")

    def source_point(self, x: int, y: int) -> tuple[int, int] | None:
        """Map a displayed pixel back to the sensor, rejecting padded areas."""
        for tile in self.tiles:
            target = tile.destination
            if target.x <= x < target.x2 and 0 <= y < target.height:
                point = rotate_roi(PixelRoi(x - target.x, y, 1, 1),
                                   target.width, target.height, -tile.rotation)
                return point.x + tile.source.x, point.y
        return None
