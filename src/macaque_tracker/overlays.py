from __future__ import annotations

from typing import Any

import numpy as np

from .models import PixelRoi


def draw_roi_box(
    canvas: np.ndarray,
    roi: PixelRoi,
    eye_id: int,
    *,
    cv2_module: Any,
    scale: float = 1.0,
    active: bool = False,
) -> None:
    """Draw the same eye-region overlay in configuration and live previews."""

    color = (0, 215, 255) if active else ((80, 220, 80), (220, 90, 220))[eye_id]
    x1 = round(roi.x * scale)
    y1 = round(roi.y * scale)
    x2 = round(roi.x2 * scale)
    y2 = round(roi.y2 * scale)
    cv2_module.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2_module.putText(
        canvas,
        f"eye {eye_id}",
        (x1 + 4, max(18, y1 - 5)),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2_module.LINE_AA,
    )
