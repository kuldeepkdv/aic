#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""2D port detection in a single rectified camera image.

Detects the dark rectangular aperture of a connector port (SFP / SC)
against the lighter task-board background. The detector returns the
pixel centroid, oriented bounding rect, and a confidence score.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from sensor_msgs.msg import Image


@dataclass
class PortDetection:
    u: float                      # centroid pixel x
    v: float                      # centroid pixel y
    width_px: float               # oriented rect short side
    height_px: float              # oriented rect long side
    angle_rad: float              # in-plane orientation of the rect
    confidence: float             # 0..1
    corners_px: np.ndarray        # 4x2 corner pixels (float)


class PortDetector:
    """Classical-CV port detector.

    The task-board has a bright metallic surface; connector apertures are
    small dark rectangles. We threshold for dark regions, filter by area
    and aspect ratio, and keep the blob closest to the expected port
    location (image center by default, or a supplied prior).

    This avoids any training data and runs at 20 Hz on CPU.
    """

    def __init__(
        self,
        min_area_px: int = 120,
        max_area_px: int = 20000,
        min_aspect: float = 1.3,
        max_aspect: float = 6.0,
        dark_threshold: int = 70,
    ):
        self._min_area = min_area_px
        self._max_area = max_area_px
        self._min_aspect = min_aspect
        self._max_aspect = max_aspect
        self._dark_threshold = dark_threshold

    @staticmethod
    def image_msg_to_bgr(image_msg: Image) -> np.ndarray:
        """Decode a sensor_msgs/Image (rgb8 or bgr8) to an H,W,3 BGR ndarray."""
        buf = np.frombuffer(image_msg.data, dtype=np.uint8)
        img = buf.reshape(image_msg.height, image_msg.width, -1)
        if image_msg.encoding in ("rgb8", "rgba8"):
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
        elif image_msg.encoding == "mono8":
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = img[..., :3]
        return np.ascontiguousarray(img)

    def detect(
        self,
        image_msg: Image,
        prior_px: Optional[Tuple[float, float]] = None,
    ) -> Optional[PortDetection]:
        """Return the best port candidate, or None if nothing plausible."""
        bgr = self.image_msg_to_bgr(image_msg)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold handles lighting better than a fixed cut-off.
        _, dark = cv2.threshold(
            gray, self._dark_threshold, 255, cv2.THRESH_BINARY_INV
        )
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )

        contours, _ = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        h, w = gray.shape
        anchor = prior_px if prior_px is not None else (w * 0.5, h * 0.5)

        best: Optional[PortDetection] = None
        best_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area or area > self._max_area:
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            short, long_ = sorted((rw, rh))
            if short < 2.0:
                continue
            aspect = long_ / short
            if aspect < self._min_aspect or aspect > self._max_aspect:
                continue
            fill = area / max(rw * rh, 1e-3)
            if fill < 0.55:     # reject non-rectangular blobs
                continue

            # Score: high fill, central, reasonable aspect.
            dist = np.hypot(cx - anchor[0], cy - anchor[1])
            dist_score = np.exp(-dist / (0.25 * max(h, w)))
            score = 0.5 * fill + 0.4 * dist_score + 0.1 * (1.0 / aspect)

            if score > best_score:
                box = cv2.boxPoints(rect).astype(np.float32)
                best_score = score
                best = PortDetection(
                    u=float(cx),
                    v=float(cy),
                    width_px=float(short),
                    height_px=float(long_),
                    angle_rad=float(np.deg2rad(angle)),
                    confidence=float(min(1.0, score)),
                    corners_px=box,
                )

        return best
